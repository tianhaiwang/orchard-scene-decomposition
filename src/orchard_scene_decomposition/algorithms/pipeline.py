"""Stage implementations of the decomposition pipeline.

Point labels: 0 unassigned, 1 ground, 2 poles, 3 wires, >= 4 tree instances.
Unassigned reasons: 1 outside ROI, 2 denoise, 3 unclustered layer point,
4 small fragment, 5 unassigned fragment, 6 other.
Label provenance: 0 automatic, 1 manual edit, 2 refill.
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
import pwlf


def concatenate_indices(groups):
    groups = [np.asarray(group, dtype=np.int64) for group in groups if len(group)]
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)


def make_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def group_indices_by_label(labels, noise_label=-1):
    ids = np.flatnonzero(labels != noise_label)
    ordered_ids = ids[np.argsort(labels[ids], kind="stable")]
    if not len(ordered_ids):
        return []
    return np.split(ordered_ids, np.flatnonzero(np.diff(labels[ordered_ids])) + 1)


def cluster_dbscan(points, eps, min_points):
    if not len(points):
        return []
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(points)
    return sorted(group_indices_by_label(labels), key=len, reverse=True)


def cluster_endpoints(xyz, ids, margin):
    selected_points = xyz[ids]
    heights = selected_points[:, 2]
    return (
        selected_points[heights > heights.max() - margin].mean(axis=0),
        selected_points[heights < heights.min() + margin].mean(axis=0),
    )


def cluster_euclidean(points, tolerance, min_cluster_size, max_cluster_size):
    """Fixed-radius region growing; cluster size limits are exclusive."""
    if not len(points):
        return []
    spatial_index = cKDTree(points)
    visited = np.zeros(len(points), bool)
    clusters = []
    for point_index in range(len(points)):
        if visited[point_index]:
            continue
        visited[point_index] = True
        queue = [point_index]
        queue_position = 0
        while queue_position < len(queue):
            neighbor_indices = np.asarray(
                spatial_index.query_ball_point(
                    points[queue[queue_position]], tolerance
                ),
                dtype=np.int64,
            )
            unvisited_indices = neighbor_indices[~visited[neighbor_indices]]
            visited[unvisited_indices] = True
            queue.extend(unvisited_indices.tolist())
            queue_position += 1
        if min_cluster_size < len(queue) < max_cluster_size:
            clusters.append(np.asarray(queue, dtype=np.int64))
    return clusters


def distances_to_axis(points, endpoints, segment=False):
    start_point, end_point = endpoints
    axis = end_point - start_point
    length = np.linalg.norm(axis)
    if length <= np.finfo(float).eps:
        raise ValueError("Coincident endpoints cannot define an axis")
    axis = axis / length
    axis_positions = (points - start_point) @ axis
    distances = np.linalg.norm(
        points - (start_point + axis_positions[:, None] * axis), axis=1
    )
    if segment:
        distances[(axis_positions < 0) | (axis_positions > length)] = np.inf
    return distances, axis


def local_pca(reference, queries, radius, min_points, return_directions=False):
    """Linearity, verticality and neighbour count around each query point."""
    tree = cKDTree(reference)
    linearity = np.zeros(len(queries))
    verticality = np.zeros(len(queries))
    neighbor_counts = np.zeros(len(queries), int)
    largest_eigenvalues = np.zeros(len(queries))
    directions = np.zeros((len(queries), 3)) if return_directions else None
    for query_index, query_point in enumerate(queries):
        neighbor_indices = tree.query_ball_point(query_point, radius)
        neighbor_counts[query_index] = len(neighbor_indices)
        if len(neighbor_indices) < min_points:
            continue
        neighbors = reference[neighbor_indices]
        covariance = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest_eigenvalues[query_index] = eigenvalues[-1]
        if return_directions:
            directions[query_index] = eigenvectors[:, -1]
        if eigenvalues[-1] > 0:
            linearity[query_index] = (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[
                -1
            ]
        verticality[query_index] = abs(eigenvectors[2, -1])
    result = (linearity, verticality, neighbor_counts, largest_eigenvalues)
    return result + (directions,) if return_directions else result


def run_stage(project, stage, log=print):
    """Run one stage on the outputs of its predecessor and save a revision."""
    from ..config import validate

    validate(project.head["config"])
    project.prepare(stage)
    seed = project.head["config"]["load"]["seed"]
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    handler = globals()["stage_" + stage]
    log("Starting " + stage)
    handler(project, project.head["config"][stage], log)
    project.head["completed"][stage] = "pending"
    arrays = project.arrays
    report = project.head["diagnostics"].setdefault(stage, {})
    report["label_counts"] = {
        str(array_name): int(point_count)
        for array_name, point_count in zip(
            *np.unique(arrays["labels"], return_counts=True)
        )
    }
    if stage in ("poles", "wires"):
        project.finish_facility_round(stage, report)
    project.save("completed " + stage)
    log("Finished " + stage)
    return report


def stage_load(project, config, log):
    vertices = project.vertices
    xyz = np.column_stack(
        [vertices[field_name] for field_name in ("x", "y", "z")]
    ).astype(float)
    center = xyz.mean(axis=0)
    centered_xyz = xyz - center
    rotation = np.eye(3)
    if config["alignment"] != "none":
        alignment_dimensions = 3 if config["alignment"] == "xyz" else 2
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.cov(centered_xyz[:, :alignment_dimensions].T)
        )
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        if eigenvectors[0, 0] < 0:
            eigenvectors[:, 0] *= -1
        if alignment_dimensions == 3 and eigenvectors[1, 1] < 0:
            eigenvectors[:, 1] *= -1
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        rotation[:alignment_dimensions, :alignment_dimensions] = eigenvectors
    aligned_xyz = centered_xyz @ rotation + center
    point_cloud = make_point_cloud(xyz)
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config["normal_radius"], max_nn=config["normal_max_neighbors"]
        )
    )
    normals = np.asarray(point_cloud.normals)
    if config["normal_coordinate_frame"] == "work":
        normals = normals @ rotation
    point_count = len(xyz)
    project.arrays = dict(
        xyz=aligned_xyz,
        normals=normals,
        rotation=rotation,
        center=center,
        labels=np.zeros(point_count, "i4"),
        auto_labels=np.zeros(point_count, "i4"),
        fragment_id=np.full(point_count, -1, "i4"),
        reason=np.zeros(point_count, "u1"),
        manual_override=np.zeros(point_count, "u1"),
        provenance=np.zeros(point_count, "u1"),
        active=np.arange(point_count, dtype=np.int64),
    )
    project.head["diagnostics"] = {
        "load": dict(
            points=point_count,
            transform="(source - center) @ rotation + center",
            normal_coordinate_frame=config["normal_coordinate_frame"],
        )
    }


def stage_ground(project, config, log):
    arrays = project.arrays
    xyz = arrays["xyz"]
    point_count = len(xyz)
    point_ids = np.arange(point_count)
    if config["method"] == "skip":
        ground_ids = np.empty(0, dtype=np.int64)
    elif config["method"] == "csf":
        import CSF

        csf = CSF.CSF()
        for parameter_name in [
            "bSloopSmooth",
            "cloth_resolution",
            "rigidness",
            "time_step",
            "class_threshold",
        ]:
            setattr(csf.params, parameter_name, config[parameter_name])
        csf.params.interations = config["iterations"]
        csf.setPointCloud(xyz)
        ground_indices = CSF.VecInt()
        nonground_indices = CSF.VecInt()
        csf.do_filtering(ground_indices, nonground_indices, False)
        ground_ids = np.asarray(ground_indices, dtype=np.int64)
    else:
        remaining_ids = point_ids
        ground_plane = None
        for plane_index in range(config["max_planes"]):
            if len(remaining_ids) < config["ransac_n"]:
                break
            plane_coefficients, plane_inlier_indices = make_point_cloud(
                xyz[remaining_ids]
            ).segment_plane(
                config["distance_threshold"],
                config["ransac_n"],
                config["num_iterations"],
            )
            plane_coefficients = np.asarray(plane_coefficients)
            plane_coefficients /= np.linalg.norm(plane_coefficients[:3])
            if abs(plane_coefficients[2]) > config["min_ground_normal_z"]:
                if plane_coefficients[2] < 0:
                    plane_coefficients = -plane_coefficients
                ground_plane = plane_coefficients
                break
            remaining_ids = np.delete(remaining_ids, plane_inlier_indices)
        if ground_plane is None:
            raise ValueError(
                "No ground plane meets the normal threshold. Adjust parameters or select skip"
            )
        ground_ids = np.flatnonzero(
            xyz @ ground_plane[:3] + ground_plane[3] < config["max_height_above_ground"]
        )
        arrays["ground_plane"] = ground_plane
    arrays["labels"][ground_ids] = 1
    arrays["ground"] = ground_ids
    arrays["active"] = np.setdiff1d(point_ids, ground_ids)
    project.head["diagnostics"]["ground"] = dict(
        ground=len(ground_ids), remaining=len(arrays["active"]), method=config["method"]
    )


class FacilityPairError(ValueError):
    def __init__(self, message, status="error"):
        super().__init__(message)
        self.status = status


def picked_pairs(project, stage):
    ids = project.head["picks"][stage]
    if not ids or len(ids) % 2:
        raise ValueError(
            "Pick endpoint pairs in " + stage + " view, or disable this stage"
        )
    return [ids[i : i + 2] for i in range(0, len(ids), 2)]


def validate_facility_pair(project, ids, active_mask):
    if any(type(i) is not int or i < 0 or i >= len(active_mask) for i in ids):
        raise FacilityPairError("Invalid endpoint row IDs. Pick this pair again")
    pair = project.arrays["xyz"][ids]
    if not np.isfinite(pair).all():
        raise FacilityPairError("Non-finite endpoint coordinates. Pick this pair again")
    if not active_mask[ids].all():
        raise FacilityPairError(
            "Endpoints were removed by a previous stage. Pick this pair again"
        )
    length = np.linalg.norm(pair[1] - pair[0])
    if not np.isfinite(length) or length <= np.finfo(float).eps:
        raise FacilityPairError(
            "Coincident endpoints or invalid axis. Pick this pair again"
        )
    return pair


def require_facility_points(ids, message):
    if not len(ids):
        raise FacilityPairError(message, status="warning")


def require_finite_features(features):
    if not np.isfinite(features).all():
        raise FacilityPairError("Cannot fit non-finite object features")


def facility_stage(project, config, stage, prefix, label, extract, log):
    """Extract poles or wires around each picked endpoint pair.

    Pairs fail independently; the stage is saved when at least one succeeds.
    Append rounds keep previously extracted objects.
    """
    arrays = project.arrays
    active_ids = arrays["active"]
    pairs = picked_pairs(project, stage) if config["enabled"] else []
    context = getattr(project, "_facility_run", None)
    if context and context.get("stage") != stage:
        context = None
    append_round = bool(context and context["mode"] == "append")
    if append_round and not config["enabled"]:
        raise ValueError(
            "Cannot disable extraction in an append round. Start a replacement round to clear facilities"
        )
    previous_ids = (
        arrays.get(stage, np.empty(0, dtype=np.int64))
        if append_round
        else np.empty(0, dtype=np.int64)
    )
    previous_axes = (
        arrays.get(prefix + "_axes", np.empty((0, 2, 3)))
        if append_round
        else np.empty((0, 2, 3))
    )
    previous_axis_pair_numbers = (
        arrays.get(prefix + "_axis_pair_numbers", np.arange(1, len(previous_axes) + 1))
        if append_round
        else np.empty(0, dtype=np.int64)
    )
    previous_diagnostics = (
        project.head["diagnostics"].get(stage, {}) if append_round else {}
    )
    existing_objects = [
        int(key.split("_")[1]) + 1
        for key in arrays
        if append_round and key.startswith(prefix + "_") and key.split("_")[1].isdigit()
    ]
    object_offset = (
        max(
            [
                previous_diagnostics.get(
                    "next_object_offset", previous_diagnostics.get("total_objects", 0)
                ),
                (
                    int(np.max(previous_axis_pair_numbers))
                    if len(previous_axis_pair_numbers)
                    else 0
                ),
            ]
            + existing_objects
        )
        if append_round
        else 0
    )
    active_mask = np.zeros(len(arrays["xyz"]), bool)
    active_mask[active_ids] = True
    selected_groups = []
    axes = []
    outputs = {}
    diagnostics = []
    for pair_index, ids in enumerate(pairs):
        object_index = object_offset + pair_index
        object_diagnostics = dict(
            object=object_index,
            pair_number=pair_index + 1,
            endpoint_ids=list(ids),
            endpoints=None,
            status="error",
            reason="",
            selected_count=0,
        )
        if context:
            object_diagnostics["round_number"] = context["number"]
        diagnostics.append(object_diagnostics)
        try:
            pair = validate_facility_pair(project, ids, active_mask)
            object_diagnostics["endpoints"] = pair.tolist()
            selected_ids, result, _ = extract(
                project, config, object_index, pair, object_diagnostics
            )
            require_facility_points(
                selected_ids, "No points survived filtering. Check picks and parameters"
            )
        except FacilityPairError as exc:
            object_diagnostics.update(status=exc.status, reason=str(exc))
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            object_diagnostics.update(
                status="error", reason="Numerical fit failed: " + str(exc)
            )
        else:
            selected_groups.append(selected_ids)
            axes.append(pair)
            outputs.update(result)
            object_diagnostics.update(
                status="ok", reason="", selected_count=len(selected_ids)
            )
        if object_diagnostics["status"] == "ok":
            log(
                stage
                + " pair "
                + str(pair_index + 1)
                + "/"
                + str(len(pairs))
                + " completed"
            )
        else:
            log(
                "["
                + object_diagnostics["status"].upper()
                + "] "
                + stage
                + " pair "
                + str(pair_index + 1)
                + " endpoints "
                + str(ids)
                + ": "
                + object_diagnostics["reason"]
            )
        if object_diagnostics.get("adaptive_fallback_reason"):
            log(
                "[WARNING] "
                + stage
                + " pair "
                + str(pair_index + 1)
                + ": adaptive correction unavailable; kept original fit: "
                + object_diagnostics["adaptive_fallback_reason"]
            )
    warnings = sum(diagnostic["status"] == "warning" for diagnostic in diagnostics)
    errors = sum(diagnostic["status"] == "error" for diagnostic in diagnostics)
    added_ids = np.unique(concatenate_indices(selected_groups))
    cumulative_ids = np.union1d(previous_ids, added_ids).astype(np.int64)
    report = dict(
        objects=diagnostics,
        points=len(cumulative_ids),
        points_added=len(added_ids),
        cumulative_points=len(cumulative_ids),
        object_offset=object_offset,
        next_object_offset=object_offset + len(pairs),
        total_objects=len(pairs),
        succeeded_objects=len(selected_groups),
        warning_objects=warnings,
        error_objects=errors,
        failed_objects=warnings + errors,
        status=(
            "disabled"
            if not config["enabled"]
            else "partial" if warnings + errors else "ok"
        ),
    )
    if context:
        report.update(
            round_number=context["number"],
            round_mode=context["mode"],
            base_revision=context["base_revision"],
        )
    if pairs and not selected_groups:
        report["status"] = "failed"
        detail = "; ".join(
            "Pair " + str(i["pair_number"]) + ": " + i["reason"] for i in diagnostics
        )
        error = ValueError(
            stage
            + ": all "
            + str(len(pairs))
            + " pairs failed. Stage was not saved. "
            + detail
        )
        error.diagnostics = report
        raise error
    if not append_round:
        for key in list(arrays):
            if key.startswith(prefix + "_"):
                del arrays[key]
    arrays.update(outputs)
    arrays[prefix + "_axes"] = np.concatenate(
        [previous_axes, np.asarray(axes, dtype=float).reshape(-1, 2, 3)]
    )
    arrays[prefix + "_axis_pair_numbers"] = np.concatenate(
        [
            previous_axis_pair_numbers,
            np.asarray(
                [i["object"] + 1 for i in diagnostics if i["status"] == "ok"],
                dtype=np.int64,
            ),
        ]
    )
    arrays[stage] = cumulative_ids
    arrays["labels"][added_ids] = label
    arrays["active"] = np.setdiff1d(active_ids, added_ids)
    project.head["diagnostics"][stage] = report
    if warnings + errors:
        log(
            "[WARNING] "
            + stage
            + ": "
            + str(len(selected_groups))
            + " pairs succeeded; "
            + str(warnings + errors)
            + " pairs failed. Successful results will be saved"
        )


def stage_poles(project, config, log):
    facility_stage(project, config, "poles", "pole", 2, extract_pole, log)


def extract_pole(project, config, pair_index, pair, diagnostics):
    """Cylinder search, feature clustering, cross-section filter and voting."""
    arrays = project.arrays
    active_ids = arrays["active"]
    xyz = arrays["xyz"]
    result = {}
    prefix = "pole_" + str(pair_index)
    axis_distances_all, axis = distances_to_axis(
        xyz[active_ids], pair, config["limit_to_segment"]
    )
    candidate_ids = active_ids[axis_distances_all <= config["search_radius"]]
    axis_distances = axis_distances_all[axis_distances_all <= config["search_radius"]]
    diagnostics["candidate_count"] = len(candidate_ids)
    result[prefix + "_candidate_ids"] = candidate_ids
    require_facility_points(candidate_ids, "No points inside the search cylinder")
    features = np.column_stack(
        [
            axis_distances,
            project.vertices["opacity"][candidate_ids],
            project.vertices["intensity"][candidate_ids],
            arrays["normals"][candidate_ids],
        ]
    )
    require_finite_features(features)
    fused = MinMaxScaler().fit_transform(features) * np.asarray(
        config["feature_weights"]
    )
    clusters = cluster_dbscan(
        fused, config["feature_cluster_epsilon"], config["feature_cluster_min_points"]
    )
    if not clusters:
        raise FacilityPairError(
            "No feature clusters. Check endpoints or clustering parameters",
            status="warning",
        )
    seed_axis_distances = axis_distances[clusters[0]]
    axis_distance_threshold = float(
        seed_axis_distances.mean()
        + config["sigma_multiplier"] * seed_axis_distances.std()
    )
    filtered_ids = active_ids[axis_distances_all <= axis_distance_threshold]
    diagnostics.update(
        radius=axis_distance_threshold,
        feature_seed=len(clusters[0]),
        axis_filtered_count=len(filtered_ids),
    )
    result[prefix + "_axis_filtered_ids"] = filtered_ids.copy()
    require_facility_points(filtered_ids, "No points survived axis-distance filtering")
    if config["cross_section_filter_enabled"]:
        candidate_points = xyz[filtered_ids]
        z = candidate_points[:, 2]
        section_midpoint = (z.min() + z.max()) / 2
        midpts = candidate_points[
            np.abs(z - section_midpoint) <= config["cross_section_half_height"]
        ]
        if len(midpts) > config["cross_section_min_points"]:
            temp = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
            perp1 = np.cross(axis, temp)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)
            reference_points = pair[0] + (midpts.mean(axis=0) - pair[0]) @ axis * axis
            mid2d = np.column_stack(
                [
                    (midpts - reference_points) @ perp1,
                    (midpts - reference_points) @ perp2,
                ]
            )
            clusters2 = cluster_dbscan(
                mid2d,
                config["cross_section_cluster_epsilon"],
                config["cross_section_cluster_min_points"],
            )
            if len(clusters2) > 1:
                main = mid2d[clusters2[0]]
                center = main.mean(axis=0)
                all2d = np.column_stack(
                    [
                        (candidate_points - reference_points) @ perp1,
                        (candidate_points - reference_points) @ perp2,
                    ]
                )
                radius = np.percentile(
                    np.linalg.norm(main, axis=1),
                    config["cross_section_radius_percentile"],
                )
                filtered_ids = filtered_ids[
                    np.linalg.norm(all2d - center, axis=1)
                    < radius * config["cross_section_radius_multiplier"]
                ]
    result[prefix + "_cross_section_filtered_ids"] = filtered_ids.copy()
    diagnostics["cross_section_filtered_count"] = len(filtered_ids)
    require_facility_points(filtered_ids, "No points survived cross-section filtering")
    if config["point_filter_enabled"]:
        candidate_points = xyz[filtered_ids]
        axis_distances, _ = distances_to_axis(candidate_points, pair)
        _, vertical, _, _ = local_pca(
            candidate_points,
            candidate_points,
            config["local_radius"],
            config["local_min_points"],
        )
        normal_dot = np.abs(arrays["normals"][filtered_ids] @ axis)
        median = np.median(axis_distances)
        mad = np.median(np.abs(axis_distances - median))
        votes = (
            (axis_distances < median + config["mad_multiplier"] * mad).astype(int)
            + (vertical > config["verticality_min"]).astype(int)
            + (normal_dot < config["max_normal_axis_dot"]).astype(int)
        )
        filtered_ids = filtered_ids[votes >= config["min_feature_votes"]]
    require_facility_points(filtered_ids, "No points survived feature voting")
    result[prefix + "_selected_ids"] = filtered_ids
    return filtered_ids, result, features


def fit_piecewise_curve(x, y, config, seed):
    if len(x) < config["max_fit_segments"] + 2 or np.ptp(x) <= np.finfo(float).eps:
        raise FacilityPairError(
            "Too few wire fit points or zero work-X span. Adjust search, features or alignment"
        )
    angle_valid_model = None
    last_candidate_model = None
    for segment_count in range(
        config["min_fit_segments"], config["max_fit_segments"] + 1
    ):
        model = pwlf.PiecewiseLinFit(x, y, seed=seed)
        model.fit(
            segment_count,
            maxiter=config["fit_max_iterations"],
            popsize=config["fit_population_multiplier"],
        )
        if not all(
            np.isfinite(value).all()
            for value in (model.slopes, model.fit_breaks, model.beta)
        ):
            raise FacilityPairError(
                "Non-finite wire fit parameters. Check this pair or fit settings"
            )
        last_candidate_model = model
        if np.all(
            np.abs(np.diff(np.degrees(np.arctan(model.slopes))))
            <= config["max_bend_angle_deg"]
        ):
            angle_valid_model = model
    if config["fit_strategy"] == "last_candidate":
        return last_candidate_model
    if angle_valid_model is None:
        raise FacilityPairError(
            "All wire fits failed the angle threshold. Adjust fit settings"
        )
    return angle_valid_model


def stage_wires(project, config, log):
    facility_stage(project, config, "wires", "wire", 3, extract_wire, log)


def piecewise_curve_tangents(x, breaks_y, beta_y, breaks_z, beta_z):
    def slopes(breaks, beta):
        values = np.cumsum(beta[1:])
        segments = np.clip(
            np.searchsorted(breaks, x, side="right") - 1, 0, len(values) - 1
        )
        return values[segments]

    tangent = np.column_stack(
        [np.ones(len(x)), slopes(breaks_y, beta_y), slopes(breaks_z, beta_z)]
    )
    return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)


def wire_point_mask(reference, queries, config, diagnostics, tangents=None):
    linearity, _, neighbor_counts, largest_eigenvalues, directions = local_pca(
        reference, queries, config["local_radius"], config["local_min_points"], True
    )
    linearity[largest_eigenvalues <= config["eigenvalue_floor"]] = 0
    linearity_mask = linearity > config["linearity_min"]
    diagnostics["linearity_rejected"] = int((~linearity_mask).sum())
    neighbor_limit = config["density_neighbor_limit"]
    dense_mask = (
        (neighbor_counts >= neighbor_limit) & linearity_mask
        if neighbor_limit > 0
        else np.zeros(len(linearity_mask), dtype=bool)
    )
    direction_tolerance_deg = config.get("dense_direction_tolerance_deg", 0.0)
    aligned_dense_mask = np.zeros(len(linearity_mask), dtype=bool)
    if direction_tolerance_deg > 0 and tangents is not None:
        direction_mask = np.abs(np.einsum("ij,ij->i", directions, tangents)) >= np.cos(
            np.radians(direction_tolerance_deg)
        )
        aligned_dense_mask = dense_mask & direction_mask
    density_rejected = dense_mask & ~aligned_dense_mask
    diagnostics["dense_aligned_rescued"] = int(aligned_dense_mask.sum())
    diagnostics["density_rejected_after_linearity"] = int(density_rejected.sum())
    diagnostics["density_cap_enabled"] = neighbor_limit > 0
    return linearity_mask & ~density_rejected


def extract_wire(project, config, pair_index, pair, diagnostics):
    """Fit a wire with piecewise-linear curves in work Y and Z.

    The axis strategies fall back to a robust axis-based fit when the piecewise
    fit fails; ``axis_adaptive`` also tries it when the fit misses an endpoint.
    """
    if config["fit_strategy"] not in ("axis_recovery", "axis_adaptive"):
        return extract_piecewise_wire(project, config, pair_index, pair, diagnostics)
    try:
        result = extract_piecewise_wire(
            project,
            dict(config, fit_strategy="angle_valid"),
            pair_index,
            pair,
            diagnostics,
        )
        diagnostics["fit_method"] = "angle_valid"
    except FacilityPairError as exc:
        diagnostics["piecewise_fit_failure"] = str(exc)
        from .wire_recovery import recover_wire

        return recover_wire(project, config, pair_index, pair, diagnostics)
    if config["fit_strategy"] == "axis_adaptive":
        from .wire_adaptive import correct_wire_fit

        return correct_wire_fit(project, config, pair_index, pair, diagnostics, result)
    return result


def extract_piecewise_wire(project, config, pair_index, pair, diagnostics):
    arrays = project.arrays
    active_ids = arrays["active"]
    xyz = arrays["xyz"]
    result = {}
    prefix = "wire_" + str(pair_index)
    axis_distances, _ = distances_to_axis(
        xyz[active_ids], pair, config["limit_to_segment"]
    )
    candidate_ids = active_ids[axis_distances <= config["search_radius"]]
    diagnostics["candidate_count"] = len(candidate_ids)
    result[prefix + "_candidate_ids"] = candidate_ids
    require_facility_points(candidate_ids, "No points inside the search cylinder")
    candidate_points = xyz[candidate_ids]
    neighbor_counts = cKDTree(candidate_points).query_ball_point(
        candidate_points, config["density_radius"], return_length=True
    )
    features = np.column_stack(
        [
            neighbor_counts,
            project.vertices["opacity"][candidate_ids],
            project.vertices["intensity"][candidate_ids],
            arrays["normals"][candidate_ids],
        ]
    )
    require_finite_features(features)
    clusters = cluster_dbscan(
        MinMaxScaler().fit_transform(features) * config["feature_weights"],
        config["feature_cluster_epsilon"],
        config["feature_cluster_min_points"],
    )
    filtered_ids = candidate_ids[concatenate_indices(clusters)]
    result[prefix + "_feature_filtered_ids"] = filtered_ids
    diagnostics["feature_filtered_count"] = len(filtered_ids)
    require_facility_points(
        filtered_ids, "No feature clusters. Check endpoints or clustering parameters"
    )
    sorted_points = xyz[filtered_ids[np.argsort(xyz[filtered_ids, 0])]]
    fit_point_count = len(sorted_points)
    trimmed_fit_points = sorted_points[
        int(fit_point_count * config["endpoint_trim_fraction"]) : int(
            fit_point_count * (1 - config["endpoint_trim_fraction"])
        )
    ]
    if not len(trimmed_fit_points):
        raise FacilityPairError("No wire fit points remain after trimming")
    model_y = fit_piecewise_curve(
        trimmed_fit_points[:, 0],
        trimmed_fit_points[:, 1],
        config,
        project.head["config"]["load"]["seed"],
    )
    model_z = fit_piecewise_curve(
        trimmed_fit_points[:, 0],
        trimmed_fit_points[:, 2],
        config,
        project.head["config"]["load"]["seed"],
    )
    curve = np.column_stack(
        [
            candidate_points[:, 0],
            model_y.predict(candidate_points[:, 0]),
            model_z.predict(candidate_points[:, 0]),
        ]
    )
    if not np.isfinite(curve).all():
        raise FacilityPairError("Wire curve contains non-finite coordinates")
    filtered_ids = candidate_ids[
        np.linalg.norm(candidate_points[:, 1:] - curve[:, 1:], axis=1)
        < config["fit_tolerance"]
    ]
    result[prefix + "_fit_inlier_ids"] = filtered_ids.copy()
    result[prefix + "_curve"] = curve
    result[prefix + "_breakpoints_y"] = model_y.fit_breaks
    result[prefix + "_coefficients_y"] = model_y.beta
    result[prefix + "_breakpoints_z"] = model_z.fit_breaks
    result[prefix + "_coefficients_z"] = model_z.beta
    diagnostics.update(
        trimmed=len(trimmed_fit_points),
        fit_inlier_count=len(filtered_ids),
        segments_y=len(model_y.slopes),
        segments_z=len(model_z.slopes),
    )
    require_facility_points(filtered_ids, "No points survived fit-residual filtering")
    if config["point_filter_enabled"]:
        tangents = piecewise_curve_tangents(
            xyz[filtered_ids, 0],
            model_y.fit_breaks,
            model_y.beta,
            model_z.fit_breaks,
            model_z.beta,
        )
        filtered_ids = filtered_ids[
            wire_point_mask(
                candidate_points, xyz[filtered_ids], config, diagnostics, tangents
            )
        ]
    require_facility_points(
        filtered_ids, "No points survived linearity or intersection filtering"
    )
    result[prefix + "_selected_ids"] = filtered_ids
    return filtered_ids, result, features


def stage_roi(project, config, log):
    arrays = project.arrays
    active_ids = arrays["active"]
    coordinates = arrays["xyz"][active_ids]
    roi_mask = np.ones(len(active_ids), bool)
    if config["mode"] == "bounds":
        for dimension in range(3):
            roi_mask &= (
                coordinates[:, dimension] >= config["bounds"][dimension * 2]
            ) & (coordinates[:, dimension] <= config["bounds"][dimension * 2 + 1])
    elif config["mode"] == "pole_side_pick":
        if not len(arrays["poles"]) or len(project.head["picks"]["roi"]) != 1:
            raise ValueError(
                "This ROI mode requires extracted poles and one side-selection point"
            )
        picked_id = project.head["picks"]["roi"][0]
        if picked_id not in active_ids:
            raise ValueError("ROI pick is not among the remaining points")
        xmin, xmax = (
            np.min(arrays["xyz"][arrays["poles"], 0]),
            np.max(arrays["xyz"][arrays["poles"], 0]),
        )
        picked_x = arrays["xyz"][picked_id, 0]
        if picked_x < xmin:
            roi_mask = coordinates[:, 0] >= xmin
        elif picked_x > xmax:
            roi_mask = coordinates[:, 0] <= xmax
    arrays["reason"][active_ids[~roi_mask]] = 1
    arrays["out_of_roi"] = active_ids[~roi_mask]
    arrays["active"] = active_ids[roi_mask]
    project.head["diagnostics"]["roi"] = dict(
        removed=int((~roi_mask).sum()), remaining=int(roi_mask.sum())
    )


def stage_denoise(project, config, log):
    arrays = project.arrays
    active_ids = arrays["active"]
    clusters = (
        cluster_dbscan(
            arrays["xyz"][active_ids],
            config["cluster_epsilon"],
            config["cluster_min_points"],
        )
        if config["enabled"]
        else [np.arange(len(active_ids))]
    )
    retained_clusters = [
        cluster
        for cluster in clusters
        if not config["enabled"] or len(cluster) >= config["min_cluster_size"]
    ]
    retained_ids = np.sort(active_ids[concatenate_indices(retained_clusters)])
    if not len(retained_ids):
        raise ValueError(
            "No tree points remain after denoising. Adjust settings or disable filtering"
        )
    arrays["reason"][np.setdiff1d(active_ids, retained_ids)] = 2
    arrays["tree_points"] = retained_ids
    arrays["active"] = retained_ids
    arrays["tree_min"] = arrays["xyz"][retained_ids].min(axis=0)
    arrays["tree_max"] = arrays["xyz"][retained_ids].max(axis=0)
    project.head["diagnostics"]["denoise"] = dict(
        clusters=len(clusters),
        valid_clusters=len(retained_clusters),
        remaining=len(retained_ids),
    )


def layer_clusters(project, config, stage):
    arrays = project.arrays
    ids = arrays["tree_points"]
    height = arrays["xyz"][ids, 2] - arrays["tree_min"][2]
    layer_ids = ids[(height >= config["height_min"]) & (height <= config["height_max"])]
    cluster_ids = [
        layer_ids[cluster]
        for cluster in cluster_euclidean(
            arrays["xyz"][layer_ids],
            config["tolerance"],
            config["min_cluster_size"],
            config["max_cluster_size"],
        )
    ]
    arrays["reason"][np.setdiff1d(layer_ids, concatenate_indices(cluster_ids))] = 3
    if stage == "stumps":
        cluster_ids.sort(
            key=lambda ids: cluster_endpoints(
                arrays["xyz"], ids, config["center_margin"]
            )[0][0]
        )
    arrays[stage + "_group"] = np.full(project.point_count, -1, "i4")
    for cluster_index, ids in enumerate(cluster_ids):
        arrays[stage + "_group"][ids] = cluster_index
    tops = []
    bottoms = []
    for ids in cluster_ids:
        top_center, bottom_center = cluster_endpoints(
            arrays["xyz"], ids, config["center_margin"]
        )
        tops.append(top_center)
        bottoms.append(bottom_center)
    arrays[stage + "_tops"] = np.asarray(tops).reshape(-1, 3)
    arrays[stage + "_bottoms"] = np.asarray(bottoms).reshape(-1, 3)
    project.head["diagnostics"][stage] = dict(
        layer=len(layer_ids),
        clusters=len(cluster_ids),
        retained=len(concatenate_indices(cluster_ids)),
    )
    return cluster_ids


def stage_stumps(project, config, log):
    groups = layer_clusters(project, config, "stumps")
    if not groups:
        raise ValueError(
            "No stump clusters found. Adjust height, radius and point thresholds"
        )
    for cluster_index, cluster in enumerate(groups):
        project.arrays["labels"][cluster] = cluster_index + 4
        project.arrays["reason"][cluster] = 0


def stage_trunks(project, config, log):
    layer_clusters(project, config, "trunks")


def stage_match_trunks(project, config, log):
    arrays = project.arrays
    trunk_bottoms = arrays["trunks_bottoms"]
    stump_tops = arrays["stumps_tops"]
    stump_assignments = np.full(len(trunk_bottoms), -1, "i4")
    match_distances = np.zeros(len(trunk_bottoms))
    for trunk_index, trunk_bottom in enumerate(trunk_bottoms):
        distances = np.linalg.norm(stump_tops - trunk_bottom, axis=1)
        stump_index = int(distances.argmin())
        match_distances[trunk_index] = distances[stump_index]
        if (
            config["max_distance"] == 0
            or distances[stump_index] <= config["max_distance"]
        ):
            stump_assignments[trunk_index] = stump_index
    arrays["trunk_to_stump"] = stump_assignments
    arrays["trunk_match_distances"] = match_distances
    arrays["matched_trunk"] = np.full(project.point_count, -1, "i4")
    for trunk_index, stump_index in enumerate(stump_assignments):
        if stump_index >= 0:
            ids = np.flatnonzero(arrays["trunks_group"] == trunk_index)
            arrays["labels"][ids] = stump_index + 4
            arrays["matched_trunk"][ids] = stump_index
            arrays["reason"][ids] = 0
    project.head["diagnostics"]["match_trunks"] = dict(
        assigned=int((stump_assignments >= 0).sum()),
        unassigned=int((stump_assignments < 0).sum()),
    )


def local_directions(points, neighbor_count):
    neighbor_count = min(neighbor_count, len(points))
    tree = cKDTree(points)
    directions = np.empty_like(points)
    if neighbor_count < 3:
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    for start in range(0, len(points), 4096):
        _, neighbor_indices = tree.query(points[start : start + 4096], k=neighbor_count)
        neighborhoods = points[neighbor_indices]
        neighborhoods = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", neighborhoods, neighborhoods) / (
            neighbor_count - 1
        )
        _, eigenvectors = np.linalg.eigh(covariance)
        directions[start : start + len(neighborhoods)] = eigenvectors[:, :, -1]
    return directions


def stage_branches(project, config, log):
    """Cut a direction-weighted MST of branch points into fragments."""
    arrays = project.arrays
    branch_ids = arrays["tree_points"]
    height = arrays["xyz"][branch_ids, 2] - arrays["tree_min"][2]
    branch_ids = branch_ids[
        (height >= config["height_min"])
        & (arrays["matched_trunk"][branch_ids] < 0)
        & (arrays["stumps_group"][branch_ids] < 0)
    ]
    branch_coordinates = arrays["xyz"][branch_ids]
    arrays["branch_points"] = branch_ids
    if not len(branch_coordinates):
        fragment_clusters = []
    elif len(branch_coordinates) == 1:
        fragment_clusters = [np.array([0])]
    else:
        directions = local_directions(
            branch_coordinates, config["direction_neighbor_count"]
        )
        arrays["branch_directions"] = directions
        graph = kneighbors_graph(
            branch_coordinates,
            n_neighbors=min(
                config["graph_neighbor_count"], len(branch_coordinates) - 1
            ),
            mode="distance",
            include_self=False,
        ).tocoo()
        cosine = np.abs(
            np.einsum("ij,ij->i", directions[graph.row], directions[graph.col])
        )
        graph.data = np.maximum(
            config["distance_weight"] * graph.data
            + config["direction_weight"] * (1 - np.clip(cosine, 0, 1)),
            np.finfo(float).eps,
        )
        spanning_tree = minimum_spanning_tree(graph.tocsr())
        spanning_tree.data[spanning_tree.data > config["cut_threshold"]] = 0
        spanning_tree.eliminate_zeros()
        _, labels = connected_components(spanning_tree, directed=False)
        fragment_clusters = group_indices_by_label(labels)
    arrays["fragment_id"][:] = -1
    valid_fragment_ids = []
    tops = []
    bottoms = []
    for fragment_id, cluster in enumerate(fragment_clusters):
        original_ids = branch_ids[cluster]
        arrays["fragment_id"][original_ids] = fragment_id
        if len(cluster) < config["min_cluster_size"]:
            arrays["reason"][original_ids] = 4
            continue
        top_center, bottom_center = cluster_endpoints(
            arrays["xyz"], original_ids, config["center_margin"]
        )
        valid_fragment_ids.append(fragment_id)
        tops.append(top_center)
        bottoms.append(bottom_center)
    order = np.argsort(np.asarray(bottoms).reshape(-1, 3)[:, 2])
    arrays["valid_fragments"] = np.asarray(valid_fragment_ids, dtype=np.int64)[order]
    arrays["fragment_tops"] = np.asarray(tops).reshape(-1, 3)[order]
    arrays["fragment_bottoms"] = np.asarray(bottoms).reshape(-1, 3)[order]
    project.head["diagnostics"]["branches"] = dict(
        branch_points=len(branch_ids),
        fragments=len(fragment_clusters),
        valid_fragments=len(valid_fragment_ids),
        small_fragment_points=int(np.count_nonzero(arrays["reason"] == 4)),
    )


def stage_assign_branches(project, config, log):
    """Assign fragments bottom-up to the nearest trunk or a lower fragment."""
    arrays = project.arrays
    trunk_ids = np.flatnonzero(arrays["matched_trunk"] >= 0)
    valid_fragment_ids = arrays["valid_fragments"]
    tree_assignments = np.full(len(valid_fragment_ids), -1, "i4")
    if len(valid_fragment_ids) and not len(trunk_ids):
        log(
            "No matched trunks. Fragments remain unassigned; assign them to stump instances during manual review"
        )
    elif len(valid_fragment_ids):
        tree = cKDTree(arrays["xyz"][trunk_ids])
        tops = arrays["fragment_tops"]
        for fragment_index, fragment_bottom in enumerate(arrays["fragment_bottoms"]):
            target_distance, neighbor_index = tree.query(fragment_bottom)
            tree_label = int(arrays["labels"][trunk_ids[neighbor_index]])
            previous_fragment_indices = np.flatnonzero(
                tree_assignments[:fragment_index] >= 4
            )
            if len(previous_fragment_indices):
                previous_distances = np.linalg.norm(
                    tops[previous_fragment_indices] - fragment_bottom, axis=1
                )
                neighbor_index = int(previous_distances.argmin())
                previous_distance = previous_distances[neighbor_index]
                if (
                    fragment_index == len(valid_fragment_ids) - 1
                    and config["last_only_previous"]
                    or config["previous_fragment_weight"] * previous_distance
                    < (1 - config["previous_fragment_weight"]) * target_distance
                ):
                    tree_label = int(
                        tree_assignments[previous_fragment_indices[neighbor_index]]
                    )
                    target_distance = previous_distance
            if config["max_distance"] == 0 or target_distance <= config["max_distance"]:
                tree_assignments[fragment_index] = tree_label
    for fragment_id, tree_label in zip(valid_fragment_ids, tree_assignments):
        ids = np.flatnonzero(arrays["fragment_id"] == fragment_id)
        if tree_label >= 4:
            arrays["labels"][ids] = tree_label
            arrays["reason"][ids] = 0
        else:
            arrays["reason"][ids] = 5
    arrays["reason"][(arrays["labels"] == 0) & (arrays["reason"] == 0)] = 6
    arrays["fragment_to_tree"] = tree_assignments
    arrays["auto_labels"] = arrays["labels"].copy()
    arrays["manual_override"][:] = 0
    arrays["provenance"][:] = 0
    project.head["diagnostics"]["assign_branches"] = dict(
        assigned=int((tree_assignments >= 4).sum()),
        unassigned=int((tree_assignments < 4).sum()),
    )


def stage_refill(project, config, log):
    """Relabel eligible unassigned points from unambiguous nearby classes."""
    arrays = project.arrays
    if not config["enabled"]:
        project.head["diagnostics"]["refill"] = {"enabled": False, "filled": 0}
        return
    eligible_ids = np.flatnonzero(
        (arrays["labels"] == 0)
        & (arrays["manual_override"] == 0)
        & np.isin(arrays["reason"], config["eligible_reasons"])
    )
    query_points = arrays["xyz"][eligible_ids]
    labels = np.unique(arrays["labels"])
    labels = labels[labels > 0]
    nearest_distances = np.full(len(eligible_ids), np.inf)
    second_nearest_distances = np.full(len(eligible_ids), np.inf)
    winning_labels = np.zeros(len(eligible_ids), "i4")
    for label in labels:
        reference_ids = np.flatnonzero(arrays["labels"] == label)
        distance, _ = cKDTree(arrays["xyz"][reference_ids]).query(query_points)
        limit = config[
            (
                "ground_distance"
                if label == 1
                else (
                    "pole_distance"
                    if label == 2
                    else "wire_distance" if label == 3 else "tree_distance"
                )
            )
        ]
        candidate_mask = distance <= limit
        if label == 1:
            if "ground_plane" not in arrays:
                candidate_mask[:] = False
            else:
                candidate_mask &= (
                    np.abs(
                        query_points @ arrays["ground_plane"][:3]
                        + arrays["ground_plane"][3]
                    )
                    <= config["ground_plane_distance"]
                )
        elif label == 2:
            axis_distance = np.full(len(eligible_ids), np.inf)
            for pair in arrays["pole_axes"]:
                axis_distance = np.minimum(
                    axis_distance, distances_to_axis(query_points, pair, True)[0]
                )
            candidate_mask &= axis_distance <= config["pole_axis_radius"]
        elif label == 3:
            curves = [
                v
                for k, v in arrays.items()
                if k.startswith("wire_") and k.endswith("_curve")
            ]
            if not curves:
                candidate_mask[:] = False
            else:
                curve_distance, _ = cKDTree(np.concatenate(curves)).query(query_points)
                candidate_mask &= curve_distance <= config["wire_curve_radius"]
        else:
            height = query_points[:, 2] - arrays["tree_min"][2]
            candidate_mask &= (height >= config["tree_height_min"]) & (
                height <= config["tree_height_max"]
            )
        distance = np.where(candidate_mask, distance, np.inf)
        closer_mask = distance < nearest_distances
        second_nearest_distances = np.where(
            closer_mask,
            nearest_distances,
            np.minimum(second_nearest_distances, distance),
        )
        nearest_distances = np.minimum(nearest_distances, distance)
        winning_labels[closer_mask] = label
    accepted_mask = np.isfinite(nearest_distances) & (
        ~np.isfinite(second_nearest_distances)
        | (
            second_nearest_distances
            - np.where(np.isfinite(nearest_distances), nearest_distances, 0)
            > config["ambiguity_margin"]
        )
    )
    arrays["labels"][eligible_ids[accepted_mask]] = winning_labels[accepted_mask]
    arrays["provenance"][eligible_ids[accepted_mask]] = 2
    project.head["diagnostics"]["refill"] = dict(
        enabled=True,
        eligible=len(eligible_ids),
        filled=int(accepted_mask.sum()),
        refused=int((~accepted_mask).sum()),
    )
