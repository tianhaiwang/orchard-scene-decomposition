"""Robust axis-based wire fit used when piecewise fitting is unreliable."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def recover_wire(project, config, pair_index, pair, diagnostics):
    from .pipeline import (
        FacilityPairError,
        local_pca,
        distances_to_axis,
        require_facility_points,
        require_finite_features,
    )

    xyz = project.arrays["xyz"]
    active_ids = project.arrays["active"]
    prefix = "wire_" + str(pair_index)
    distance, axis = distances_to_axis(
        xyz[active_ids], pair, config["limit_to_segment"]
    )
    candidate_ids = active_ids[distance <= config["search_radius"]]
    require_facility_points(candidate_ids, "No points inside the recovery cylinder")
    candidate_points = xyz[candidate_ids]
    length = np.linalg.norm(pair[1] - pair[0])
    reference_axis = np.eye(3)[np.argmin(np.abs(axis))]
    transverse_axis_y = np.cross(axis, reference_axis)
    transverse_axis_y /= np.linalg.norm(transverse_axis_y)
    transverse_axis_z = np.cross(axis, transverse_axis_y)
    basis = np.column_stack([axis, transverse_axis_y, transverse_axis_z])
    axis_coordinates = (candidate_points - pair[0]) @ basis
    density = cKDTree(candidate_points).query_ball_point(
        candidate_points, config["density_radius"], return_length=True
    )
    features = np.column_stack(
        [
            density,
            project.vertices["opacity"][candidate_ids],
            project.vertices["intensity"][candidate_ids],
            project.arrays["normals"][candidate_ids],
        ]
    )
    require_finite_features(features)
    linearity, _, neighbor_counts, largest_eigenvalues, directions = local_pca(
        candidate_points,
        candidate_points,
        config["local_radius"],
        config["local_min_points"],
        True,
    )
    geometry_mask = (
        (linearity > config["linearity_min"])
        & (largest_eigenvalues > config["eigenvalue_floor"])
        & (neighbor_counts >= config["local_min_points"])
    )
    axis_alignment_mask = np.abs(directions @ axis) >= np.cos(
        np.radians(config["recovery_direction_tolerance_deg"])
    )
    seed_mask = (
        geometry_mask
        & axis_alignment_mask
        & (np.linalg.norm(axis_coordinates[:, 1:], axis=1) <= config["fit_tolerance"])
    )
    endpoint_masks = [
        np.linalg.norm(candidate_points - end, axis=1) <= config["local_radius"]
        for end in pair
    ]
    endpoint_counts = [
        int(np.count_nonzero(seed_mask & mask)) for mask in endpoint_masks
    ]
    diagnostics["recovery_endpoint_support"] = endpoint_counts
    endpoint_angles = []
    for mask in endpoint_masks:
        supported = mask & geometry_mask
        endpoint_angles.append(
            float(
                np.median(
                    np.degrees(
                        np.arccos(np.clip(np.abs(directions[supported] @ axis), 0, 1))
                    )
                )
            )
            if supported.any()
            else None
        )
    diagnostics["recovery_endpoint_axis_angles"] = endpoint_angles
    if min(endpoint_counts) < config["local_min_points"]:
        if all(
            value is not None and value > config["recovery_direction_tolerance_deg"]
            for value in endpoint_angles
        ):
            raise FacilityPairError(
                "Picked endpoints disagree with both local wire directions; they may be on different wires. Re-pick both endpoints on the same wire"
            )
        raise FacilityPairError(
            "Recovery lacks geometrically supported points near both endpoints"
        )
    segment_bin_count = max(3, int(np.ceil(length / (2 * config["local_radius"]))))
    bin_width = length / segment_bin_count
    bin_indices = np.floor(axis_coordinates[:, 0] / bin_width).astype(int)
    inside_segment_mask = (axis_coordinates[:, 0] >= 0) & (
        axis_coordinates[:, 0] <= length
    )
    bin_indices[inside_segment_mask] = np.clip(
        bin_indices[inside_segment_mask], 0, segment_bin_count - 1
    )
    supported_bins = np.unique(bin_indices[seed_mask])
    first_bin, last_bin = (
        (0, segment_bin_count - 1)
        if config["limit_to_segment"]
        else (
            min(0, int(supported_bins.min())),
            max(segment_bin_count - 1, int(supported_bins.max())),
        )
    )
    bin_centers = []
    for bin_index in range(first_bin, last_bin + 1):
        indices = np.flatnonzero(seed_mask & (bin_indices == bin_index))
        if len(indices) >= config["local_min_points"]:
            bin_centers.append(np.median(axis_coordinates[indices], axis=0))
    total_bins = last_bin - first_bin + 1
    coverage = len(bin_centers) / total_bins
    diagnostics.update(
        recovery_bins=total_bins,
        recovery_supported_bins=len(bin_centers),
        recovery_coverage=coverage,
    )
    if len(bin_centers) < 3 or coverage < config["recovery_min_coverage"]:
        raise FacilityPairError(
            "Recovery has insufficient longitudinal support; check picks or missing wire data"
        )
    bin_centers = np.asarray(bin_centers)
    trim_count = int(len(bin_centers) * config["endpoint_trim_fraction"])
    initial_centers = bin_centers[trim_count : len(bin_centers) - trim_count]
    if len(initial_centers) < 3:
        raise FacilityPairError("Too few recovery fit centers after endpoint trimming")
    fit_origin = float(bin_centers[0, 0])
    fit_span = float(bin_centers[-1, 0] - fit_origin)
    degree = 2 if len(initial_centers) >= 6 else 1
    design_matrix = np.polynomial.polynomial.polyvander(
        2 * (bin_centers[:, 0] - fit_origin) / fit_span - 1, degree
    )
    initial_design_matrix = design_matrix[trim_count : len(bin_centers) - trim_count]
    initial_coefficients = np.linalg.lstsq(
        initial_design_matrix, initial_centers[:, 1:], rcond=None
    )[0]
    model = least_squares(
        lambda coefficients: (
            initial_design_matrix @ coefficients.reshape(degree + 1, 2)
            - initial_centers[:, 1:]
        ).ravel(),
        initial_coefficients.ravel(),
        loss="soft_l1",
        f_scale=config["local_radius"] / 2,
    )
    coefficients = model.x.reshape(degree + 1, 2)
    if not model.success or not np.isfinite(coefficients).all():
        raise FacilityPairError("Recovery robust fit did not converge")
    rng = np.random.default_rng(project.head["config"]["load"]["seed"])
    best_median_residual = np.median(
        np.linalg.norm(design_matrix @ coefficients - bin_centers[:, 1:], axis=1)
    )
    for _ in range(256):
        sample_indices = rng.choice(len(bin_centers), degree + 1, replace=False)
        if np.linalg.cond(design_matrix[sample_indices]) > 1e5:
            continue
        candidate = np.linalg.lstsq(
            design_matrix[sample_indices], bin_centers[sample_indices, 1:], rcond=None
        )[0]
        median_residual = np.median(
            np.linalg.norm(design_matrix @ candidate - bin_centers[:, 1:], axis=1)
        )
        if median_residual < best_median_residual:
            best_median_residual = median_residual
            coefficients = candidate
    for iteration in range(10):
        errors = np.linalg.norm(
            design_matrix @ coefficients - bin_centers[:, 1:], axis=1
        )
        median_error = np.median(errors)
        error_mad = np.median(np.abs(errors - median_error))
        # Tukey biweight cut-off; 1.4826 scales the MAD to a standard deviation.
        residual_cutoff = min(
            config["fit_tolerance"],
            max(config["local_radius"] / 2, median_error + 4.685 * 1.4826 * error_mad),
        )
        weights = np.square(np.maximum(0, 1 - (errors / residual_cutoff) ** 2))
        trusted_center_mask = weights > 0.05
        if np.count_nonzero(trusted_center_mask) < max(3, degree + 1):
            raise FacilityPairError(
                "Too few consistent wire centers after branch rejection"
            )
        sqrt_weights = np.sqrt(weights)[:, None]
        updated_coefficients = np.linalg.lstsq(
            design_matrix * sqrt_weights, bin_centers[:, 1:] * sqrt_weights, rcond=None
        )[0]
        converged = np.max(np.abs(updated_coefficients - coefficients)) < 1e-7
        coefficients = updated_coefficients
        if converged:
            break
    training_centers = bin_centers[trusted_center_mask]
    diagnostics.update(
        recovery_trim_each_end=trim_count,
        recovery_training_centers=len(training_centers),
        recovery_rejected_centers=int((~trusted_center_mask).sum()),
        recovery_center_cutoff=float(residual_cutoff),
    )

    def evaluate(axis_positions):
        normalized_positions = 2 * (axis_positions - fit_origin) / fit_span - 1
        transverse_offsets = (
            np.polynomial.polynomial.polyvander(normalized_positions, degree)
            @ coefficients
        )
        transverse_slopes = np.zeros((len(axis_positions), 2))
        for power in range(1, degree + 1):
            transverse_slopes += (
                power
                * normalized_positions[:, None] ** (power - 1)
                * coefficients[power]
                * (2 / fit_span)
            )
        tangent = (
            np.column_stack([np.ones(len(axis_positions)), transverse_slopes]) @ basis.T
        )
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        curve = (
            pair[0] + np.column_stack([axis_positions, transverse_offsets]) @ basis.T
        )
        return curve, tangent

    sample_curve, sample_tangent = evaluate(
        np.linspace(min(0, bin_centers[0, 0]), max(length, bin_centers[-1, 0]), 101)
    )
    bend = float(
        np.degrees(np.arccos(np.clip(np.min(sample_tangent @ sample_tangent.T), -1, 1)))
    )
    diagnostics["recovery_bend_angle"] = bend
    if bend > config["max_bend_angle_deg"]:
        raise FacilityPairError("Recovery curve exceeds the allowed bend angle")
    endpoint_curve, _ = evaluate(np.array([0.0, length]))
    endpoint_error = np.linalg.norm(endpoint_curve - pair, axis=1)
    if np.any(endpoint_error > config["fit_tolerance"]):
        raise FacilityPairError("Recovery curve misses the picked endpoints")
    center_curve, _ = evaluate(training_centers[:, 0])
    center_world = pair[0] + training_centers @ basis.T
    center_residual = np.linalg.norm(center_curve - center_world, axis=1)
    median = np.median(center_residual)
    mad = np.median(np.abs(center_residual - median))
    radius = min(
        config["fit_tolerance"],
        max(
            config["local_radius"] / 2,
            median + config["recovery_sigma_multiplier"] * mad,
            float(endpoint_error.max()) + config["local_radius"] / 4,
        ),
    )
    diagnostics["recovery_fit_radius"] = float(radius)
    curve, tangents = evaluate(axis_coordinates[:, 0])
    residual = np.linalg.norm(candidate_points - curve, axis=1)
    fit_inlier_mask = residual < radius
    aligned_fit = np.abs(np.einsum("ij,ij->i", directions, tangents)) >= np.cos(
        np.radians(config["recovery_direction_tolerance_deg"])
    )
    selected_mask = (
        fit_inlier_mask
        & geometry_mask
        & aligned_fit
        & (bin_indices >= first_bin)
        & (bin_indices <= last_bin)
    )
    selected_ids = candidate_ids[selected_mask]
    require_facility_points(
        selected_ids, "No geometrically supported wire points survived recovery"
    )
    final_bins = sum(
        np.count_nonzero(selected_mask & (bin_indices == bin_index))
        >= config["local_min_points"]
        for bin_index in range(first_bin, last_bin + 1)
    )
    final_endpoints = [
        int(np.count_nonzero(selected_mask & mask)) for mask in endpoint_masks
    ]
    diagnostics.update(
        recovery_final_coverage=float(final_bins / total_bins),
        recovery_final_endpoint_support=final_endpoints,
    )
    if (
        final_bins / total_bins < config["recovery_min_coverage"]
        or min(final_endpoints) < config["local_min_points"]
    ):
        raise FacilityPairError(
            "Recovered fit does not retain enough longitudinal or endpoint support"
        )
    diagnostics.update(
        recovery_limit_to_segment=bool(config["limit_to_segment"]),
        recovery_extent=[
            float(axis_coordinates[selected_mask, 0].min()),
            float(axis_coordinates[selected_mask, 0].max()),
        ],
        recovery_outside_picks=int(
            np.count_nonzero(
                selected_mask
                & ((axis_coordinates[:, 0] < 0) | (axis_coordinates[:, 0] > length))
            )
        ),
        fit_method="axis_recovery",
        candidate_count=len(candidate_ids),
        feature_filtered_count=int(seed_mask.sum()),
        trimmed=len(training_centers),
        fit_inlier_count=int(fit_inlier_mask.sum()),
        recovery_degree=degree,
        recovery_direction_rejected=int(
            (fit_inlier_mask & geometry_mask & ~aligned_fit).sum()
        ),
        recovery_residual_p90=float(np.quantile(residual[selected_mask], 0.9)),
    )
    result = {
        prefix + "_candidate_ids": candidate_ids,
        prefix + "_feature_filtered_ids": candidate_ids[seed_mask],
        prefix + "_fit_inlier_ids": candidate_ids[fit_inlier_mask],
        prefix + "_selected_ids": selected_ids,
        prefix + "_curve": curve[selected_mask],
        prefix + "_recovery_basis": basis,
        prefix + "_recovery_origin": pair[0],
        prefix + "_recovery_length": np.asarray(length),
        prefix + "_recovery_fit_domain": np.asarray([fit_origin, fit_span]),
        prefix + "_recovery_center_weights": weights,
        prefix + "_recovery_training": training_centers,
        prefix + "_recovery_coefficients": coefficients,
        prefix + "_recovery_centers": bin_centers,
    }
    return selected_ids, result, features
