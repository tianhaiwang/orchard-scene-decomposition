"""Pipeline stages and the parameter schema with defaults and validation."""

import copy
import math

STAGES = [
    "load",
    "ground",
    "poles",
    "wires",
    "roi",
    "denoise",
    "stumps",
    "trunks",
    "match_trunks",
    "branches",
    "assign_branches",
    "refill",
]
TITLES = dict(
    zip(
        STAGES,
        [
            "Load & align",
            "Ground",
            "Poles",
            "Wires",
            "Region of interest",
            "Denoise",
            "Stump clusters",
            "Trunk clusters",
            "Match trunks",
            "Branch fragments",
            "Assign branches",
            "Refill unassigned points",
        ],
    )
)
SCHEMA = {}


def add(
    stage_name, parameter_name, default, label, minimum=None, maximum=None, choices=None
):
    SCHEMA[stage_name + "." + parameter_name] = dict(
        default=default,
        label=label,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        stage=stage_name,
    )


add("load", "alignment", "xyz", "PCA alignment", choices=["xyz", "xy", "none"])
add("load", "normal_radius", 0.5, "Normal radius (m)", 1e-6)
add("load", "normal_max_neighbors", 30, "Normal neighbors (max)", 3)
add(
    "load",
    "normal_coordinate_frame",
    "work",
    "Normal coordinate frame",
    choices=["work", "source"],
)
add("load", "seed", 42, "Random seed", 0, 2147483647)
add("ground", "method", "ransac", "Ground method", choices=["ransac", "csf", "skip"])
for parameter_name, default_value, label, minimum, maximum in [
    ("distance_threshold", 0.1, "RANSAC distance (m)", 1e-6, None),
    ("ransac_n", 3, "RANSAC samples", 3, None),
    ("num_iterations", 1000, "RANSAC iterations", 1, None),
    ("min_ground_normal_z", 0.95, "Ground normal cosine (min)", 0, 1),
    ("max_planes", 5, "Candidate planes (max)", 1, None),
    ("max_height_above_ground", 0.2, "Height above ground (m)", 0, None),
    ("cloth_resolution", 0.01, "CSF cloth resolution (m)", 1e-6, None),
    ("rigidness", 2, "CSF rigidity", 1, 3),
    ("time_step", 0.65, "CSF time step", 1e-6, None),
    ("class_threshold", 0.15, "CSF classification distance (m)", 1e-6, None),
    ("iterations", 500, "CSF iterations", 1, None),
]:
    add("ground", parameter_name, default_value, label, minimum, maximum)
add("ground", "bSloopSmooth", False, "CSF slope smoothing")
SCHEMA["ground.method"]["choice_labels"] = {
    "ransac": "RANSAC plane fitting",
    "csf": "CSF cloth simulation",
    "skip": "Skip ground extraction",
}
for parameter_name in [
    "distance_threshold",
    "ransac_n",
    "num_iterations",
    "min_ground_normal_z",
    "max_planes",
    "max_height_above_ground",
]:
    SCHEMA["ground." + parameter_name]["ground_method"] = "ransac"
for parameter_name in [
    "cloth_resolution",
    "rigidness",
    "time_step",
    "class_threshold",
    "iterations",
    "bSloopSmooth",
]:
    SCHEMA["ground." + parameter_name]["ground_method"] = "csf"
for stage_name in ["poles", "wires"]:
    add(stage_name, "enabled", True, "Extract facilities (off = skip)")
    add(stage_name, "search_radius", 0.25, "Search cylinder radius (m)", 1e-6)
    add(stage_name, "limit_to_segment", False, "Limit search to picked segment")
    add(
        stage_name,
        "feature_cluster_epsilon",
        0.2 if stage_name == "poles" else 0.03,
        "Feature DBSCAN epsilon",
        1e-6,
    )
    add(
        stage_name, "feature_cluster_min_points", 5, "Feature DBSCAN neighbors (min)", 1
    )
    add(
        stage_name,
        "feature_weights",
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Feature weights [distance/density, opacity, intensity, nx, ny, nz]",
    )
for parameter_name, default_value, label, minimum, maximum in [
    ("sigma_multiplier", 2.0, "Initial cylinder: mean + k sigma", 0, None),
    ("cross_section_half_height", 0.3, "Cross-section half-height (m)", 1e-6, None),
    ("cross_section_min_points", 20, "Cross-section points (min)", 3, None),
    (
        "cross_section_cluster_epsilon",
        0.03,
        "Cross-section DBSCAN epsilon (m)",
        1e-6,
        None,
    ),
    (
        "cross_section_cluster_min_points",
        5,
        "Cross-section DBSCAN neighbors (min)",
        1,
        None,
    ),
    ("cross_section_radius_percentile", 95.0, "Main cluster radius percentile", 0, 100),
    ("cross_section_radius_multiplier", 2.0, "Main cluster radius multiplier", 0, None),
    ("local_radius", 0.15, "Local PCA radius (m)", 1e-6, None),
    ("local_min_points", 5, "Local PCA neighbors (min)", 3, None),
    ("mad_multiplier", 2.5, "Axis distance: median + k MAD", 0, None),
    ("verticality_min", 0.6, "Local verticality (min)", 0, 1),
    ("max_normal_axis_dot", 0.3, "Normal-axis dot product (max)", 0, 1),
    ("min_feature_votes", 2, "Required feature votes", 1, 3),
]:
    add("poles", parameter_name, default_value, label, minimum, maximum)
add("poles", "cross_section_filter_enabled", True, "Filter cross-section density gaps")
add("poles", "point_filter_enabled", True, "Enable point feature voting")
for parameter_name, default_value, label, minimum, maximum in [
    ("density_radius", 0.1, "Density search radius (m)", 1e-6, None),
    ("endpoint_trim_fraction", 0.05, "Fit endpoint trim fraction", 0, 0.49),
    ("min_fit_segments", 2, "Fit segments (min)", 1, None),
    ("max_fit_segments", 6, "Fit segments (max)", 1, None),
    ("max_bend_angle_deg", 30.0, "Adjacent segment angle (deg)", 0, 180),
    ("fit_tolerance", 0.1, "Joint YZ fit tolerance (m)", 1e-6, None),
    ("fit_max_iterations", 1000, "PWLF iterations (max)", "MIN", None),
    ("fit_population_multiplier", 15, "PWLF population multiplier", 1, None),
    ("local_radius", 0.05, "Linearity/density radius (m)", 1e-6, None),
    ("local_min_points", 5, "PCA neighbors (min)", 3, None),
    ("eigenvalue_floor", 1e-6, "PCA eigenvalue floor", 0, None),
    ("linearity_min", 0.6, "Linearity (min)", 0, 1),
    ("density_neighbor_limit", 75, "Density check neighbors (0 = off)", 0, None),
]:
    add(
        "wires",
        parameter_name,
        default_value,
        label,
        1 if minimum == "MIN" else minimum,
        maximum,
    )
add(
    "wires",
    "dense_direction_tolerance_deg",
    15.0,
    "Dense-point tangent angle (deg; 0 = density limit)",
    0,
    90,
)
add(
    "wires",
    "fit_strategy",
    "axis_adaptive",
    "Fit selection",
    choices=["axis_adaptive", "axis_recovery", "angle_valid", "last_candidate"],
)
add(
    "wires",
    "recovery_direction_tolerance_deg",
    30.0,
    "Recovery direction tolerance (deg)",
    0,
    90,
)
add("wires", "recovery_min_coverage", 0.5, "Recovery occupied-bin fraction (min)", 0, 1)
add(
    "wires", "recovery_sigma_multiplier", 3.0, "Recovery center MAD multiplier", 0, None
)
add("wires", "point_filter_enabled", True, "Filter linearity and intersections")
add("roi", "mode", "none", "ROI mode", choices=["none", "pole_side_pick", "bounds"])
add(
    "roi",
    "bounds",
    [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    "Work bounds [xmin,xmax,ymin,ymax,zmin,zmax]",
)
add("denoise", "enabled", True, "Enable noise filtering")
add("denoise", "cluster_epsilon", 0.2, "Spatial DBSCAN epsilon (m)", 1e-6)
add("denoise", "cluster_min_points", 100, "DBSCAN neighbors (min)", 1)
add("denoise", "min_cluster_size", 1000, "Retained cluster points (min)", 1)
for stage_name, low, high, tol, mn, mx in [
    ("stumps", 0.0, 0.5, 0.05, 100, 50000),
    ("trunks", 0.5, 1.0, 0.03, 300, 500000),
]:
    add(stage_name, "height_min", low, "Height above tree base (min, m)", 0)
    add(stage_name, "height_max", high, "Height above tree base (max, m)", 0)
    add(stage_name, "tolerance", tol, "Euclidean connection radius (m)", 1e-6)
    add(stage_name, "min_cluster_size", mn, "Cluster points (exclusive min)", 0)
    add(stage_name, "max_cluster_size", mx, "Cluster points (exclusive max)", 1)
    add(stage_name, "center_margin", 0.02, "Endpoint center thickness (m)", 1e-6)
add("match_trunks", "max_distance", 0.0, "Trunk-stump distance (m; 0 = unlimited)", 0)
for parameter_name, default_value, label, minimum, maximum in [
    ("height_min", 0.5, "Branch layer height (min, m)", 0, None),
    ("direction_neighbor_count", 30, "Direction PCA neighbors", 3, None),
    ("graph_neighbor_count", 15, "MST neighbors", 1, None),
    ("distance_weight", 2.0, "MST distance weight", 0, None),
    ("direction_weight", 2.0, "MST direction weight", 0, None),
    ("cut_threshold", 0.2, "MST edge cut threshold", 1e-9, None),
    ("min_cluster_size", 100, "Fragment points (min)", 1, None),
    ("center_margin", 0.02, "Fragment endpoint thickness (m)", 1e-6, None),
]:
    add("branches", parameter_name, default_value, label, minimum, maximum)
add(
    "assign_branches",
    "previous_fragment_weight",
    0.5,
    "Previous branch distance weight",
    0,
    1,
)
add(
    "assign_branches",
    "last_only_previous",
    True,
    "Top fragment: previous fragments only",
)
add(
    "assign_branches",
    "max_distance",
    0.0,
    "Fragment link distance (m; 0 = unlimited)",
    0,
)
add("refill", "enabled", False, "Enable refill after manual review")
add(
    "refill",
    "eligible_reasons",
    [2, 3, 4],
    "Refill reasons [2=noise, 3=small cluster, 4=small fragment]",
)
for parameter_name, default_value, label in [
    ("ground_distance", 0.03, "Ground refill distance (m)"),
    ("pole_distance", 0.02, "Pole refill distance (m)"),
    ("wire_distance", 0.015, "Wire refill distance (m)"),
    ("tree_distance", 0.04, "Tree refill distance (m)"),
    ("ground_plane_distance", 0.2, "Ground plane constraint (m)"),
    ("pole_axis_radius", 0.15, "Pole axis constraint (m)"),
    ("wire_curve_radius", 0.1, "Wire curve constraint (m)"),
    ("ambiguity_margin", 0.01, "Class/tree ambiguity margin (m)"),
]:
    add("refill", parameter_name, default_value, label, 0)
add("refill", "tree_height_min", 0.0, "Tree refill height (min, m)", 0)
add("refill", "tree_height_max", 10.0, "Tree refill height (max, m)", 0)


def defaults():
    return {
        stage_name: {
            parameter_key.split(".")[1]: copy.deepcopy(specification["default"])
            for parameter_key, specification in SCHEMA.items()
            if specification["stage"] == stage_name
        }
        for stage_name in STAGES
    }


def validate(config):
    if set(config) != set(STAGES):
        raise ValueError("Missing or unknown configuration stages")
    if isinstance(config.get("wires"), dict):
        config["wires"].setdefault("dense_direction_tolerance_deg", 0.0)
        config["wires"].setdefault("recovery_direction_tolerance_deg", 30.0)
        config["wires"].setdefault("recovery_min_coverage", 0.5)
        config["wires"].setdefault("recovery_sigma_multiplier", 3.0)
    for stage in STAGES:
        expected = {
            parameter_key.split(".")[1]
            for parameter_key, item in SCHEMA.items()
            if item["stage"] == stage
        }
        if set(config[stage]) != expected:
            raise ValueError(stage + ": Missing or unknown parameters")
    for key, specification in SCHEMA.items():
        group_name, parameter_name = key.split(".")
        value = config[group_name][parameter_name]
        default = specification["default"]
        if type(default) is bool:
            valid_type = type(value) is bool
        elif type(default) is int:
            valid_type = type(value) is int
        elif type(default) is float:
            valid_type = type(value) in (float, int) and math.isfinite(value)
        else:
            valid_type = isinstance(value, type(default))
        if not valid_type:
            raise ValueError(key + ": Invalid parameter type or non-finite number")
        if specification["choices"] and value not in specification["choices"]:
            raise ValueError(key + ": Invalid option")
        if specification["minimum"] is not None and value < specification["minimum"]:
            raise ValueError(key + ": Below minimum")
        if specification["maximum"] is not None and value > specification["maximum"]:
            raise ValueError(key + ": Above maximum")
    for group_name in ["stumps", "trunks"]:
        if config[group_name]["height_max"] <= config[group_name]["height_min"]:
            raise ValueError(group_name + ": Upper height must exceed lower height")
        if (
            config[group_name]["max_cluster_size"]
            <= config[group_name]["min_cluster_size"]
        ):
            raise ValueError(group_name + ": Maximum cluster size must exceed minimum")
    if config["wires"]["max_fit_segments"] < config["wires"]["min_fit_segments"]:
        raise ValueError("Maximum fit segments must be at least the minimum")
    for group_name in ["poles", "wires"]:
        feature_weights = config[group_name]["feature_weights"]
        if (
            len(feature_weights) != 6
            or any(
                type(element) not in (int, float)
                or not math.isfinite(element)
                or element < 0
                for element in feature_weights
            )
            or not any(feature_weights)
        ):
            raise ValueError(
                group_name
                + ": Provide six finite non-negative weights, at least one positive"
            )
    bounds = config["roi"]["bounds"]
    if (
        len(bounds) != 6
        or any(
            type(element) not in (float, int) or not math.isfinite(element)
            for element in bounds
        )
        or any(bounds[i] >= bounds[i + 1] for i in (0, 2, 4))
    ):
        raise ValueError("ROI bounds require six valid lower/upper limits")
    if any(
        type(reason_code) is not int or reason_code not in range(1, 7)
        for reason_code in config["refill"]["eligible_reasons"]
    ):
        raise ValueError("Refill reason codes must be in 1-6")
    if config["refill"]["tree_height_max"] <= config["refill"]["tree_height_min"]:
        raise ValueError("Invalid tree refill height range")
    return config
