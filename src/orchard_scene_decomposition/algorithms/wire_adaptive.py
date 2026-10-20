"""Check a piecewise wire fit at the picked endpoints and correct it if needed."""

import numpy as np


def predict_piecewise_curve(x, breakpoints, coefficients):
    design_matrix = np.column_stack(
        [np.ones(len(x)), x - breakpoints[0]]
        + [np.maximum(x - break_point, 0) for break_point in breakpoints[1:-1]]
    )
    return design_matrix @ coefficients


def evaluate_recovery_at_aligned_x(result, prefix, queries):
    from .pipeline import FacilityPairError

    basis = result[prefix + "_recovery_basis"]
    origin = result[prefix + "_recovery_origin"]
    coefficients = result[prefix + "_recovery_coefficients"]
    fit_origin, fit_span = result[prefix + "_recovery_fit_domain"]
    aligned_polynomial = coefficients @ basis[:, 1:].T
    aligned_polynomial[0] += origin + (fit_origin + fit_span / 2) * basis[:, 0]
    aligned_polynomial[1] += fit_span / 2 * basis[:, 0]
    if len(aligned_polynomial) > 3:
        raise FacilityPairError("Unsupported adaptive recovery degree")
    reference_parameters = (
        2 * (((queries - origin) @ basis)[:, 0] - fit_origin) / fit_span - 1
    )
    quadratic_coefficient = (
        aligned_polynomial[2, 0] if len(aligned_polynomial) > 2 else 0.0
    )
    linear_coefficient = aligned_polynomial[1, 0]
    constant_term = aligned_polynomial[0, 0] - queries[:, 0]
    if abs(quadratic_coefficient) < 1e-10:
        if abs(linear_coefficient) < 1e-10:
            raise FacilityPairError(
                "Adaptive curve has no invertible work-X coordinate"
            )
        curve_parameters = -constant_term / linear_coefficient
    else:
        discriminant = (
            linear_coefficient * linear_coefficient
            - 4 * quadratic_coefficient * constant_term
        )
        if np.any(discriminant < 0):
            raise FacilityPairError("Adaptive curve has no real work-X solution")
        positive_root = (-linear_coefficient + np.sqrt(discriminant)) / (
            2 * quadratic_coefficient
        )
        negative_root = (-linear_coefficient - np.sqrt(discriminant)) / (
            2 * quadratic_coefficient
        )
        curve_parameters = np.where(
            abs(positive_root - reference_parameters)
            < abs(negative_root - reference_parameters),
            positive_root,
            negative_root,
        )
    derivative = (
        aligned_polynomial[1] + 2 * curve_parameters[:, None] * aligned_polynomial[2]
        if len(aligned_polynomial) > 2
        else np.tile(aligned_polynomial[1], (len(queries), 1))
    )
    x_derivative = derivative[:, 0]
    if np.any(abs(x_derivative) < 1e-8) or (
        np.any(x_derivative > 0) and np.any(x_derivative < 0)
    ):
        raise FacilityPairError("Adaptive curve is not monotone in work-X")
    curve = (
        np.polynomial.polynomial.polyvander(
            curve_parameters, len(aligned_polynomial) - 1
        )
        @ aligned_polynomial
    )
    tangent = derivative / np.linalg.norm(derivative, axis=1, keepdims=True)
    if not np.isfinite(curve).all() or not np.isfinite(tangent).all():
        raise FacilityPairError("Adaptive curve contains non-finite coordinates")
    return curve, tangent


def correct_wire_fit(project, config, pair_index, pair, diagnostics, piecewise_result):
    from .pipeline import FacilityPairError, require_facility_points, wire_point_mask
    from .wire_recovery import recover_wire

    prefix = "wire_" + str(pair_index)
    piecewise_arrays = piecewise_result[1]
    endpoint_predictions = np.column_stack(
        [
            predict_piecewise_curve(
                pair[:, 0],
                piecewise_arrays[prefix + "_breakpoints_" + axis],
                piecewise_arrays[prefix + "_coefficients_" + axis],
            )
            for axis in ("y", "z")
        ]
    )
    endpoint_errors = np.linalg.norm(pair[:, 1:] - endpoint_predictions, axis=1)
    diagnostics.update(
        adaptive_endpoint_errors=endpoint_errors.tolist(),
        adaptive_triggered=bool(np.any(endpoint_errors > config["fit_tolerance"])),
        adaptive_status="consistent",
    )
    if not diagnostics["adaptive_triggered"]:
        return piecewise_result
    corrected_info = {}
    try:
        _, result, features = recover_wire(
            project, config, pair_index, pair, corrected_info
        )
        candidate_ids = result[prefix + "_candidate_ids"]
        fit_points = project.arrays["xyz"][candidate_ids]
        curve, tangents = evaluate_recovery_at_aligned_x(result, prefix, fit_points)
        endpoint_curve, _ = evaluate_recovery_at_aligned_x(result, prefix, pair)
        corrected_endpoint_errors = np.linalg.norm(
            pair[:, 1:] - endpoint_curve[:, 1:], axis=1
        )
        if np.any(corrected_endpoint_errors > config["fit_tolerance"]):
            raise FacilityPairError(
                "Adaptive curve misses endpoints in work-X/YZ coordinates"
            )
        filtered_ids = candidate_ids[
            np.linalg.norm(fit_points[:, 1:] - curve[:, 1:], axis=1)
            < config["fit_tolerance"]
        ]
        require_facility_points(
            filtered_ids, "No points survived adaptive fit-residual filtering"
        )
        result[prefix + "_fit_inlier_ids"] = filtered_ids.copy()
        result[prefix + "_curve"] = curve
        filter_info = {}
        if config["point_filter_enabled"]:
            fit_inlier_mask = (
                np.linalg.norm(fit_points[:, 1:] - curve[:, 1:], axis=1)
                < config["fit_tolerance"]
            )
            filtered_ids = filtered_ids[
                wire_point_mask(
                    fit_points,
                    project.arrays["xyz"][filtered_ids],
                    config,
                    filter_info,
                    tangents[fit_inlier_mask],
                )
            ]
        require_facility_points(
            filtered_ids, "No points survived adaptive point filtering"
        )
        result[prefix + "_selected_ids"] = filtered_ids
    except (FacilityPairError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics.update(
            adaptive_status="kept_original",
            adaptive_fallback_reason=str(exc),
            adaptive_recovery_diagnostics=corrected_info,
        )
        return piecewise_result
    original_info = dict(diagnostics)
    diagnostics.update(corrected_info)
    diagnostics.update(filter_info)
    diagnostics.update(
        fit_method="axis_adaptive",
        adaptive_status="corrected",
        adaptive_triggered=True,
        adaptive_endpoint_errors=endpoint_errors.tolist(),
        adaptive_corrected_endpoint_errors=corrected_endpoint_errors.tolist(),
        adaptive_original_fit=original_info,
        fit_inlier_count=len(result[prefix + "_fit_inlier_ids"]),
    )
    return filtered_ids, result, features
