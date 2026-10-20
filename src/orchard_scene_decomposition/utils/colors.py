import numpy as np

# Zeroth-order spherical harmonic constant used by 3D Gaussian Splatting.
SH_C0 = 0.28209479177387814
RGB_FIELDS = (
    ("red", "green", "blue"),
    ("r", "g", "b"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
)
DC_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")


def source_colors(vertices, ids=None):
    field_names = vertices.dtype.names or ()
    if ids is None:
        ids = np.arange(len(vertices), dtype=np.int64)
    else:
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError(
                "Color indices must be a 1D array of original integer point IDs"
            )
        if len(ids) and (ids.min() < 0 or ids.max() >= len(vertices)):
            raise ValueError("Color point index out of range")
    for fields in RGB_FIELDS:
        if all(
            field_name in field_names and vertices.dtype[field_name].kind in "uif"
            for field_name in fields
        ):
            normalized_rgb = all(
                vertices.dtype[field_name].kind == "f"
                and np.isfinite(vertices[field_name]).all()
                and (
                    not len(vertices)
                    or (
                        vertices[field_name].min() >= 0
                        and vertices[field_name].max() <= 1
                    )
                )
                for field_name in fields
            )
            rgb = np.column_stack(
                [vertices[field_name][ids] for field_name in fields]
            ).astype(np.float64)
            if normalized_rgb:
                rgb *= 255.0
            color_metadata = dict(
                kind="rgb",
                fields=list(fields),
                encoding="normalized_float" if normalized_rgb else "byte_values",
                description="Source RGB channels",
            )
            break
    else:
        if all(
            field_name in field_names and vertices.dtype[field_name].kind in "uif"
            for field_name in DC_FIELDS
        ):
            dc_coefficients = np.column_stack(
                [vertices[field_name][ids] for field_name in DC_FIELDS]
            ).astype(np.float64)
            rgb = (0.5 + SH_C0 * dc_coefficients) * 255.0
            color_metadata = dict(
                kind="sh_dc",
                fields=list(DC_FIELDS),
                encoding="sh_dc",
                description="Approximate RGB from 3DGS DC coefficients (no view dependence or opacity)",
            )
        else:
            rgb = np.full((len(ids), 3), 128.0)
            color_metadata = dict(
                kind="fallback",
                fields=[],
                encoding="gray",
                description="No RGB or 3DGS DC channels; displaying gray",
            )
    invalid_color_mask = ~np.isfinite(rgb).all(axis=1)
    if invalid_color_mask.any():
        rgb[invalid_color_mask] = 128.0
        color_metadata["invalid_display_rows"] = int(invalid_color_mask.sum())
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8), color_metadata
