"""Fail-closed square-marker pose estimation for a downward camera."""

import cv2
import numpy as np


def checked_square_pose(
    corners,
    camera_matrix,
    distortion,
    size_m,
    *,
    min_depth_m=0.035,
    max_depth_m=0.090,
    max_reprojection_px=3.0,
    max_centre_disagreement_m=0.006,
):
    """Return validated rotation, translation and reprojection RMSE.

    Coordinates are expressed in the OpenCV optical camera frame.

    Image corners must be ordered:
    top-left, top-right, bottom-right, bottom-left.
    """

    pixels = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(4, 2)

    intrinsic = np.asarray(
        camera_matrix,
        dtype=np.float64,
    ).reshape(3, 3)

    if (
        not np.all(np.isfinite(pixels))
        or not np.all(np.isfinite(intrinsic))
        or not np.isfinite(size_m)
        or size_m <= 0
    ):
        raise ValueError(
            "Invalid square geometry or camera calibration"
        )

    half = size_m / 2.0

    object_points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    # Avoid IPPE_SQUARE: the installed version produced
    # an incorrect translation with a successful return.
    ok, rotation, translation = cv2.solvePnP(
        object_points,
        pixels,
        intrinsic,
        distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not ok:
        raise ValueError("SQPNP square pose failed")

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    ).reshape(3, 1)

    translation = np.asarray(
        translation,
        dtype=np.float64,
    ).reshape(3, 1)

    if (
        not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError("Nonfinite square pose")

    depth = float(translation[2, 0])

    # Physical camera-to-floor distance validation.
    if not min_depth_m <= depth <= max_depth_m:
        raise ValueError(
            f"Implausible camera-to-floor depth: {depth:.4f} m"
        )

    # Reproject the square into the original image.
    projected, _ = cv2.projectPoints(
        object_points,
        rotation,
        translation,
        intrinsic,
        distortion,
    )

    residual = (
        projected.reshape(4, 2) - pixels
    )

    rmse = float(np.sqrt(
        np.mean(np.sum(residual ** 2, axis=1))
    ))

    if (
        not np.isfinite(rmse)
        or rmse > max_reprojection_px
    ):
        raise ValueError(
            f"Square reprojection RMSE: {rmse:.2f} px"
        )

    # Independent image-centre cross-check.
    rays = cv2.undistortPoints(
        pixels.reshape(4, 1, 2),
        intrinsic,
        distortion,
    ).reshape(4, 2)

    ray_centre = rays.mean(axis=0) * depth

    centre_disagreement = float(np.linalg.norm(
        ray_centre - translation[:2, 0]
    ))

    if (
        not np.isfinite(centre_disagreement)
        or centre_disagreement > max_centre_disagreement_m
    ):
        raise ValueError(
            "Square centre disagreement: "
            f"{centre_disagreement:.4f} m"
        )

    return rotation, translation, rmse
