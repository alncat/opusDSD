import argparse
import os

import numpy as np
from scipy import ndimage

from cryodrgn import mrc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a solvent mask from an input 3D MRC volume."
    )
    parser.add_argument("input", type=os.path.abspath, help="input volume (.mrc/.map)")
    parser.add_argument(
        "-o",
        "--output",
        type=os.path.abspath,
        required=True,
        help="output mask MRC path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="absolute density threshold; if not set, use --percentile",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="percentile-based threshold when --threshold is not provided (default: %(default)s)",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="threshold on absolute density",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="keep only the largest connected component",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="fill holes in the binary mask",
    )
    parser.add_argument(
        "--open",
        type=int,
        default=0,
        help="binary opening iterations",
    )
    parser.add_argument(
        "--close",
        type=int,
        default=0,
        help="binary closing iterations",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="binary dilation iterations",
    )
    parser.add_argument(
        "--erode",
        type=int,
        default=0,
        help="binary erosion iterations",
    )
    parser.add_argument(
        "--soft-edge",
        type=float,
        default=0.0,
        help="soft edge width in voxels (0 -> hard mask)",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=None,
        help="create a spherical mask with this radius (voxels); bypasses thresholding",
    )
    parser.add_argument(
        "--sphere-center",
        type=float,
        nargs=3,
        default=None,
        metavar=("CZ", "CY", "CX"),
        help="sphere center in voxel coordinates (default: volume center)",
    )
    return parser.parse_args()


def _select_largest_component(mask):
    labels, n_comp = ndimage.label(mask)
    if n_comp == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return labels == largest


def _make_sphere_mask(shape, radius, center=None, soft_edge=0.0):
    if radius <= 0:
        raise ValueError("--sphere-radius must be > 0")

    z = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    x = np.arange(shape[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    if center is None:
        cz = (shape[0] - 1) * 0.5
        cy = (shape[1] - 1) * 0.5
        cx = (shape[2] - 1) * 0.5
    else:
        cz, cy, cx = [float(v) for v in center]

    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    if soft_edge > 0:
        mask = np.clip(1.0 - (dist - float(radius)) / float(soft_edge), 0.0, 1.0)
    else:
        mask = (dist <= float(radius)).astype(np.float32)
    return mask.astype(np.float32), (cz, cy, cx)


def make_mask(volume, args):
    v = np.asarray(volume, dtype=np.float32)

    if args.sphere_radius is not None:
        mask, sphere_center = _make_sphere_mask(
            v.shape,
            radius=float(args.sphere_radius),
            center=args.sphere_center,
            soft_edge=float(args.soft_edge),
        )
        return mask, None, "sphere", sphere_center

    score = np.abs(v) if args.use_abs else v

    if args.threshold is None:
        threshold = float(np.percentile(score, args.percentile))
    else:
        threshold = float(args.threshold)

    mask = score >= threshold

    if args.open > 0:
        mask = ndimage.binary_opening(mask, iterations=args.open)
    if args.close > 0:
        mask = ndimage.binary_closing(mask, iterations=args.close)
    if args.dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=args.dilate)
    if args.erode > 0:
        mask = ndimage.binary_erosion(mask, iterations=args.erode)
    if args.fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if args.largest_component:
        mask = _select_largest_component(mask)

    mask = mask.astype(np.float32)
    if args.soft_edge > 0:
        dist_out = ndimage.distance_transform_edt(mask < 0.5)
        soft = np.clip(1.0 - dist_out / float(args.soft_edge), 0.0, 1.0)
        mask = soft.astype(np.float32)

    return mask, threshold, "threshold", None


def main():
    args = parse_args()
    vol, header = mrc.parse_mrc(args.input)
    mask, threshold, mode, sphere_center = make_mask(vol, args)

    apix = header.get_apix()
    xorg, yorg, zorg = header.get_origin()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mrc.write(
        args.output,
        mask.astype(np.float32),
        Apix=apix,
        xorg=float(xorg),
        yorg=float(yorg),
        zorg=float(zorg),
        is_vol=True,
    )

    n_vox = np.prod(mask.shape)
    frac = float((mask > 0.5).sum()) / float(n_vox)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if mode == "threshold":
        print(f"Threshold used: {threshold:.6g}")
    else:
        print(f"Sphere radius (vox): {float(args.sphere_radius):.6g}")
        print(
            "Sphere center (vox z,y,x): "
            f"({sphere_center[0]:.3f}, {sphere_center[1]:.3f}, {sphere_center[2]:.3f})"
        )
    print(f"Mask occupancy (>0.5): {frac:.4%}")


if __name__ == "__main__":
    main()
