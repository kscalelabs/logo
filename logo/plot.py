"""Script for plotting the K-Scale logo."""

import numpy as np
import PIL.Image as I


def get_inner_image(
    pix: int,
    exp_factor: float = 4,
    exp_offset: tuple[float, float] = (-0.4, 0.4),
    outer_pad: float = 0.1,
    edge_pad: float = 0.1,
) -> np.ndarray:
    hpix_a, hpix_b = pix // 2, (pix + 1) // 2
    x, y = np.meshgrid(np.arange(-hpix_a, hpix_b), np.arange(-hpix_a, hpix_b))
    x = x.astype(np.float64) / hpix_a
    y = y.astype(np.float64) / hpix_a
    x_off, y_off = exp_offset
    arr = np.ones(shape=(pix, pix, 3), dtype=np.float64)
    exp_mask = np.exp((x + x_off) * exp_factor) < (y + y_off) * exp_factor
    arr[exp_mask] = 0.0
    arr[(x**2 + y**2) > (1.0 - edge_pad - outer_pad) ** 2] = 1.0
    arr[(x**2 + y**2) > (1.0 - edge_pad) ** 2] = 0.0
    return arr


def get_ring_image(
    pix: int,
    vsquish: float = 4.0,
    ring_warp: tuple[float, float] = (0.1, -0.05),
    edge_pad: float = 0.1,
) -> np.ndarray:
    hpix_a, hpix_b = pix // 2, (pix + 1) // 2
    x, y = np.meshgrid(np.arange(-hpix_a, hpix_b), np.arange(-hpix_a, hpix_b))
    x = x.astype(np.float64) / hpix_a
    y = y.astype(np.float64) / hpix_a * vsquish
    arr = np.ones(shape=(pix, pix, 3), dtype=np.float64)
    arr[(x**2 + y**2) > (1.0 - edge_pad) ** 2] = 0.0
    xwarp = 1.0 + ring_warp[0]
    ywarp = 1.0 + ring_warp[1]
    ring_mask = (x * xwarp) ** 2 + (y * ywarp) ** 2 < (1.0 - edge_pad) ** 2
    arr[ring_mask] = 0.0
    return arr


def get_image(
    pix: int,
    exp_factor: float = 5,
    exp_offset: tuple[float, float] = (-0.3, 0.3),
) -> np.ndarray:
    arr_a = get_inner_image(pix, exp_factor, exp_offset, 0.03, 0.5)
    arr_b = get_ring_image(pix)
    arr = (arr_a + arr_b).clip(0, 1)
    return arr


def main() -> None:
    # arr = get_image(1024)
    # arr = np.flipud(arr)

    arrs = []
    for i in (3, 4, 5):
        larrs = []
        for j in (-0.2, -0.3, -0.4):
            for k in (0.2, 0.3, 0.4):
                larrs.append(get_image(256, i, (j, k)))
        arrs.append(np.concatenate(larrs, axis=1))
    arr = np.concatenate(arrs, axis=0)
    arr = np.flipud(arr)

    img = I.fromarray((arr * 255).round().astype(np.uint8))
    img.show()


if __name__ == "__main__":
    main()
