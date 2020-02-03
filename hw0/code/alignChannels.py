import numpy as np


def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    blue_shift = find_alignment(blue, red)
    green_shift = find_alignment(green, red)

    return np.stack((red, green_shift, blue_shift), axis=2)

# Computing the metrics (SSD)


def find_alignment(image1, image2):
    height = image1.shape[0]
    width = image1.shape[1]
    best_alignment = 0
    min_cost = height * width * 500
    for dy in range(-30, 31, 1):
        for dx in range(-30, 31, 1):
            shift_image1 = np.roll(image1, (dy, dx), (0, 1))
            if dy > 0:
                shift_image1[:dy, :] = 0
            else:
                shift_image1[dy:, :] = 0
            if dx > 0:
                shift_image1[:, :dx] = 0
            else:
                shift_image1[:, dx:] = 0

            cur_cost = np.sum(np.abs((image2 - shift_image1) ** 2))

            if cur_cost < min_cost:
                min_cost = cur_cost
                best_alignment = shift_image1
    return best_alignment
