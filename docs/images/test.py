from PIL import Image
import numpy as np

def add_noise_to_specific_colors(image_path, target_colors, noise_level=10, tolerance=0):
    """
    Adds subtle random noise to pixels matching target colors in an image.

    Parameters:
    - image_path: str, path to the image
    - target_colors: list of RGB tuples, e.g. [(255, 0, 0)]
    - noise_level: int, max noise to add/subtract per channel (default: 10)
    - tolerance: int, how close a pixel needs to be to a target color to be considered a match
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)

    def matches_color(pixel, color, tolerance):
        return np.all(np.abs(pixel - color) <= tolerance)

    noisy_pixels = pixels.copy()

    for color in target_colors:
        mask = np.all(np.abs(pixels - color) <= tolerance, axis=-1)

        # Generate random noise within the specified range
        noise = np.random.randint(-noise_level, noise_level + 1, size=pixels.shape, dtype=np.int16)

        # Apply noise only where the mask is True
        noisy_pixels[mask] = np.clip(pixels[mask] + noise[mask], 0, 255)

    noisy_img = Image.fromarray(noisy_pixels.astype(np.uint8))
    return noisy_img

# Example usage:
image = add_noise_to_specific_colors("quasar_logo.png", target_colors=[(52, 67, 76), (113, 164, 165)], noise_level=8, tolerance=5)
image.save("quasar_logo.png")

