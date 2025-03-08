import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Progress bar library

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def compute_mandelbrot_region(region):
    x_min, x_max, y_min, y_max, width, height, max_iter = region
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    mandelbrot_set = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = complex(real[j], imag[i])
            mandelbrot_set[i, j] = mandelbrot(c, max_iter)

    return mandelbrot_set

def generate_multiple_mandelbrot_cpu(regions, max_iter):
    # Add max_iter to each region tuple
    regions_with_iter = [(x_min, x_max, y_min, y_max, width, height, max_iter) for x_min, x_max, y_min, y_max, width, height in regions]

    # Use multiprocessing to compute Mandelbrot sets in parallel
    with Pool(cpu_count()) as pool:
        # Use tqdm to display a progress bar
        mandelbrot_sets = list(tqdm(pool.imap(compute_mandelbrot_region, regions_with_iter), total=len(regions)))

    return mandelbrot_sets

if __name__ == '__main__':
    # Parameters for multiple Mandelbrot sets
    regions = [
        (-2.0, 1.0, -1.5, 1.5, 2000, 2000),  # Full Mandelbrot set
        (-0.74877, -0.74872, 0.065053, 0.065103, 2000, 2000),  # Zoomed-in region
        (-0.1, 0.1, -0.1, 0.1, 2000, 2000)  # Another zoomed-in region
    ]
    max_iter = 1000

    # Generate multiple Mandelbrot sets in parallel
    mandelbrot_sets = generate_multiple_mandelbrot_cpu(regions, max_iter)

    # Plot the results
    fig, axes = plt.subplots(1, len(regions), figsize=(15, 5))
    for i, ax in enumerate(axes):
        x_min, x_max, y_min, y_max, width, height = regions[i]
        ax.imshow(
            mandelbrot_sets[i],
            extent=(x_min, x_max, y_min, y_max),
            cmap='hot',
            interpolation='bilinear'
        )
        ax.set_title(f"Mandelbrot Set {i+1}")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
    plt.tight_layout()
    plt.show()