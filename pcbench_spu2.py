import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

def mandelbrot(c, max_iter):
    z = 0
    start_time = time.time()  # Start timing for a single pixel
    for n in range(max_iter):
        if abs(z) > 2:
            return n, time.time() - start_time  # Return iteration count and time taken
        z = z*z + c
    return max_iter, time.time() - start_time  # Return max_iter and time taken

def compute_mandelbrot_region(region):
    x_min, x_max, y_min, y_max, width, height, max_iter = region
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    mandelbrot_set = np.zeros((height, width), dtype=np.int32)
    total_time_per_pixel = 0.0
    total_pixels = width * height

    # Process rows in chunks to allow progress updates
    chunk_size = 10  # Number of rows to process per chunk
    for i in range(0, height, chunk_size):
        for row in range(i, min(i + chunk_size, height)):
            for col in range(width):
                c = complex(real[col], imag[row])
                iterations, pixel_time = mandelbrot(c, max_iter)
                mandelbrot_set[row, col] = iterations
                total_time_per_pixel += pixel_time

    return mandelbrot_set, total_time_per_pixel / total_pixels

def generate_multiple_mandelbrot_cpu(regions, max_iter):
    # Add max_iter to each region tuple
    regions_with_iter = [(x_min, x_max, y_min, y_max, width, height, max_iter) for x_min, x_max, y_min, y_max, width, height in regions]

    # Start timing
    start_time = time.time()

    # Use multiprocessing to compute Mandelbrot sets in parallel
    with Pool(cpu_count()) as pool:
        # Use tqdm to display a progress bar
        results = list(tqdm(pool.imap(compute_mandelbrot_region, regions_with_iter), total=len(regions)))

    # Separate Mandelbrot sets and average time per pixel
    mandelbrot_sets = [result[0] for result in results]
    avg_time_per_pixel = sum(result[1] for result in results) / len(results)

    # End timing
    total_time = time.time() - start_time

    return mandelbrot_sets, total_time, avg_time_per_pixel

def benchmark_cpu(regions, max_iter):
    print("Starting CPU benchmark...")
    print(f"Number of CPU cores: {cpu_count()}")
    print(f"Regions to compute: {len(regions)}")
    print(f"Resolution: {regions[0][4]}x{regions[0][5]} pixels")
    print(f"Maximum iterations: {max_iter}\n")

    # Generate Mandelbrot sets and measure time
    start_benchmark = time.time()
    mandelbrot_sets, total_time, avg_time_per_pixel = generate_multiple_mandelbrot_cpu(regions, max_iter)
    end_benchmark = time.time()

    # Benchmark results
    total_pixels = sum(width * height for _, _, _, _, width, height in regions)
    performance_score = total_pixels / (end_benchmark - start_benchmark) / 1e6  # Pixels per second (in millions)

    print("\n--- Benchmark Results ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per region: {total_time / len(regions):.2f} seconds")
    print(f"Average time per pixel: {avg_time_per_pixel * 1e6:.2f} microseconds")
    print(f"Total pixels processed: {total_pixels:,}")
    print(f"Performance score: {performance_score:.2f} MPixels/s\n")

    return mandelbrot_sets

if __name__ == '__main__':
    # Parameters for multiple Mandelbrot sets
    regions = [
        (-2.0, 1.0, -1.5, 1.5, 2000, 2000),  # Full Mandelbrot set
        (-0.74877, -0.74872, 0.065053, 0.065103, 2000, 2000),  # Zoomed-in region
        (-0.1, 0.1, -0.1, 0.1, 2000, 2000)  # Another zoomed-in region
    ]
    max_iter = 1000

    # Benchmark CPU and generate Mandelbrot sets
    mandelbrot_sets = benchmark_cpu(regions, max_iter)

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