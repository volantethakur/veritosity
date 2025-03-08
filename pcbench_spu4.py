import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numba import cuda
import cv2
import time
from tqdm import tqdm

# Mandelbrot Set (CPU)
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

def generate_mandelbrot_cpu(regions, max_iter, use_threads=True):
    regions_with_iter = [(x_min, x_max, y_min, y_max, width, height, max_iter) for x_min, x_max, y_min, y_max, width, height in regions]

    start_time = time.time()
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor() as executor:
        futures = [executor.submit(compute_mandelbrot_region, region) for region in regions_with_iter]
        results = [future.result() for future in tqdm(as_completed(futures), total=len(futures))]

    total_time = time.time() - start_time
    return results, total_time

# Gaussian Blur (GPU)
@cuda.jit
def gaussian_blur_kernel(image, kernel, output):
    row, col = cuda.grid(2)
    if row < image.shape[0] and col < image.shape[1]:
        value = 0.0
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                x = row - kernel.shape[0] // 2 + i
                y = col - kernel.shape[1] // 2 + j
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    value += image[x, y] * kernel[i, j]
        output[row, col] = value

def gpu_gaussian_blur(image, kernel):
    d_image = cuda.to_device(image)
    d_kernel = cuda.to_device(kernel)
    d_output = cuda.device_array_like(image)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_time = time.time()
    gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_image, d_kernel, d_output)
    d_output.copy_to_host(image)
    total_time = time.time() - start_time

    return image, total_time

# Matrix Multiplication (GPU)
@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def gpu_matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (C.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (C.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_time = time.time()
    matmul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    d_C.copy_to_host(C)
    total_time = time.time() - start_time

    return C, total_time

# Prime Number Calculation (CPU)
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(start, end):
    return [n for n in range(start, end) if is_prime(n)]

def benchmark_cpu_primes(limit, use_threads=True):
    num_processes = 8  # Fixed number of processes/threads
    chunk_size = limit // num_processes

    start_time = time.time()
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor() as executor:
        ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
        results = list(executor.starmap(find_primes, ranges))

    total_time = time.time() - start_time
    primes = [p for sublist in results for p in sublist]

    return primes, total_time

# Main Benchmark Function
def comprehensive_benchmark(use_gpu=False, use_threads=True):
    print("\nStarting Comprehensive Benchmark...")
    print(f"Using {'GPU' if use_gpu else 'CPU'} for computation.")
    if not use_gpu:
        print(f"Using {'threads' if use_threads else 'processes'} for CPU computation.")

    # Parameters for Mandelbrot
    regions = [
        (-2.0, 1.0, -1.5, 1.5, 2000, 2000),
        (-0.74877, -0.74872, 0.065053, 0.065103, 2000, 2000),
        (-0.1, 0.1, -0.1, 0.1, 2000, 2000)
    ]
    max_iter = 1000

    # Parameters for Gaussian Blur
    image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

    # Parameters for Matrix Multiplication
    size = 2048
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Parameters for Prime Number Calculation
    limit = 10**7

    # Run Benchmarks
    if use_gpu:
        print("\nRunning GPU Benchmarks...")
        _, mandelbrot_time = generate_mandelbrot_cpu(regions, max_iter, use_threads=False)  # CPU fallback for Mandelbrot
        _, blur_time = gpu_gaussian_blur(image, kernel)
        _, matmul_time = gpu_matmul(A, B)
        gflops = (2 * size**3) / (matmul_time * 1e9)
    else:
        print("\nRunning CPU Benchmarks...")
        _, mandelbrot_time = generate_mandelbrot_cpu(regions, max_iter, use_threads=use_threads)
        _, blur_time = gpu_gaussian_blur(image, kernel)  # GPU fallback for Gaussian Blur
        _, matmul_time = gpu_matmul(A, B)  # GPU fallback for Matrix Multiplication
        _, prime_time = benchmark_cpu_primes(limit, use_threads=use_threads)

    # Print Results
    print("\n--- Comprehensive Benchmark Results ---")
    print(f"Mandelbrot Set Time: {mandelbrot_time:.2f} seconds")
    print(f"Gaussian Blur Time: {blur_time:.2f} seconds")
    print(f"Matrix Multiplication Time: {matmul_time:.2f} seconds")
    if not use_gpu:
        print(f"Prime Number Calculation Time: {prime_time:.2f} seconds")
    print(f"Matrix Multiplication Performance: {gflops:.2f} GFLOPS\n")

if __name__ == '__main__':
    # Choose between CPU and GPU
    use_gpu = True  # Set to False to use CPU instead
    use_threads = True  # Only relevant if using CPU (True for threads, False for processes)

    # Run the comprehensive benchmark
    comprehensive_benchmark(use_gpu=use_gpu, use_threads=use_threads)