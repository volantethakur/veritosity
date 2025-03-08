import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numba import cuda
import cv2
import time
from tqdm import tqdm

def mandelbrot(c, m):
    z = 0
    for n in range(m):
        if abs(z) > 2:
            return n
        z = z*z + c
    return m

def compute_region(r):
    x1, x2, y1, y2, w, h, m = r
    real = np.linspace(x1, x2, w)
    imag = np.linspace(y1, y2, h)
    res = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            res[i, j] = mandelbrot(complex(real[j], imag[i]), m)
    return res

def gen_mandelbrot_cpu(rs, m, t=True):
    rs = [(x1, x2, y1, y2, w, h, m) for x1, x2, y1, y2, w, h in rs]
    s = time.time()
    e = ThreadPoolExecutor if t else ProcessPoolExecutor
    with e() as p:
        res = [f.result() for f in tqdm(as_completed([p.submit(compute_region, r) for r in rs]), total=len(rs))]
    return res, time.time() - s

@cuda.jit
def blur_kernel(img, k, out):
    i, j = cuda.grid(2)
    if i < img.shape[0] and j < img.shape[1]:
        v = 0.0
        for x in range(k.shape[0]):
            for y in range(k.shape[1]):
                xi, yj = i - k.shape[0] // 2 + x, j - k.shape[1] // 2 + y
                if 0 <= xi < img.shape[0] and 0 <= yj < img.shape[1]:
                    v += img[xi, yj] * k[x, y]
        out[i, j] = v

def gpu_blur(img, k):
    d_img, d_k = cuda.to_device(img), cuda.to_device(k)
    d_out = cuda.device_array_like(img)
    b = ((img.shape[0] + 15) // 16, (img.shape[1] + 15) // 16)
    s = time.time()
    blur_kernel[b, (16, 16)](d_img, d_k, d_out)
    d_out.copy_to_host(img)
    return img, time.time() - s

@cuda.jit
def matmul_kernel(a, b, c):
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        v = 0.0
        for k in range(a.shape[1]):
            v += a[i, k] * b[k, j]
        c[i, j] = v

def gpu_matmul(a, b):
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    d_a, d_b, d_c = cuda.to_device(a), cuda.to_device(b), cuda.to_device(c)
    b = ((c.shape[0] + 15) // 16, (c.shape[1] + 15) // 16)
    s = time.time()
    matmul_kernel[b, (16, 16)](d_a, d_b, d_c)
    d_c.copy_to_host(c)
    return c, time.time() - s

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(s, e):
    return [n for n in range(s, e) if is_prime(n)]

def bench_primes(l, t=True):
    p = 8
    c = l // p
    s = time.time()
    e = ThreadPoolExecutor if t else ProcessPoolExecutor
    with e() as ex:
        res = list(ex.map(find_primes, [(i * c, (i + 1) * c) for i in range(p)]))
    return [x for y in res for x in y], time.time() - s

def comprehensive_bench(g=False, t=True):
    print("\nStarting Benchmark...")
    print(f"Using {'GPU' if g else 'CPU'}")
    if not g:
        print(f"Using {'threads' if t else 'processes'}")

    rs = [
        (-2.0, 1.0, -1.5, 1.5, 2000, 2000),
        (-0.74877, -0.74872, 0.065053, 0.065103, 2000, 2000),
        (-0.1, 0.1, -0.1, 0.1, 2000, 2000)
    ]
    m = 1000
    img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    s = 2048
    a, b = np.random.rand(s, s).astype(np.float32), np.random.rand(s, s).astype(np.float32)
    l = 10**7

    if g:
        _, mt = gen_mandelbrot_cpu(rs, m, False)
        _, bt = gpu_blur(img, k)
        _, mtm = gpu_matmul(a, b)
        gf = (2 * s**3) / (mtm * 1e9)
    else:
        _, mt = gen_mandelbrot_cpu(rs, m, t)
        _, bt = gpu_blur(img, k)
        _, mtm = gpu_matmul(a, b)
        _, pt = bench_primes(l, t)

    print("\n--- Results ---")
    print(f"Mandelbrot: {mt:.2f}s")
    print(f"Blur: {bt:.2f}s")
    print(f"MatMul: {mtm:.2f}s")
    if not g:
        print(f"Primes: {pt:.2f}s")
    print(f"GFLOPS: {gf:.2f}\n")

if __name__ == '__main__':
    g, t = True, True
    comprehensive_bench(g, t)