import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt


lib = ctypes.cdll.LoadLibrary("./matrix_lib.dll")


lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]


sizes = [512, 1024, 2048]
times = []

print("Python → CUDA tiled matrix multiplication")
print("------------------------------------------")

for N in sizes:
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)


    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)


    start = time.time()
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
    end = time.time()

    elapsed = end - start
    times.append(elapsed)

    print(f"N = {N:<4d} | Time = {elapsed:.4f} seconds")

print("------------------------------------------")


plt.figure()
plt.plot(sizes, times, marker='o')
plt.xlabel("Matrix size (N)")
plt.ylabel("Execution time (seconds)")
plt.title("Python → CUDA Tiled Matrix Multiplication Performance")
plt.grid(True)
plt.show()
