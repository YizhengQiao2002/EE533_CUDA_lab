# quick_test_dll.py
import ctypes, numpy as np
lib = ctypes.cdll.LoadLibrary("./conv_lib.dll")   # Windows: conv_lib.dll ; Linux: ./libconv.so
lib.gpu_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
]

M = 512
A = (np.random.rand(M, M) * 255).astype(np.uint32)
K = np.ones((3,3), dtype=np.int32)
out = np.zeros((M*M,), dtype=np.uint32)

lib.gpu_convolve(A.ravel(), M, K.ravel(), 3, out)
print("call succeeded, out[0] =", out[0])
