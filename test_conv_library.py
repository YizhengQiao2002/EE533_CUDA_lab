# test_conv_library.py
import ctypes, subprocess, time, sys, os
import numpy as np
import matplotlib.pyplot as plt


DLL_NAME = "./conv_lib.dll"     
CPU_EXE  = "./conv_cpu.exe"     
CUDA_EXE = "./conv_compare.exe" 


lib = ctypes.cdll.LoadLibrary(DLL_NAME)


lib.gpu_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
]

sizes = [512, 1024, 2048]
kernel_sizes = [3,5,7]

gpu_times = []
cpu_times = []
cudaexe_times = []

def run_external(exe, args):
    proc = subprocess.Popen([exe] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return out, err, proc.returncode

for M in sizes:
    A = (np.random.rand(M, M) * 255).astype(np.uint32)
    for N in kernel_sizes:
        K = np.ones((N,N), dtype=np.int32)
        out = np.zeros((M, M), dtype=np.uint32)

  
        lib.gpu_convolve(A.ravel(), M, K.ravel(), N, out.ravel())
        t0 = time.time()
        lib.gpu_convolve(A.ravel(), M, K.ravel(), N, out.ravel())
        t1 = time.time()
        gpu_elapsed = t1 - t0

        gpu_times.append((M, N, gpu_elapsed))
        print(f"[GPU DLL] M={M} N={N} time={gpu_elapsed:.6f} s")

        out_txt, err_txt, rc = run_external(CPU_EXE, [])
        cpu_time = None
        for line in out_txt.splitlines():
            if f"M={M} " in line and f"N={N} " in line and "filter=blur" in line:
                idx = line.find("time=")
                if idx>=0:
                    val = line[idx+5:].split()[0]
                    try:
                        cpu_time = float(val)
                        break
                    except:
                        pass
        if cpu_time is None:
            cpu_time = None
        cpu_times.append((M,N,cpu_time))
        if cpu_time is not None:
            print(f"[CPU EXE]  M={M} N={N} time={cpu_time:.6f} s")
        else:
            print(f"[CPU EXE]  M={M} N={N} time=PARSE_FAIL")

       
        out_txt, err_txt, rc = run_external(CUDA_EXE, [])
        cuda_time = None
        for line in out_txt.splitlines():
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    m0 = int(parts[0])
                    n0 = int(parts[1])
                except:
                    continue
                if m0 == M and n0 == N and parts[2].startswith("blur"):
                    try:
                        cuda_time = float(parts[5]) 
                        cuda_total_s = float(parts[4])
                        cuda_time = cuda_total_s
                        break
                    except:
                        pass
        cudaexe_times.append((M,N,cuda_time))
        if cuda_time is not None:
            print(f"[CUDA EXE] M={M} N={N} time={cuda_time:.6f} s")
        else:
            print(f"[CUDA EXE] M={M} N={N} time=PARSE_FAIL")


import collections
gdict = {(m,n):t for (m,n,t) in gpu_times}
cdict = {(m,n):t for (m,n,t) in cpu_times if t is not None}
xdict = {(m,n):t for (m,n,t) in cudaexe_times if t is not None}


plt.figure(figsize=(8,6))
for N in kernel_sizes:
    xs = sizes
    y_gpu = [gdict.get((M,N), None) for M in xs]
    y_cpu = [cdict.get((M,N), None) for M in xs]
    y_xe = [xdict.get((M,N), None) for M in xs]
   
    import math
    yg = [v if v is not None else math.nan for v in y_gpu]
    yc = [v if v is not None else math.nan for v in y_cpu]
    yx = [v if v is not None else math.nan for v in y_xe]
    plt.plot(xs, yg, marker='o', label=f'GPU DLL N={N}')
    plt.plot(xs, yc, marker='x', linestyle='--', label=f'CPU EXE N={N}')
    plt.plot(xs, yx, marker='s', linestyle=':', label=f'CUDA EXE N={N}')

plt.xlabel('Image size (M)')
plt.ylabel('Time (s)')
plt.title('Convolution: CPU vs GPU (DLL) vs CUDA exe')
plt.legend()
plt.grid(True)
plt.savefig('conv_compare_plot.png')
print("Saved conv_compare_plot.png")
plt.show()
