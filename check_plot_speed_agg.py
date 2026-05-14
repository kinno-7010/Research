import time
import sys
import os
import platform

t0 = time.perf_counter()

print("=== Environment ===")
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version}")
print(f"Platform          : {platform.platform()}")
print(f"Working dir       : {os.getcwd()}")

import numpy as np
import matplotlib

# 表示なし backend を強制
matplotlib.use("Agg")

import matplotlib.pyplot as plt

print("\n=== Matplotlib ===")
print(f"Backend           : {matplotlib.get_backend()}")

t1 = time.perf_counter()

x = np.linspace(0, 100, 2_000_000)
y = np.sin(x) + 0.1 * np.cos(10 * x)

t2 = time.perf_counter()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, lw=0.5)
ax.set_title("Plot speed check (Agg)")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()

t3 = time.perf_counter()

outfile = "check_plot_speed_agg.png"
fig.savefig(outfile, dpi=150)

t4 = time.perf_counter()

print("\n=== Timing ===")
print(f"Make arrays       : {t2 - t1:.3f} s")
print(f"Build figure      : {t3 - t2:.3f} s")
print(f"Save figure       : {t4 - t3:.3f} s")
print(f"Total             : {t4 - t0:.3f} s")
print(f"Saved file        : {os.path.abspath(outfile)}")