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

t1 = time.perf_counter()

import numpy as np

t2 = time.perf_counter()

import matplotlib
import matplotlib.pyplot as plt

t3 = time.perf_counter()

print("\n=== Matplotlib ===")
print(f"Backend           : {matplotlib.get_backend()}")

# 疑似データ作成
x = np.linspace(0, 100, 2_000_000)
y = np.sin(x) + 0.1 * np.cos(10 * x)

t4 = time.perf_counter()

# プロット作成
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, lw=0.5)
ax.set_title("Plot speed check")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()

t5 = time.perf_counter()

# 保存
outfile = "check_plot_speed.png"
fig.savefig(outfile, dpi=150)

t6 = time.perf_counter()

print("\n=== Timing ===")
print(f"Startup only      : {t1 - t0:.3f} s")
print(f"Import numpy      : {t2 - t1:.3f} s")
print(f"Import matplotlib : {t3 - t2:.3f} s")
print(f"Make arrays       : {t4 - t3:.3f} s")
print(f"Build figure      : {t5 - t4:.3f} s")
print(f"Save figure       : {t6 - t5:.3f} s")
print(f"Total before show : {t6 - t0:.3f} s")
print(f"Saved file        : {os.path.abspath(outfile)}")

# 表示
print("\nNow calling plt.show() ...")
t7 = time.perf_counter()
plt.show()
t8 = time.perf_counter()

print(f"plt.show() time   : {t8 - t7:.3f} s")
print(f"Grand total       : {t8 - t0:.3f} s")