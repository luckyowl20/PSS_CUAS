import numpy as np
import matplotlib.pyplot as plt

# ---------- domain (height) ----------
# Avoid x=0 because g(x) ~ 1/x, and keep within the real domain of h(x) (x ≤ 150).
x = np.linspace(0.1, 150.0, 1500)

# ---------- functions ----------
def deg_per_sec(x):
    # g(x) = (15*180)/(pi*x)
    return (30 * 180.0) / (np.pi * x)

def time_vs_height(x):
    # h(x) = 5.5 - sqrt((121/600)*(150 - x))
    return 5.5 - np.sqrt((121.0/600.0) * (150.0 - x))

def deg_lead(x):
    # green curve = g(x) * h(x)
    return deg_per_sec(x) * time_vs_height(x)

# ---------- apply clipping to deg/s ----------
g_raw = deg_per_sec(x)
#g_clipped = np.minimum(g_raw, 50.0)   # cap at 50 deg/s

# ---------- 1) Purple: deg/s vs height (capped at 50) ----------
plt.figure()
plt.plot(x, g_raw, color="purple", linewidth=2)
plt.title("Angular rate (deg/s) vs Height")
plt.xlabel("height (m)")
plt.ylabel("deg/s")
plt.ylim(0, 100)
plt.xlim(0, 160)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# ---------- 2) Green: deg lead vs height ----------
plt.figure()
plt.plot(x, deg_lead(x), color="green", linewidth=2)
plt.title("Deg lead vs Height")
plt.xlabel("height (m)")
plt.ylabel("deg lead")
plt.xlim(0, 160)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# ---------- 3) Blue: time vs height ----------
plt.figure()
plt.plot(x, time_vs_height(x), color="blue", linewidth=2)
plt.title("Time vs Height")
plt.xlabel("height (m)")
plt.ylabel("Time")
plt.xlim(0, 160)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
