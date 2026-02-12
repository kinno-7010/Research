import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Setup figure for the 3-phase scenario
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Common settings for solar disk
def draw_sun(ax):
    sun = patches.Circle((0, 0), 1, color='gold', zorder=10, label='Sun')
    ax.add_patch(sun)
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    # Solar surface line approximation
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)

# --- Phase 1: Faint CME & Slow Expansion ---
ax1 = axes[0]
draw_sun(ax1)
ax1.set_title("Phase 1: Faint CME Expansion\n(01:26 UT ~)", fontsize=14, weight='bold')

# Faint CME (Diffuse blob)
faint_cme = patches.Ellipse((1.8, 0), 1.5, 1.2, angle=0, color='gray', alpha=0.3)
ax1.add_patch(faint_cme)
ax1.text(1.8, 0, "Faint CME\n(No Sheath)", ha='center', fontsize=10, color='black')

# "EUV Wave" representation (Slow expansion of loops)
# Instead of a shock wave, we draw stretching loops
theta_loop = np.linspace(-0.5, 0.5, 50)
r_loop = 1.0 + 0.5 * np.cos(theta_loop*3) # Loop shape
ax1.plot(1.2 * np.cos(theta_loop*2), 1.2 * np.sin(theta_loop*2), 'k--', alpha=0.6)
ax1.text(0.8, 0.8, "Slow Expansion\n(~40 km/s)\n= Field Stretching", fontsize=10, color='gray')
ax1.annotate("", xy=(1.5, 0.6), xytext=(1.1, 0.3), arrowprops=dict(arrowstyle="->", color='gray'))

# --- Phase 2: Seeding & Turbulence ---
ax2 = axes[1]
draw_sun(ax2)
ax2.set_title("Phase 2: Wake & Seeding\n(02:00 - 03:00 UT)", fontsize=14, weight='bold')

# Faint CME moved further out
faint_cme_far = patches.Ellipse((3.5, 0), 2.0, 1.5, angle=0, color='gray', alpha=0.15)
ax2.add_patch(faint_cme_far)

# Turbulent Wake & Seeds
# Random dots for seed particles
np.random.seed(42)
seeds_x = np.random.uniform(1.2, 3.0, 50)
seeds_y = np.random.uniform(-0.8, 0.8, 50)
ax2.scatter(seeds_x, seeds_y, s=10, c='red', marker='.', label='Seed Particles')

# Turbulence lines (squiggly)
t = np.linspace(1.2, 3.0, 100)
for offset in [-0.5, 0, 0.5]:
    turb = offset + 0.05 * np.sin(20 * t)
    ax2.plot(t, turb, 'b-', alpha=0.2, linewidth=0.5)

ax2.text(2.0, -1.2, "Turbulent Wake\n+ Seed Electrons", ha='center', fontsize=12, color='darkred', weight='bold')


# --- Phase 3: Main CME Interaction ---
ax3 = axes[2]
draw_sun(ax3)
ax3.set_title("Phase 3: Interaction & Type II\n(03:12 UT ~)", fontsize=14, weight='bold')

# Faint CME (Far away)
faint_cme_very_far = patches.Ellipse((5.0, 0), 2.2, 1.8, angle=0, color='gray', alpha=0.1)
ax3.add_patch(faint_cme_very_far)

# Main CME (Fast Eruption)
main_cme = patches.Wedge((0,0), 2.5, -30, 30, width=0.5, color='red', alpha=0.4, label='Main CME')
ax3.add_patch(main_cme)

# Shock Front (Main CME driving shock)
theta_shock = np.linspace(-0.6, 0.6, 50)
x_shock = 2.6 * np.cos(theta_shock)
y_shock = 2.6 * np.sin(theta_shock)
ax3.plot(x_shock, y_shock, 'r-', linewidth=3, label='Shock Front')

# Interaction Region (Seeds hitting Shock)
# Highlight the overlap area
ax3.scatter(seeds_x[seeds_x > 2.2], seeds_y[seeds_x > 2.2], s=30, c='yellow', edgecolors='red', zorder=20)

# Radio Emission (Zig-zag lines)
x_radio = np.linspace(2.6, 3.5, 20)
y_radio = 0.5 * np.ones_like(x_radio)
y_zigzag = y_radio + 0.1 * np.cos(30 * x_radio)
ax3.plot(x_radio, y_zigzag, color='limegreen', linewidth=2)
ax3.text(3.0, 0.7, "Type II Radio\nEmission", color='green', fontsize=12, weight='bold')

# Annotation for interaction
ax3.annotate("Shock hits Seeds!", xy=(2.6, 0.2), xytext=(3.5, -1.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

plt.tight_layout()
plt.savefig('scenario_schematic.png', dpi=150)
plt.show()