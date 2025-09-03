# Domain wall lensing simulation against a starry background
# - Single-figure rendering (no subplots), matplotlib only
# - Stars vary in luminosity and color (blackbody approximation)
# - Toy model for an infinite planar wall causing a piecewise-constant deflection
#
# Outputs (written to the current working directory):
#   - domain_wall_static.png
#   - domain_wall_simulation.gif  (set MAKE_GIF=False to skip)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------- CONFIG --------------------
SEED = 7                # RNG seed (reproducible stars)
N_STARS = 1500          # Number of stars
WALL_THETA_DEG = 28.0   # Wall orientation: 0° = horizontal wall
DELTA = 0.045           # Deflection amplitude (fraction of FOV); think α/θ_FOV
THICKNESS = 0.010       # Wall thickness (controls smoothing of the "jump")
GRID_RES = 480          # Resolution for the faint wall overlay image
MAKE_GIF = True         # Toggle to render the animation
FRAMES = 72             # Animation frames
FPS = 24                # Animation framerate

PNG_PATH = "domain_wall_static.png"
GIF_PATH = "domain_wall_simulation.gif"

rng = np.random.default_rng(SEED)

# -------------------- Utilities --------------------
def temperature_to_rgb(temp_K: float):
    """
    Approximate blackbody color (1000K..40000K) -> (r,g,b) in [0,1].
    Tanner Helland / Bisqwit-style piecewise fit.
    """
    t = temp_K / 100.0
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        r = np.clip(r, 0.0, 255.0)

    if t <= 66.0:
        g = 99.4708025861 * np.log(max(t, 1e-6)) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    g = np.clip(g, 0.0, 255.0)

    if t >= 66.0:
        b = 255.0
    else:
        if t <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    b = np.clip(b, 0.0, 255.0)

    return (r/255.0, g/255.0, b/255.0)

def sample_star_temperatures(n, rng):
    """
    Rough mixture to resemble sky colors: many cool stars, fewer hot ones.
    """
    weights = np.array([0.28, 0.32, 0.22, 0.12, 0.06])  # cool -> hot
    counts = (weights * n).astype(int)
    counts[-1] += n - counts.sum()
    temps = np.concatenate([
        rng.uniform(3000, 4500, size=counts[0]),   # M/K: red/orange
        rng.uniform(4500, 6000, size=counts[1]),   # G: yellowish
        rng.uniform(6000, 7500, size=counts[2]),   # F: white
        rng.uniform(7500, 10000, size=counts[3]),  # A/B: white/blue-white
        rng.uniform(10000, 15000, size=counts[4])  # B/O: blue
    ])
    rng.shuffle(temps)
    return temps

def generate_star_field(n, rng):
    x = rng.random(n); y = rng.random(n)

    # Heavy-tailed brightness -> a few bright stars pop
    tail = rng.pareto(a=3.0, size=n) + 1.0
    tail = tail / np.max(tail)
    sizes = 2.0 + (tail ** 1.4) * 36.0          # marker sizes (pt^2)
    alphas = 0.35 + 0.65 * (tail ** 0.6)        # slightly more opaque for bright stars

    temps = sample_star_temperatures(n, rng)
    colors = [temperature_to_rgb(float(T)) for T in temps]

    # Make the brightest handful even more prominent
    bright_idx = np.argsort(tail)[-max(6, n//200):]
    sizes[bright_idx] *= 1.8
    alphas[bright_idx] = np.minimum(1.0, alphas[bright_idx] + 0.2)

    return x, y, sizes, alphas, np.array(colors)

def wall_normal(theta_deg):
    """
    Wall is a straight line with direction angle theta_deg (in degrees).
    Returns the unit normal vector pointing across the wall.
    """
    theta = np.deg2rad(theta_deg)
    t = np.array([np.cos(theta), np.sin(theta)])  # direction along the wall
    n = np.array([-t[1], t[0]])                   # rotate +90° to get normal
    return n / np.linalg.norm(n)

def displacement_across_wall(x, y, point_on_wall, nvec, delta, thickness):
    """
    Signed distance s to the wall:
        s = n · (r - r0),  r=(x,y), r0=point_on_wall.
    Smooth step of the deflection across the wall:
        frac = 0.5 * (1 - tanh(s/thickness))
    For s >> 0: no shift; for s << 0: shift ≈ delta * n.
    To make a perfectly sharp topological 'cut', replace the tanh line with:
        frac = (s < 0).astype(float)
    """
    s = nvec[0]*(x - point_on_wall[0]) + nvec[1]*(y - point_on_wall[1])
    frac = 0.5 * (1.0 - np.tanh(s / thickness))
    dx = frac * delta * nvec[0]
    dy = frac * delta * nvec[1]
    return dx, dy

def make_wall_overlay(H, W, point_on_wall, nvec, thickness, alpha_max=0.35):
    """
    Faint white ribbon to indicate the wall's location (purely visual).
    """
    xs = np.linspace(0.0, 1.0, W)
    ys = np.linspace(0.0, 1.0, H)
    X, Y = np.meshgrid(xs, ys)
    s = nvec[0]*(X - point_on_wall[0]) + nvec[1]*(Y - point_on_wall[1])
    sigma = thickness * 1.5
    glow = np.exp(-(s**2) / (2.0 * sigma**2))
    rgba = np.zeros((H, W, 4), dtype=float)
    rgba[..., :3] = 1.0
    rgba[..., 3] = alpha_max * (glow / (np.max(glow) + 1e-12))
    return rgba

# -------------------- Generate data --------------------
x, y, sizes, alphas, colors = generate_star_field(N_STARS, rng)
nvec = wall_normal(WALL_THETA_DEG)

# -------------------- Static frame --------------------
p_mid = np.array([0.5, 0.5])  # wall through the center
dx, dy = displacement_across_wall(x, y, p_mid, nvec, DELTA, THICKNESS)
x2, y2 = x + dx, y + dy

fig = plt.figure(figsize=(7.2, 7.2), facecolor="black")
ax = plt.axes([0, 0, 1, 1])
ax.set_facecolor("black")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

ax.scatter(x2, y2, s=sizes, c=colors, alpha=alphas, edgecolors="none")
overlay_img = make_wall_overlay(GRID_RES, GRID_RES, p_mid, nvec, THICKNESS, 0.35)
ax.imshow(overlay_img, origin="lower", extent=[0, 1, 0, 1], zorder=3)

ax.text(0.015, 0.98, "Domain wall lensing (static frame, toy model)",
        color="white", fontsize=12, ha="left", va="top", alpha=0.9,
        transform=ax.transAxes)

fig.savefig(PNG_PATH, dpi=220, facecolor=fig.get_facecolor())
plt.close(fig)

# -------------------- Animation (optional) --------------------
if MAKE_GIF:
    start_offset, end_offset = -0.6, +0.6   # move wall across the FOV along its normal
    offsets = np.linspace(start_offset, end_offset, FRAMES)

    fig = plt.figure(figsize=(6.8, 6.8), facecolor="black")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    sc = ax.scatter(x, y, s=sizes, c=colors, alpha=alphas, edgecolors="none")
    overlay_img = make_wall_overlay(GRID_RES, GRID_RES,
                                    point_on_wall=np.array([0.5, 0.5]) + start_offset * nvec,
                                    nvec=nvec, thickness=THICKNESS, alpha_max=0.35)
    im = ax.imshow(overlay_img, origin="lower", extent=[0, 1, 0, 1], zorder=3)

    base_positions = np.column_stack((x, y))

    def init():
        sc.set_offsets(base_positions)
        im.set_data(overlay_img)
        return sc, im

    def animate(frame):
        p = np.array([0.5, 0.5]) + offsets[frame] * nvec
        dx, dy = displacement_across_wall(x, y, p, nvec, DELTA, THICKNESS)
        new_pos = base_positions + np.column_stack((dx, dy))
        sc.set_offsets(new_pos)
        ov = make_wall_overlay(GRID_RES, GRID_RES, p, nvec, THICKNESS, 0.35)
        im.set_data(ov)
        return sc, im

    ani = animation.FuncAnimation(fig, animate, frames=FRAMES, init_func=init,
                                  interval=1000/FPS, blit=True)
    ani.save(GIF_PATH, writer="pillow", fps=FPS)
    plt.close(fig)

print(f"Saved: {PNG_PATH}")
if MAKE_GIF:
    print(f"Saved: {GIF_PATH}")
