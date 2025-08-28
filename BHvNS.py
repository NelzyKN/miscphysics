import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter

def create_synthetic_lmc_background(width, height, num_stars=3000, noise_intensity=0.15):
    """Generates a synthetic starfield resembling the LMC."""
    # Base colors (deep space)
    background = np.zeros((height, width, 3))
    background[..., 0] = 0.01
    background[..., 1] = 0.01
    background[..., 2] = 0.05

    # Add stars
    # Generate coordinates and properties vectorized for efficiency
    coords = np.random.rand(num_stars, 2)
    coords[:, 0] *= height
    coords[:, 1] *= width
    # Power law distribution favors dimmer stars
    brightness = np.random.power(3, num_stars)

    for i in range(num_stars):
        y, x = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= y < height and 0 <= x < width:
            # Simple star color (slightly randomized white/yellow/blue)
            color = np.array([1.0, np.random.uniform(0.9, 1.0), np.random.uniform(0.8, 1.0)])
            # Additive brightness
            background[y, x] = np.clip(background[y, x] + color * brightness[i], 0, 1)

    # Add nebulosity (Approximation using Gaussian filters on noise)
    noise = np.random.rand(height, width)
    # Large structures
    nebulosity = gaussian_filter(noise, sigma=60) * noise_intensity * 2
    # Smaller structures
    nebulosity += gaussian_filter(noise, sigma=20) * noise_intensity
    
    # Add color to nebulosity
    background[..., 0] += nebulosity * 0.6 # Reddish tint
    background[..., 1] += nebulosity * 0.3 # Greenish tint
    background[..., 2] += nebulosity * 0.4 # Bluish tint

    return np.clip(background, 0, 1)

def apply_lensing_distortion(image, center, mass_param, rotation_param, object_type='black_hole'):
    """
    Applies a conceptual gravitational lensing and frame-dragging distortion.
    This is an artistic interpretation, not a physical ray-tracing simulation.
    """
    h, w, _ = image.shape
    # Create coordinate grids for the output image
    Y, X = np.ogrid[:h, :w]

    # Calculate distance (r) and angle (theta) from the center
    dx = X - center[0]
    dy = Y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Avoid division by zero at the center
    r[r == 0] = 0.001

    # 1. Gravitational Lensing (Bending effect)
    # Conceptual model: deflection is proportional to M/r.
    # We approximate this by mapping the apparent position (r) back to a modified source position (r_lensed).
    deflection = mass_param / r
    r_lensed = r - deflection
    # Ensure the source radius remains positive and within bounds
    r_lensed = np.clip(r_lensed, 0.001, np.max([h,w]))

    # 2. Frame-Dragging (Rotation effect)
    # Conceptual model: Spacetime is twisted. The effect decreases rapidly with distance (e.g., 1/r^2).
    twist_angle = rotation_param / (r**2)
    theta_twisted = theta + twist_angle

    # Calculate new source coordinates (X_new, Y_new) corresponding to the output pixels
    X_new = center[0] + r_lensed * np.cos(theta_twisted)
    Y_new = center[1] + r_lensed * np.sin(theta_twisted)

    # Interpolate the background image at the new coordinates using map_coordinates
    coords = [Y_new, X_new]
    distorted_image = np.zeros_like(image)
    for i in range(3):
        # Use order=3 (cubic interpolation) for smoother results
        distorted_image[..., i] = map_coordinates(image[..., i], coords, order=3, mode='reflect')

    # 3. Object Specific Modifications
    if object_type == 'black_hole':
        # Simulate the shadow (Event Horizon)
        shadow_radius = mass_param * 0.9 # Approximate shadow size based on mass parameter
        mask = r < shadow_radius
        distorted_image[mask] = 0

        # Simulate Photon Ring (brightening near the shadow edge)
        ring_mask = (r >= shadow_radius) & (r < shadow_radius * 1.15)
        # Apply a brightness boost to the ring
        distorted_image[ring_mask] = np.clip(distorted_image[ring_mask] * 1.8, 0, 1)

    elif object_type == 'naked_singularity':
        # Extreme brightness and blueshift near the center
        # The effect decreases rapidly with distance (1/r^2).
        brightness_boost = 6000 / (r**2 + 10) # Added constant to avoid infinite brightness at center
        brightness_boost = np.clip(brightness_boost, 0, 5)

        # Apply brightness boost
        distorted_image = distorted_image * (1 + brightness_boost[..., None])

        # Apply blueshift (conceptual high energy)
        distorted_image[..., 2] = distorted_image[..., 2] * (1 + brightness_boost*0.6) # Strong blue boost
        distorted_image[..., 1] = distorted_image[..., 1] * (1 + brightness_boost*0.3) # Moderate green boost

        # Normalize brightness if the center is blown out, maintaining color balance
        max_val = np.max(distorted_image)
        if max_val > 1:
             distorted_image = distorted_image / max_val

    return np.clip(distorted_image, 0, 1)

# --- Main Execution ---

# Parameters
WIDTH, HEIGHT = 800, 600
# These parameters are tuned for visual effect, not physical units
MASS = 150          # Controls lensing strength and shadow size
ROTATION = 8000     # Controls frame-dragging (twisting) strength

# 1. Generate Background
print("Generating synthetic LMC background...")
# Use a fixed seed for reproducibility if desired
# np.random.seed(42) 
background_img = create_synthetic_lmc_background(WIDTH, HEIGHT, num_stars=4000)

# 2. Simulate Black Hole
print("Simulating Black Hole lensing...")
# Center the black hole slightly to the left for the comparison view
bh_center = (WIDTH // 3, HEIGHT // 2)
black_hole_img = apply_lensing_distortion(
    background_img,
    bh_center,
    MASS,
    ROTATION,
    object_type='black_hole'
)

# 3. Simulate Naked Singularity
print("Simulating Naked Singularity lensing...")
# Center the singularity slightly to the right
ns_center = (2 * WIDTH // 3, HEIGHT // 2)
naked_singularity_img = apply_lensing_distortion(
    background_img,
    ns_center,
    MASS,
    ROTATION * 1.5, # Assuming a faster spin or stronger visible effect for the singularity visualization
    object_type='naked_singularity'
)

# 4. Display the comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor='black')

# Black Hole Plot
axes[0].imshow(black_hole_img)
axes[0].set_title("Black Hole (Kerr)", color='white', fontsize=16)
axes[0].axis('off')
# Crop the view to focus on the relevant area
axes[0].set_xlim(bh_center[0] - 250, bh_center[0] + 250)
axes[0].set_ylim(bh_center[1] + 250, bh_center[1] - 250) # Inverted Y axis for imshow

# Naked Singularity Plot
axes[1].imshow(naked_singularity_img)
axes[1].set_title("Naked Singularity (Hypothetical)", color='white', fontsize=16)
axes[1].axis('off')
# Crop the view
axes[1].set_xlim(ns_center[0] - 250, ns_center[0] + 250)
axes[1].set_ylim(ns_center[1] + 250, ns_center[1] - 250)

plt.tight_layout()
print("Displaying visualization.")
# plt.show() # Uncomment this to display the plot when running the script locally
