import numpy as np
import matplotlib.pyplot as plt

def generate_stars(num_stars=1500, seed=42):
    """Generates random star coordinates in a larger field (-2 to 2) to allow for distortion."""
    # Use a fixed seed for reproducibility
    np.random.seed(seed)
    # Generate stars in a wider area than the viewport
    return np.random.rand(num_stars, 2) * 4 - 2

def plot_approach_frame(stars, distortion_factor, wall_thickness, wall_glow, title):
    """
    Plots a frame of the simulation, applying distortion to stars and visualizing the wall.
    
    Args:
        stars (np.array): Base star coordinates.
        distortion_factor (float): Strength of the gravitational lensing (simulates proximity).
        wall_thickness (float): Visual thickness of the wall's core.
        wall_glow (float): Brightness of the wall's glow (0 to 1).
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('black')
    
    # Define the viewing boundaries
    Y_MAX = 1.5
    X_MAX = 1.5
    ax.set_xlim(-X_MAX, X_MAX)
    ax.set_ylim(-Y_MAX, Y_MAX)
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=14)

    # --- Apply Distortion (Repulsive Gravitational Lensing) ---
    x = stars[:, 0]
    y = stars[:, 1]

    # Phenomenological Model: Repulsion increases closer to the wall (x=0).
    # The deflection scales with distortion_factor and falls off with distance.
    # The (1 + np.abs(x*5))**2 term controls how localized the effect is near the wall.
    deflection = distortion_factor / (1 + np.abs(x*5))**2
    
    # Apply the deflection away from the center (x=0)
    x_distorted = x + np.sign(x) * deflection

    # Plot stars
    ax.scatter(x_distorted, y, color='white', s=1.5, alpha=0.9)

    # --- Plot the Domain Wall Visualization ---
    if wall_thickness > 0:
        # 1. Main wall structure (the core of the scalar field)
        ax.axvspan(-wall_thickness/2, wall_thickness/2, color='#0088ff', alpha=wall_glow*0.6)

        # 2. Energy glow
        glow_extent = min(X_MAX, wall_thickness * 10 + distortion_factor*0.5)
        
        # Create a horizontal gradient for the glow effect
        gradient_res = 256
        gradient_h = np.linspace(1, 0, gradient_res).reshape(1, -1)

        # Apply the glow on both sides using the 'plasma' colormap.
        
        # Right side glow (High energy near x=0). We reverse the gradient array [:, ::-1].
        ax.imshow(gradient_h[:, ::-1], extent=[0, glow_extent, -Y_MAX, Y_MAX], aspect='auto', cmap='plasma', alpha=wall_glow*0.8)
        # Left side glow
        ax.imshow(gradient_h, extent=[-glow_extent, 0, -Y_MAX, Y_MAX], aspect='auto', cmap='plasma', alpha=wall_glow*0.8)

    # In a live environment, use plt.show() to display the plot
    # plt.show()

if __name__ == "__main__":
    # Generate stars
    stars = generate_stars(1500)

    # Example usage:
    # Frame 1: Distant View
    plot_approach_frame(stars, distortion_factor=0.01, wall_thickness=0.005, wall_glow=0.1, title="1. Distant View")

    # Frame 3: Close Proximity
    plot_approach_frame(stars, distortion_factor=0.5, wall_thickness=0.1, wall_glow=0.8, title="3. Close Proximity")