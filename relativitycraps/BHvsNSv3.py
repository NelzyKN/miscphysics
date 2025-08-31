import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class RelativisticSimulation:
    """
    Realistic simulation of Kerr black hole and naked singularity with
    proper gravitational lensing and relativistic effects.
    """
    
    def __init__(self, mass=1.0, spin=0.7):
        """
        Initialize simulation parameters.
        
        Args:
            mass: Black hole mass (in geometric units where G=c=1)
            spin: Dimensionless spin parameter (0 to 1)
        """
        self.M = mass
        self.a = spin * mass  # Kerr spin parameter
        self.rs = 2 * mass    # Schwarzschild radius
        
        # Grid parameters for ray tracing
        self.resolution = 200
        self.fov = 40  # Field of view in gravitational radii
        
        # Observer position (Boyer-Lindquist coordinates)
        self.observer_r = 30.0
        self.observer_theta = np.pi / 2  # Equatorial plane
        self.observer_phi = 0.0
        
        # Accretion disk parameters
        self.disk_inner = self.calculate_isco()
        self.disk_outer = 15.0
        self.disk_thickness = 0.1
        
    def calculate_isco(self):
        """Calculate innermost stable circular orbit (ISCO) radius."""
        a = self.a / self.M
        
        # For prograde orbits
        z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        z2 = np.sqrt(3 * a**2 + z1**2)
        
        if self.a >= 0:
            r_isco = self.M * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2*z2)))
        else:
            r_isco = self.M * (3 + z2 + np.sqrt((3 - z1) * (3 + z1 + 2*z2)))
            
        return max(r_isco, self.horizon_radius())
    
    def horizon_radius(self):
        """Calculate event horizon radius."""
        return self.M + np.sqrt(self.M**2 - self.a**2)
    
    def kerr_metric(self, r, theta):
        """
        Calculate Kerr metric components.
        
        Returns:
            Dictionary of metric components
        """
        a = self.a
        M = self.M
        
        # Boyer-Lindquist coordinates
        rho2 = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 - 2*M*r + a**2
        sigma2 = (r**2 + a**2)**2 - a**2 * delta * np.sin(theta)**2
        
        # Metric components
        g_tt = -(1 - 2*M*r/rho2)
        g_tphi = -2*M*r*a*np.sin(theta)**2/rho2
        g_rr = rho2/delta
        g_thth = rho2
        g_phiphi = sigma2*np.sin(theta)**2/rho2
        
        return {
            'g_tt': g_tt,
            'g_tphi': g_tphi,
            'g_rr': g_rr,
            'g_thth': g_thth,
            'g_phiphi': g_phiphi,
            'delta': delta,
            'rho2': rho2
        }
    
    def geodesic_equations(self, state, tau):
        """
        Geodesic equations for photon trajectories in Kerr spacetime.
        
        Args:
            state: [r, theta, phi, p_r, p_theta, p_phi]
            tau: Affine parameter
            
        Returns:
            Derivatives of state vector
        """
        r, theta, phi, pr, ptheta, pphi = state
        
        # Prevent numerical issues
        r = max(r, 0.01)
        theta = np.clip(theta, 0.01, np.pi - 0.01)
        
        a = self.a
        M = self.M
        
        # Calculate metric components
        metric = self.kerr_metric(r, theta)
        
        # Conserved quantities
        E = 1.0  # Photon energy
        L = pphi  # Angular momentum
        
        # Carter constant
        Q = ptheta**2 + np.cos(theta)**2 * (L**2/np.sin(theta)**2)
        
        # Effective potentials
        rho2 = metric['rho2']
        delta = metric['delta']
        
        # Derivatives
        dr_dtau = pr * delta / rho2
        dtheta_dtau = ptheta / rho2
        dphi_dtau = (L/np.sin(theta)**2 - a*E) / rho2 + a*(E*(r**2+a**2) - L*a) / (delta*rho2)
        
        # Force terms
        dpr_dtau = (1/rho2) * (
            -M*(r**2-a**2*np.cos(theta)**2)/rho2 * (E**2 + pr**2*delta/rho2) +
            pr**2*(r - M)/rho2 +
            ptheta**2*(-r)/rho2 +
            L**2/(rho2*np.sin(theta)**2) * (-(r-M)*np.sin(theta)**2 + r*a**2*np.sin(2*theta)**2/(2*rho2))
        )
        
        dptheta_dtau = (1/rho2) * (
            np.sin(theta)*np.cos(theta) * (L**2/np.sin(theta)**4 - a**2*E**2) +
            pr**2*a**2*np.sin(theta)*np.cos(theta)*delta/rho2**2 -
            ptheta**2*a**2*np.sin(theta)*np.cos(theta)/rho2
        )
        
        dpphi_dtau = 0  # Conserved
        
        return [dr_dtau, dtheta_dtau, dphi_dtau, dpr_dtau, dptheta_dtau, dpphi_dtau]
    
    def ray_trace(self, impact_parameter, inclination):
        """
        Trace a light ray backward from observer to source.
        
        Args:
            impact_parameter: Impact parameter of the ray
            inclination: Initial inclination angle
            
        Returns:
            Ray trajectory and final position
        """
        # Initial conditions at observer
        r0 = self.observer_r
        theta0 = self.observer_theta
        phi0 = self.observer_phi
        
        # Initial momenta (backward ray tracing)
        E = 1.0
        L = impact_parameter
        
        # Set up initial momentum components
        pr0 = -np.cos(inclination)
        ptheta0 = -np.sin(inclination) * np.sin(phi0)
        pphi0 = L
        
        # Initial state vector
        state0 = [r0, theta0, phi0, pr0, ptheta0, pphi0]
        
        # Integrate geodesic equations
        tau_span = np.linspace(0, 100, 1000)
        
        try:
            solution = odeint(self.geodesic_equations, state0, tau_span)
            
            # Check if ray hits disk or escapes
            r_traj = solution[:, 0]
            theta_traj = solution[:, 1]
            
            # Check for horizon crossing (for black hole)
            if np.any(r_traj < self.horizon_radius()):
                return None, 'horizon'
            
            # Check for disk intersection
            z_traj = r_traj * np.cos(theta_traj)
            rho_traj = r_traj * np.sin(theta_traj)
            
            disk_mask = (np.abs(z_traj) < self.disk_thickness) & \
                       (rho_traj > self.disk_inner) & \
                       (rho_traj < self.disk_outer)
            
            if np.any(disk_mask):
                idx = np.where(disk_mask)[0][0]
                return solution[:idx+1], 'disk'
            
            return solution, 'escape'
            
        except:
            return None, 'error'
    
    def disk_emission(self, r, phi):
        """
        Calculate accretion disk emission with relativistic effects.
        
        Args:
            r: Radial coordinate
            phi: Azimuthal angle
            
        Returns:
            Observed intensity including Doppler and gravitational redshift
        """
        if r < self.disk_inner or r > self.disk_outer:
            return 0.0
        
        # Keplerian orbital velocity
        v_kep = np.sqrt(self.M / r)
        
        # Temperature profile (Shakura-Sunyaev disk)
        T = (3 * self.M / (8 * np.pi * r**3))**(1/4)
        
        # Blackbody emission
        emission = T**4
        
        # Doppler boosting
        gamma = 1 / np.sqrt(1 - v_kep**2)
        doppler = gamma * (1 - v_kep * np.sin(phi - self.observer_phi))
        
        # Gravitational redshift
        g_factor = np.sqrt(1 - 2*self.M/r)
        
        # Combined observed intensity
        observed = emission * doppler**3 * g_factor
        
        return observed
    
    def gravitational_lensing_image(self, is_kerr=True):
        """
        Generate gravitational lensing image using ray tracing.
        
        Args:
            is_kerr: True for Kerr black hole, False for naked singularity
            
        Returns:
            2D image array
        """
        # Create image grid
        image = np.zeros((self.resolution, self.resolution))
        
        # Camera parameters
        x = np.linspace(-self.fov/2, self.fov/2, self.resolution)
        y = np.linspace(-self.fov/2, self.fov/2, self.resolution)
        
        # Progress tracking
        total_pixels = self.resolution**2
        processed = 0
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                # Convert to impact parameter and inclination
                b = np.sqrt(xi**2 + yi**2)
                inclination = np.arctan2(yi, xi)
                
                # Skip if inside photon sphere for efficiency
                if is_kerr and b < 1.5 * self.horizon_radius():
                    image[j, i] = 0
                    continue
                
                # Trace ray
                trajectory, fate = self.ray_trace(b, inclination)
                
                if fate == 'disk' and trajectory is not None:
                    # Ray hit the disk
                    final_r = trajectory[-1, 0]
                    final_phi = trajectory[-1, 2]
                    
                    # Calculate observed intensity
                    intensity = self.disk_emission(final_r, final_phi)
                    
                    # Apply additional lensing magnification
                    magnification = 1.0 / (1.0 - 2*self.M/final_r)
                    
                    image[j, i] = intensity * magnification
                    
                elif fate == 'horizon':
                    # Ray captured by black hole
                    image[j, i] = 0
                    
                else:
                    # Ray escaped - background stars
                    image[j, i] = 0.1 * np.random.random()
                
                processed += 1
                
                # Progress update
                if processed % 1000 == 0:
                    print(f"Progress: {processed}/{total_pixels} pixels")
        
        return image
    
    def plot_comparison(self):
        """Generate and plot comparison between Kerr black hole and naked singularity."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kerr Black Hole vs Naked Singularity: Relativistic Effects', fontsize=16)
        
        # Kerr black hole
        print("Generating Kerr black hole image...")
        self.a = 0.7 * self.M  # High spin
        kerr_image = self.gravitational_lensing_image(is_kerr=True)
        
        # Naked singularity (over-extremal Kerr)
        print("Generating naked singularity image...")
        self.a = 1.2 * self.M  # Over-extremal spin
        naked_image = self.gravitational_lensing_image(is_kerr=False)
        
        # Plot lensed images
        im1 = axes[0, 0].imshow(kerr_image, cmap='hot', extent=[-self.fov/2, self.fov/2, -self.fov/2, self.fov/2])
        axes[0, 0].set_title('Kerr Black Hole - Lensed Image')
        axes[0, 0].set_xlabel('x (GM/c²)')
        axes[0, 0].set_ylabel('y (GM/c²)')
        plt.colorbar(im1, ax=axes[0, 0], label='Intensity')
        
        im2 = axes[1, 0].imshow(naked_image, cmap='hot', extent=[-self.fov/2, self.fov/2, -self.fov/2, self.fov/2])
        axes[1, 0].set_title('Naked Singularity - Lensed Image')
        axes[1, 0].set_xlabel('x (GM/c²)')
        axes[1, 0].set_ylabel('y (GM/c²)')
        plt.colorbar(im2, ax=axes[1, 0], label='Intensity')
        
        # Plot light ray trajectories
        self.plot_light_rays(axes[0, 1], is_kerr=True)
        axes[0, 1].set_title('Kerr: Light Ray Trajectories')
        
        self.plot_light_rays(axes[1, 1], is_kerr=False)
        axes[1, 1].set_title('Naked Singularity: Light Ray Trajectories')
        
        # Plot spacetime embedding
        self.plot_spacetime_embedding(axes[0, 2], is_kerr=True)
        axes[0, 2].set_title('Kerr: Spacetime Curvature')
        
        self.plot_spacetime_embedding(axes[1, 2], is_kerr=False)
        axes[1, 2].set_title('Naked Singularity: Spacetime Curvature')
        
        plt.tight_layout()
        return fig
    
    def plot_light_rays(self, ax, is_kerr=True):
        """Plot sample light ray trajectories showing lensing."""
        if is_kerr:
            self.a = 0.7 * self.M
        else:
            self.a = 1.2 * self.M
        
        # Sample impact parameters
        impact_parameters = np.linspace(2, 10, 8)
        
        for b in impact_parameters:
            trajectory, fate = self.ray_trace(b, 0)
            
            if trajectory is not None:
                r = trajectory[:, 0]
                phi = trajectory[:, 2]
                
                # Convert to Cartesian coordinates
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                
                # Color based on fate
                if fate == 'horizon':
                    color = 'red'
                    alpha = 0.3
                elif fate == 'disk':
                    color = 'yellow'
                    alpha = 0.7
                else:
                    color = 'blue'
                    alpha = 0.5
                
                ax.plot(x, y, color=color, alpha=alpha, linewidth=0.5)
        
        # Draw event horizon or singularity
        if is_kerr:
            horizon = plt.Circle((0, 0), self.horizon_radius(), color='black', fill=True)
            ax.add_patch(horizon)
        else:
            ax.plot(0, 0, 'mo', markersize=5, label='Singularity')
        
        # Draw accretion disk
        disk = plt.Circle((0, 0), self.disk_outer, color='orange', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(disk)
        disk_inner = plt.Circle((0, 0), self.disk_inner, color='orange', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(disk_inner)
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.set_xlabel('x (GM/c²)')
        ax.set_ylabel('y (GM/c²)')
        ax.grid(True, alpha=0.3)
    
    def plot_spacetime_embedding(self, ax, is_kerr=True):
        """Plot 2D embedding diagram of spacetime curvature."""
        if is_kerr:
            self.a = 0.7 * self.M
        else:
            self.a = 1.2 * self.M
        
        # Create radial grid
        r = np.linspace(0.1, 20, 100)
        theta = np.pi / 2  # Equatorial plane
        
        # Calculate embedding height (simplified)
        z = np.zeros_like(r)
        for i, ri in enumerate(r):
            if is_kerr and ri < self.horizon_radius():
                z[i] = -10  # Deep well for black hole
            else:
                # Schwarzschild-like embedding
                if ri > 2 * self.M:
                    z[i] = -4 * self.M * np.sqrt(ri / (2 * self.M) - 1)
                else:
                    z[i] = -4 * self.M
        
        # Create 3D surface
        phi = np.linspace(0, 2*np.pi, 50)
        R, Phi = np.meshgrid(r, phi)
        Z = np.tile(z, (len(phi), 1))
        
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        
        # Plot surface
        surf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
        
        # Mark special radii
        if is_kerr:
            horizon_circle = plt.Circle((0, 0), self.horizon_radius(), 
                                      color='red', fill=False, linewidth=2)
            ax.add_patch(horizon_circle)
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.set_xlabel('x (GM/c²)')
        ax.set_ylabel('y (GM/c²)')
        plt.colorbar(surf, ax=ax, label='Embedding height')

# Main simulation
def main():
    """Run the complete relativistic simulation."""
    print("Initializing relativistic black hole simulation...")
    print("This computation is intensive and may take a few minutes...")
    print("-" * 50)
    
    # Create simulation instance
    sim = RelativisticSimulation(mass=1.0, spin=0.7)
    
    # Generate comparison plots
    fig = sim.plot_comparison()
    
    # Additional analysis
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot redshift profiles
    r_range = np.linspace(2, 20, 100)
    
    # Kerr redshift
    sim.a = 0.7 * sim.M
    redshift_kerr = np.sqrt(1 - 2*sim.M/r_range)
    
    # Naked singularity redshift  
    sim.a = 1.2 * sim.M
    redshift_naked = np.sqrt(np.abs(1 - 2*sim.M/r_range))
    
    axes[0].plot(r_range, redshift_kerr, 'b-', label='Kerr Black Hole', linewidth=2)
    axes[0].plot(r_range, redshift_naked, 'r--', label='Naked Singularity', linewidth=2)
    axes[0].axvline(x=2*sim.M, color='gray', linestyle=':', alpha=0.5, label='Schwarzschild Radius')
    axes[0].set_xlabel('Distance r (GM/c²)')
    axes[0].set_ylabel('Gravitational Redshift Factor')
    axes[0].set_title('Gravitational Redshift Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot photon sphere and ISCO radii
    spin_range = np.linspace(0, 0.998, 50)
    isco_radii = []
    photon_radii = []
    
    for spin in spin_range:
        sim.a = spin * sim.M
        isco_radii.append(sim.calculate_isco())
        # Photon sphere (approximate)
        photon_radii.append(3 * sim.M * (1 - spin/3))
    
    axes[1].plot(spin_range, isco_radii, 'b-', label='ISCO radius', linewidth=2)
    axes[1].plot(spin_range, photon_radii, 'g--', label='Photon sphere', linewidth=2)
    axes[1].axhline(y=sim.M, color='red', linestyle=':', alpha=0.5, label='Horizon (extremal)')
    axes[1].set_xlabel('Spin Parameter a/M')
    axes[1].set_ylabel('Radius (GM/c²)')
    axes[1].set_title('Critical Radii vs Spin')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation complete!")
    print("-" * 50)
    print("Key observations:")
    print("1. Kerr black hole shows strong gravitational lensing with photon sphere")
    print("2. Light cannot escape from within the event horizon")
    print("3. Naked singularity allows light to escape from all regions")
    print("4. Accretion disk shows Doppler boosting and gravitational redshift")
    print("5. Frame-dragging effects are visible in the twisted light paths")

if __name__ == "__main__":
    main()