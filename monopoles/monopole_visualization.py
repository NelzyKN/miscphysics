#!/usr/bin/env python3
"""
Comprehensive visualization of magnetic monopole signals in Pierre Auger Surface Detectors
Shows detailed FADC trace characteristics and comparisons with cosmic ray signals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy import signal, stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Pierre Auger SD constants
ADC_PER_VEM = 180.0  # ADC counts per Vertical Equivalent Muon
NS_PER_BIN = 25.0    # 25 ns per bin (40 MHz sampling)
BASELINE_ADC = 50.0  # Typical baseline in ADC counts
N_BINS = 2048        # Number of FADC bins

def generate_monopole_trace(mass_gev=1e16, velocity_beta=0.9, charge_n=1):
    """
    Generate realistic magnetic monopole FADC trace
    
    Parameters:
    -----------
    mass_gev : float
        Monopole mass in GeV
    velocity_beta : float
        Velocity as fraction of c
    charge_n : int
        Magnetic charge in units of Dirac charge (g = n * 68.5e)
    """
    trace = np.ones(N_BINS) * BASELINE_ADC
    time_ns = np.arange(N_BINS) * NS_PER_BIN
    
    # Monopole parameters
    g_over_e = charge_n * 68.5  # Magnetic to electric charge ratio
    
    # Energy loss rate in water (Ahlen formula)
    # dE/dx ≈ 4πNα(g/e)² * f(β) where f(β) ≈ 1 for relativistic monopoles
    energy_loss_mev_cm = 4700 * charge_n**2  # MeV/cm in water
    
    # Cherenkov light yield enhanced by (g/e)²
    cherenkov_enhancement = (g_over_e)**2
    base_vem_per_cm = 2.0  # Typical for minimum ionizing particle
    monopole_vem_per_cm = base_vem_per_cm * cherenkov_enhancement / 100  # Scaled
    
    # Tank parameters (cylinder: 10m² area, 1.2m height)
    tank_radius = 1.78  # meters
    water_height = 1.2  # meters
    
    # Monopole trajectory through tank (simplified: vertical for now)
    if velocity_beta > 0.7:  # Fast monopole
        # Entry time (when monopole enters tank)
        entry_time_bins = 400  # ~10 microseconds into trace
        
        # Transit time through water
        path_length_m = water_height / np.cos(np.radians(30))  # 30 degree zenith
        transit_time_ns = (path_length_m / (velocity_beta * 3e8)) * 1e9
        transit_bins = int(transit_time_ns / NS_PER_BIN)
        
        # Signal characteristics
        peak_vem = monopole_vem_per_cm * path_length_m * 100  # Total VEM
        peak_vem = min(peak_vem, 500)  # Limit to avoid saturation
        
        # Rise time (very fast for monopoles)
        rise_bins = 2  # ~50 ns rise
        fall_bins = 3  # ~75 ns fall
        
        # Generate smooth monopole pulse
        for i in range(entry_time_bins, min(entry_time_bins + transit_bins, N_BINS)):
            # Distance from entry
            rel_pos = (i - entry_time_bins) / max(transit_bins, 1)
            
            # Signal shape: nearly rectangular with smooth edges
            if i < entry_time_bins + rise_bins:
                # Rising edge
                rise_frac = (i - entry_time_bins) / rise_bins
                signal_vem = peak_vem * (1 - np.exp(-5 * rise_frac))
            elif i > entry_time_bins + transit_bins - fall_bins:
                # Falling edge
                fall_frac = (entry_time_bins + transit_bins - i) / fall_bins
                signal_vem = peak_vem * (1 - np.exp(-5 * fall_frac))
            else:
                # Plateau with small fluctuations
                signal_vem = peak_vem * (1 + 0.02 * np.random.randn())
            
            trace[i] = BASELINE_ADC + signal_vem * ADC_PER_VEM
        
        # Add small afterglow (Cherenkov photons bouncing in tank)
        afterglow_start = entry_time_bins + transit_bins
        for i in range(afterglow_start, min(afterglow_start + 50, N_BINS)):
            decay_const = 0.1
            afterglow_vem = peak_vem * 0.05 * np.exp(-decay_const * (i - afterglow_start))
            trace[i] += afterglow_vem * ADC_PER_VEM
            
    else:  # Slow monopole (β < 0.7)
        # Longer transit time, lower but sustained signal
        entry_time_bins = 400
        transit_time_ns = (water_height / (velocity_beta * 3e8)) * 1e9
        transit_bins = int(transit_time_ns / NS_PER_BIN)
        
        # Lower peak due to sub-Cherenkov threshold
        peak_vem = 50 * charge_n**2 * velocity_beta**2
        
        for i in range(entry_time_bins, min(entry_time_bins + transit_bins, N_BINS)):
            signal_vem = peak_vem * (1 + 0.05 * np.random.randn())
            trace[i] = BASELINE_ADC + signal_vem * ADC_PER_VEM
    
    # Add realistic noise
    noise = np.random.normal(0, 2, N_BINS)  # 2 ADC counts RMS
    trace += noise
    
    # Ensure no negative values
    trace = np.maximum(trace, 0)
    
    return trace, peak_vem, transit_bins * NS_PER_BIN

def generate_hadronic_shower_trace(energy_eev=0.1, core_distance_m=500):
    """Generate typical hadronic cosmic ray shower trace"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    
    # Main electromagnetic component
    em_start = 600 + np.random.randint(-50, 50)
    em_peak_vem = 30 * (energy_eev / 0.1) * np.exp(-core_distance_m / 1000)
    
    # EM pulse with rise and decay
    for i in range(em_start, min(em_start + 300, N_BINS)):
        t_rel = (i - em_start) / 100
        if t_rel < 1:
            # Rising edge
            signal_vem = em_peak_vem * t_rel * np.exp(-2 * t_rel)
        else:
            # Decay
            signal_vem = em_peak_vem * np.exp(-2 * (t_rel - 1))
        
        trace[i] = BASELINE_ADC + signal_vem * ADC_PER_VEM
    
    # Add muon spikes (characteristic of hadronic showers)
    n_muons = np.random.poisson(3 + int(10 * energy_eev))
    
    for _ in range(n_muons):
        muon_time = em_start + np.random.randint(-200, 400)
        if 0 <= muon_time < N_BINS - 10:
            muon_vem = np.random.exponential(5) * (1 + energy_eev)
            muon_width = np.random.randint(2, 6)
            
            for j in range(muon_width):
                if muon_time + j < N_BINS:
                    trace[muon_time + j] += muon_vem * ADC_PER_VEM * np.exp(-j/2)
    
    # Add noise
    noise = np.random.normal(0, 2, N_BINS)
    trace += noise
    trace = np.maximum(trace, 0)
    
    return trace

def generate_photon_shower_trace(energy_eev=0.1, core_distance_m=500):
    """Generate photon-induced shower trace (muon-poor)"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    
    # Photon showers are muon-poor, mostly EM component
    em_start = 650 + np.random.randint(-30, 30)
    em_peak_vem = 25 * (energy_eev / 0.1) * np.exp(-core_distance_m / 800)
    
    # Smoother, broader EM pulse
    for i in range(em_start, min(em_start + 400, N_BINS)):
        t_rel = (i - em_start) / 150
        if t_rel < 1:
            signal_vem = em_peak_vem * (1 - np.exp(-3 * t_rel)) * np.exp(-t_rel)
        else:
            signal_vem = em_peak_vem * 0.7 * np.exp(-2 * (t_rel - 1))
        
        trace[i] = BASELINE_ADC + signal_vem * ADC_PER_VEM
    
    # Very few muons (90% less than hadronic)
    n_muons = np.random.poisson(0.3 * (1 + energy_eev))
    
    for _ in range(n_muons):
        muon_time = em_start + np.random.randint(-100, 300)
        if 0 <= muon_time < N_BINS - 5:
            muon_vem = np.random.exponential(3)
            for j in range(3):
                if muon_time + j < N_BINS:
                    trace[muon_time + j] += muon_vem * ADC_PER_VEM * np.exp(-j)
    
    # Add noise
    noise = np.random.normal(0, 2, N_BINS)
    trace += noise
    trace = np.maximum(trace, 0)
    
    return trace

def calculate_trace_features(trace):
    """Calculate discriminating features from FADC trace"""
    signal_vem = (trace - BASELINE_ADC) / ADC_PER_VEM
    signal_vem = np.maximum(signal_vem, 0)
    
    features = {}
    
    # Basic features
    features['peak_vem'] = np.max(signal_vem)
    features['total_charge_vem'] = np.sum(signal_vem) * NS_PER_BIN / 1000  # VEM·μs
    
    # Time features
    above_10vem = np.where(signal_vem > 10)[0]
    if len(above_10vem) > 0:
        features['signal_start_us'] = above_10vem[0] * NS_PER_BIN / 1000
        features['signal_duration_us'] = (above_10vem[-1] - above_10vem[0]) * NS_PER_BIN / 1000
    else:
        features['signal_start_us'] = 0
        features['signal_duration_us'] = 0
    
    # Sustained signal metrics
    features['bins_above_50vem'] = np.sum(signal_vem > 50)
    features['bins_above_100vem'] = np.sum(signal_vem > 100)
    features['sustained_fraction'] = features['bins_above_50vem'] / N_BINS
    
    # Smoothness (RMS of derivative in high signal region)
    high_signal = signal_vem > 20
    if np.sum(high_signal) > 10:
        derivative = np.diff(signal_vem)
        high_signal_deriv = derivative[high_signal[:-1]]
        features['smoothness_rms'] = np.std(high_signal_deriv)
    else:
        features['smoothness_rms'] = 999
    
    # Pulse shape
    if features['peak_vem'] > 20:
        peak_idx = np.argmax(signal_vem)
        # Find FWHM
        half_max = features['peak_vem'] / 2
        above_half = signal_vem > half_max
        if np.sum(above_half) > 0:
            first_half = np.where(above_half)[0][0]
            last_half = np.where(above_half)[0][-1]
            features['fwhm_us'] = (last_half - first_half) * NS_PER_BIN / 1000
            
            # Calculate plateau level (average in middle 50% of pulse)
            if last_half > first_half + 10:
                mid_start = first_half + int((last_half - first_half) * 0.25)
                mid_end = first_half + int((last_half - first_half) * 0.75)
                features['plateau_vem'] = np.mean(signal_vem[mid_start:mid_end])
                features['peak_to_plateau'] = features['peak_vem'] / features['plateau_vem']
            else:
                features['plateau_vem'] = 0
                features['peak_to_plateau'] = 999
        else:
            features['fwhm_us'] = 0
            features['plateau_vem'] = 0
            features['peak_to_plateau'] = 999
    else:
        features['fwhm_us'] = 0
        features['plateau_vem'] = 0
        features['peak_to_plateau'] = 999
    
    # Muon content indicator (spikiness)
    if len(signal_vem) > 100:
        peaks, properties = signal.find_peaks(signal_vem, height=10, distance=20)
        features['n_peaks'] = len(peaks)
        if len(peaks) > 0:
            features['peak_asymmetry'] = stats.skew(properties['peak_heights'])
        else:
            features['peak_asymmetry'] = 0
    else:
        features['n_peaks'] = 0
        features['peak_asymmetry'] = 0
    
    return features

def create_comprehensive_visualization():
    """Create detailed figure showing monopole trace characteristics"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Generate traces
    monopole_trace, monopole_peak, monopole_duration = generate_monopole_trace(
        mass_gev=1e16, velocity_beta=0.9, charge_n=1)
    hadronic_trace = generate_hadronic_shower_trace(energy_eev=0.1, core_distance_m=500)
    photon_trace = generate_photon_shower_trace(energy_eev=0.1, core_distance_m=500)
    
    time_bins = np.arange(N_BINS) * NS_PER_BIN / 1000  # Convert to microseconds
    
    # ========== Row 1: Raw FADC Traces ==========
    
    # Monopole trace
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_bins, monopole_trace, 'b-', linewidth=0.8, alpha=0.8)
    ax1.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.axhline(y=BASELINE_ADC + 100*ADC_PER_VEM, color='red', linestyle='--', 
                alpha=0.5, label='100 VEM level')
    ax1.fill_between(time_bins, BASELINE_ADC, monopole_trace, 
                     where=(monopole_trace > BASELINE_ADC + 50*ADC_PER_VEM),
                     alpha=0.3, color='blue', label='Signal >50 VEM')
    ax1.set_title('Magnetic Monopole Signal', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time [μs]')
    ax1.set_ylabel('ADC Counts')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, max(monopole_trace) * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Add annotation
    peak_time = np.argmax(monopole_trace) * NS_PER_BIN / 1000
    ax1.annotate(f'Sustained signal\n~{monopole_duration:.0f} ns\nPeak: {monopole_peak:.0f} VEM',
                xy=(peak_time, max(monopole_trace)), 
                xytext=(peak_time + 5, max(monopole_trace) * 0.9),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=9, color='blue', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='blue'))
    
    # Hadronic shower trace
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_bins, hadronic_trace, 'g-', linewidth=0.8, alpha=0.8)
    ax2.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.5)
    
    # Mark muon spikes
    signal_h = (hadronic_trace - BASELINE_ADC) / ADC_PER_VEM
    peaks_h, _ = signal.find_peaks(signal_h, height=5, distance=20)
    for peak in peaks_h[:5]:  # Mark first 5 peaks
        ax2.plot(peak * NS_PER_BIN / 1000, hadronic_trace[peak], 'ro', markersize=6)
        if peak < 3:
            ax2.annotate('μ', xy=(peak * NS_PER_BIN / 1000, hadronic_trace[peak]),
                        xytext=(peak * NS_PER_BIN / 1000, hadronic_trace[peak] + 100),
                        fontsize=8, color='red', fontweight='bold')
    
    ax2.set_title('Hadronic Shower Signal', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time [μs]')
    ax2.set_ylabel('ADC Counts')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, max(max(hadronic_trace), max(monopole_trace)) * 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Photon shower trace
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_bins, photon_trace, 'orange', linewidth=0.8, alpha=0.8)
    ax3.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Photon Shower Signal (muon-poor)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time [μs]')
    ax3.set_ylabel('ADC Counts')
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, max(max(photon_trace), max(monopole_trace)) * 1.1)
    ax3.grid(True, alpha=0.3)
    
    # ========== Row 2: Signal in VEM units ==========
    
    # Convert to VEM
    monopole_vem = np.maximum((monopole_trace - BASELINE_ADC) / ADC_PER_VEM, 0)
    hadronic_vem = np.maximum((hadronic_trace - BASELINE_ADC) / ADC_PER_VEM, 0)
    photon_vem = np.maximum((photon_trace - BASELINE_ADC) / ADC_PER_VEM, 0)
    
    # Monopole VEM
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.fill_between(time_bins, 0, monopole_vem, alpha=0.6, color='blue')
    ax4.plot(time_bins, monopole_vem, 'b-', linewidth=1)
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50 VEM threshold')
    ax4.axhline(y=100, color='darkred', linestyle='--', alpha=0.5, label='100 VEM threshold')
    ax4.set_title('Monopole Signal [VEM]', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time [μs]')
    ax4.set_ylabel('Signal [VEM]')
    ax4.set_xlim(0, 50)
    ax4.set_ylim(0, max(monopole_vem) * 1.1)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=8)
    
    # Hadronic VEM
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(time_bins, 0, hadronic_vem, alpha=0.6, color='green')
    ax5.plot(time_bins, hadronic_vem, 'g-', linewidth=1)
    ax5.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax5.set_title('Hadronic Signal [VEM]', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time [μs]')
    ax5.set_ylabel('Signal [VEM]')
    ax5.set_xlim(0, 50)
    ax5.set_ylim(0, max(max(monopole_vem), max(hadronic_vem)) * 1.1)
    ax5.grid(True, alpha=0.3)
    
    # Photon VEM
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.fill_between(time_bins, 0, photon_vem, alpha=0.6, color='orange')
    ax6.plot(time_bins, photon_vem, color='darkorange', linewidth=1)
    ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax6.set_title('Photon Signal [VEM]', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time [μs]')
    ax6.set_ylabel('Signal [VEM]')
    ax6.set_xlim(0, 50)
    ax6.set_ylim(0, max(max(monopole_vem), max(hadronic_vem)) * 1.1)
    ax6.grid(True, alpha=0.3)
    
    # ========== Row 3: Feature Analysis ==========
    
    # Calculate features
    monopole_features = calculate_trace_features(monopole_trace)
    hadronic_features = calculate_trace_features(hadronic_trace)
    photon_features = calculate_trace_features(photon_trace)
    
    # Feature comparison bar chart
    ax7 = fig.add_subplot(gs[2, :2])
    
    features_to_plot = ['peak_vem', 'total_charge_vem', 'signal_duration_us', 
                       'smoothness_rms', 'peak_to_plateau', 'n_peaks']
    feature_labels = ['Peak\n[VEM]', 'Total Charge\n[VEM·μs]', 'Duration\n[μs]',
                     'Smoothness\n(RMS)', 'Peak/Plateau\nRatio', 'N Peaks']
    
    x = np.arange(len(features_to_plot))
    width = 0.25
    
    monopole_values = [monopole_features[f] for f in features_to_plot]
    hadronic_values = [hadronic_features[f] for f in features_to_plot]
    photon_values = [photon_features[f] for f in features_to_plot]
    
    bars1 = ax7.bar(x - width, monopole_values, width, label='Monopole', 
                   color='blue', alpha=0.7)
    bars2 = ax7.bar(x, hadronic_values, width, label='Hadronic', 
                   color='green', alpha=0.7)
    bars3 = ax7.bar(x + width, photon_values, width, label='Photon', 
                   color='orange', alpha=0.7)
    
    ax7.set_xlabel('Feature', fontsize=11)
    ax7.set_ylabel('Value (log scale)', fontsize=11)
    ax7.set_title('Feature Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(feature_labels, fontsize=9)
    ax7.set_yscale('log')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Discrimination plot
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Calculate monopole score
    def monopole_score(features):
        score = 0
        if features['signal_duration_us'] > 1: score += 0.25
        if features['bins_above_100vem'] > 20: score += 0.25
        if features['smoothness_rms'] < 5: score += 0.2
        if features['peak_to_plateau'] < 1.5: score += 0.2
        if features['n_peaks'] < 3: score += 0.1
        return score
    
    scores = [monopole_score(monopole_features),
             monopole_score(hadronic_features),
             monopole_score(photon_features)]
    
    colors = ['blue', 'green', 'orange']
    labels = ['Monopole', 'Hadronic', 'Photon']
    
    bars = ax8.bar(labels, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax8.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Detection Threshold')
    ax8.set_ylabel('Monopole Score', fontsize=11)
    ax8.set_title('Monopole Detection Score', fontsize=12, fontweight='bold')
    ax8.set_ylim(0, 1)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.legend(fontsize=10)
    
    # Add score values on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        # Add detection status
        status = 'DETECTED' if score > 0.6 else 'REJECTED'
        color = 'darkgreen' if score > 0.6 else 'darkred'
        ax8.text(bar.get_x() + bar.get_width()/2., 0.05,
                status, ha='center', fontsize=9, color=color, fontweight='bold')
    
    # ========== Row 4: Time Profile Analysis ==========
    
    # Integrated charge over time
    ax9 = fig.add_subplot(gs[3, 0])
    
    monopole_cumsum = np.cumsum(monopole_vem) * NS_PER_BIN / 1000
    hadronic_cumsum = np.cumsum(hadronic_vem) * NS_PER_BIN / 1000
    photon_cumsum = np.cumsum(photon_vem) * NS_PER_BIN / 1000
    
    ax9.plot(time_bins, monopole_cumsum, 'b-', linewidth=2, label='Monopole')
    ax9.plot(time_bins, hadronic_cumsum, 'g-', linewidth=2, label='Hadronic')
    ax9.plot(time_bins, photon_cumsum, 'orange', linewidth=2, label='Photon')
    ax9.set_xlabel('Time [μs]')
    ax9.set_ylabel('Integrated Charge [VEM·μs]')
    ax9.set_title('Cumulative Charge', fontsize=11, fontweight='bold')
    ax9.set_xlim(0, 50)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=9)
    
    # Derivative analysis (smoothness)
    ax10 = fig.add_subplot(gs[3, 1])
    
    # Calculate derivatives
    monopole_deriv = np.diff(monopole_vem)
    hadronic_deriv = np.diff(hadronic_vem)
    photon_deriv = np.diff(photon_vem)
    
    time_deriv = time_bins[:-1]
    
    ax10.plot(time_deriv, monopole_deriv, 'b-', alpha=0.6, linewidth=0.8, label='Monopole')
    ax10.plot(time_deriv, hadronic_deriv, 'g-', alpha=0.6, linewidth=0.8, label='Hadronic')
    ax10.plot(time_deriv, photon_deriv, 'orange', alpha=0.6, linewidth=0.8, label='Photon')
    ax10.set_xlabel('Time [μs]')
    ax10.set_ylabel('dSignal/dt [VEM/bin]')
    ax10.set_title('Signal Derivative (Smoothness)', fontsize=11, fontweight='bold')
    ax10.set_xlim(0, 50)
    ax10.set_ylim(-20, 20)
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=9)
    
    # Summary table
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('tight')
    ax11.axis('off')
    
    summary_data = [
        ['Property', 'Monopole', 'Hadronic', 'Photon'],
        ['Peak [VEM]', f'{monopole_features["peak_vem"]:.1f}', 
         f'{hadronic_features["peak_vem"]:.1f}', f'{photon_features["peak_vem"]:.1f}'],
        ['Duration [μs]', f'{monopole_features["signal_duration_us"]:.2f}', 
         f'{hadronic_features["signal_duration_us"]:.2f}', 
         f'{photon_features["signal_duration_us"]:.2f}'],
        ['Total Charge', f'{monopole_features["total_charge_vem"]:.1f}', 
         f'{hadronic_features["total_charge_vem"]:.1f}', 
         f'{photon_features["total_charge_vem"]:.1f}'],
        ['Smoothness', f'{monopole_features["smoothness_rms"]:.2f}', 
         f'{hadronic_features["smoothness_rms"]:.2f}', 
         f'{photon_features["smoothness_rms"]:.2f}'],
        ['Bins >100VEM', f'{monopole_features["bins_above_100vem"]}', 
         f'{hadronic_features["bins_above_100vem"]}', 
         f'{photon_features["bins_above_100vem"]}'],
        ['Detection', 'YES' if scores[0] > 0.6 else 'NO',
         'YES' if scores[1] > 0.6 else 'NO',
         'YES' if scores[2] > 0.6 else 'NO']
    ]
    
    table = ax11.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color detection row
    for i in range(1, 4):
        if summary_data[6][i] == 'YES':
            table[(6, i)].set_facecolor('#90EE90')
        else:
            table[(6, i)].set_facecolor('#FFB6C1')
    
    ax11.set_title('Summary Table', fontsize=11, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('Magnetic Monopole Signal Characteristics in Pierre Auger Surface Detectors',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def create_monopole_signature_histogram():
    """Create histogram comparing key monopole signatures"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Generate multiple traces for statistics
    n_samples = 1000
    
    monopole_peaks = []
    monopole_durations = []
    monopole_charges = []
    monopole_smoothness = []
    
    hadronic_peaks = []
    hadronic_durations = []
    hadronic_charges = []
    hadronic_smoothness = []
    
    photon_peaks = []
    photon_durations = []
    photon_charges = []
    photon_smoothness = []
    
    print("Generating sample traces for statistical analysis...")
    
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Processing {i}/{n_samples}...")
        
        # Monopole
        m_trace, _, _ = generate_monopole_trace(velocity_beta=0.8 + 0.15*np.random.rand())
        m_feat = calculate_trace_features(m_trace)
        monopole_peaks.append(m_feat['peak_vem'])
        monopole_durations.append(m_feat['signal_duration_us'])
        monopole_charges.append(m_feat['total_charge_vem'])
        monopole_smoothness.append(min(m_feat['smoothness_rms'], 20))
        
        # Hadronic
        h_trace = generate_hadronic_shower_trace(
            energy_eev=0.05 + 0.15*np.random.rand(),
            core_distance_m=300 + 400*np.random.rand())
        h_feat = calculate_trace_features(h_trace)
        hadronic_peaks.append(h_feat['peak_vem'])
        hadronic_durations.append(h_feat['signal_duration_us'])
        hadronic_charges.append(h_feat['total_charge_vem'])
        hadronic_smoothness.append(min(h_feat['smoothness_rms'], 20))
        
        # Photon
        p_trace = generate_photon_shower_trace(
            energy_eev=0.05 + 0.15*np.random.rand(),
            core_distance_m=300 + 400*np.random.rand())
        p_feat = calculate_trace_features(p_trace)
        photon_peaks.append(p_feat['peak_vem'])
        photon_durations.append(p_feat['signal_duration_us'])
        photon_charges.append(p_feat['total_charge_vem'])
        photon_smoothness.append(min(p_feat['smoothness_rms'], 20))
    
    # Plot histograms
    
    # Peak VEM
    ax = axes[0, 0]
    bins = np.linspace(0, 300, 50)
    ax.hist(monopole_peaks, bins=bins, alpha=0.5, label='Monopole', color='blue', density=True)
    ax.hist(hadronic_peaks, bins=bins, alpha=0.5, label='Hadronic', color='green', density=True)
    ax.hist(photon_peaks, bins=bins, alpha=0.5, label='Photon', color='orange', density=True)
    ax.axvline(x=100, color='red', linestyle='--', label='100 VEM threshold')
    ax.set_xlabel('Peak Signal [VEM]')
    ax.set_ylabel('Probability Density')
    ax.set_title('Peak Amplitude Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Signal Duration
    ax = axes[0, 1]
    bins = np.linspace(0, 5, 50)
    ax.hist(monopole_durations, bins=bins, alpha=0.5, label='Monopole', color='blue', density=True)
    ax.hist(hadronic_durations, bins=bins, alpha=0.5, label='Hadronic', color='green', density=True)
    ax.hist(photon_durations, bins=bins, alpha=0.5, label='Photon', color='orange', density=True)
    ax.axvline(x=1.0, color='red', linestyle='--', label='1 μs threshold')
    ax.set_xlabel('Signal Duration [μs]')
    ax.set_ylabel('Probability Density')
    ax.set_title('Signal Duration Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total Charge
    ax = axes[0, 2]
    bins = np.linspace(0, 500, 50)
    ax.hist(monopole_charges, bins=bins, alpha=0.5, label='Monopole', color='blue', density=True)
    ax.hist(hadronic_charges, bins=bins, alpha=0.5, label='Hadronic', color='green', density=True)
    ax.hist(photon_charges, bins=bins, alpha=0.5, label='Photon', color='orange', density=True)
    ax.set_xlabel('Total Charge [VEM·μs]')
    ax.set_ylabel('Probability Density')
    ax.set_title('Integrated Charge Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Smoothness
    ax = axes[1, 0]
    bins = np.linspace(0, 20, 50)
    ax.hist(monopole_smoothness, bins=bins, alpha=0.5, label='Monopole', color='blue', density=True)
    ax.hist(hadronic_smoothness, bins=bins, alpha=0.5, label='Hadronic', color='green', density=True)
    ax.hist(photon_smoothness, bins=bins, alpha=0.5, label='Photon', color='orange', density=True)
    ax.axvline(x=5, color='red', linestyle='--', label='Smoothness threshold')
    ax.set_xlabel('Signal Smoothness (RMS of derivative)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Signal Smoothness Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2D Distribution: Peak vs Duration
    ax = axes[1, 1]
    ax.scatter(monopole_durations, monopole_peaks, alpha=0.3, label='Monopole', 
              color='blue', s=20)
    ax.scatter(hadronic_durations, hadronic_peaks, alpha=0.3, label='Hadronic', 
              color='green', s=20)
    ax.scatter(photon_durations, photon_peaks, alpha=0.3, label='Photon', 
              color='orange', s=20)
    
    # Add discrimination boundary
    x_boundary = np.array([0.8, 5])
    y_boundary = np.array([80, 80])
    ax.plot(x_boundary, y_boundary, 'r--', linewidth=2, label='Discrimination boundary')
    ax.fill_between(x_boundary, y_boundary, 300, alpha=0.2, color='red')
    ax.text(2.5, 150, 'Monopole Region', fontsize=11, color='red', fontweight='bold')
    
    ax.set_xlabel('Signal Duration [μs]')
    ax.set_ylabel('Peak Signal [VEM]')
    ax.set_title('Peak vs Duration (2D discrimination)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 300)
    
    # ROC Curve
    ax = axes[1, 2]
    
    # Calculate simple discrimination scores
    def discrimination_score(peaks, durations, charges, smoothness):
        scores = []
        for p, d, c, s in zip(peaks, durations, charges, smoothness):
            score = 0
            if p > 100: score += 0.3
            if d > 1.0: score += 0.3
            if c > 100: score += 0.2
            if s < 5: score += 0.2
            scores.append(score)
        return np.array(scores)
    
    monopole_scores = discrimination_score(monopole_peaks, monopole_durations, 
                                          monopole_charges, monopole_smoothness)
    hadronic_scores = discrimination_score(hadronic_peaks, hadronic_durations,
                                          hadronic_charges, hadronic_smoothness)
    
    # Calculate ROC curve
    thresholds = np.linspace(0, 1, 100)
    tpr = []  # True Positive Rate (monopole detection efficiency)
    fpr = []  # False Positive Rate (hadronic contamination)
    
    for thresh in thresholds:
        tp = np.sum(monopole_scores > thresh) / len(monopole_scores)
        fp = np.sum(hadronic_scores > thresh) / len(hadronic_scores)
        tpr.append(tp)
        fpr.append(fp)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='Monopole vs Hadronic')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
    
    # Mark operating point
    op_thresh = 0.6
    op_tpr = np.sum(monopole_scores > op_thresh) / len(monopole_scores)
    op_fpr = np.sum(hadronic_scores > op_thresh) / len(hadronic_scores)
    ax.plot(op_fpr, op_tpr, 'ro', markersize=10, label=f'Operating point (thresh={op_thresh})')
    
    ax.set_xlabel('False Positive Rate (Hadronic contamination)')
    ax.set_ylabel('True Positive Rate (Monopole efficiency)')
    ax.set_title('ROC Curve for Monopole Detection', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    ax.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    
    fig.suptitle('Statistical Analysis of Monopole Signatures (1000 simulated traces)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("MAGNETIC MONOPOLE SIGNAL ANALYSIS FOR PIERRE AUGER OBSERVATORY")
    print("=" * 70)
    print()
    
    # Create comprehensive visualization
    print("Creating comprehensive trace visualization...")
    fig1 = create_comprehensive_visualization()
    
    # Create statistical histograms
    print("\nGenerating statistical analysis...")
    fig2 = create_monopole_signature_histogram()
    
    # Save figures
    fig1.savefig('monopole_trace_comprehensive.png', dpi=150, bbox_inches='tight')
    fig2.savefig('monopole_statistical_analysis.png', dpi=150, bbox_inches='tight')
    
    print("\nFigures saved:")
    print("  - monopole_trace_comprehensive.png")
    print("  - monopole_statistical_analysis.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("KEY MONOPOLE SIGNATURES IN PIERRE AUGER SD:")
    print("=" * 70)
    print()
    print("1. SUSTAINED HIGH AMPLITUDE")
    print("   - Signal >100 VEM for >1 microsecond")
    print("   - Nearly rectangular pulse shape")
    print("   - Peak-to-plateau ratio <1.5")
    print()
    print("2. EXTREME SMOOTHNESS")
    print("   - RMS of derivative <5 VEM (vs >10 for hadronic)")
    print("   - No muon spikes")
    print("   - Continuous signal, not fragmented")
    print()
    print("3. ENORMOUS TOTAL CHARGE")
    print("   - Total charge >100-500 VEM·μs")
    print("   - Due to (g/e)² ≈ 4700 enhancement")
    print("   - Proportional to path length through tank")
    print()
    print("4. CHARACTERISTIC TIMING")
    print("   - Very fast rise time (<50 ns)")
    print("   - Duration = tank_thickness/(β·c)")
    print("   - For β=0.9: ~4.4 ns/cm → ~500 ns for 1.2m")
    print()
    print("5. MULTI-STATION PATTERN")
    print("   - Straight track through array")
    print("   - Similar signals in adjacent stations")
    print("   - No lateral attenuation like air showers")
    print()
    print("=" * 70)
    
    plt.show()
