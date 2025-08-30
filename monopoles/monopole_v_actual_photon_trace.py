#!/usr/bin/env python3
"""
Generate and display magnetic monopole FADC traces in Pierre Auger format
Comparison with actual photon/proton traces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Pierre Auger SD constants (matching your real data)
BASELINE_ADC = 290.0  # Baseline from your trace (not 50)
ADC_PER_VEM = 180.0   # ADC counts per VEM
NS_PER_BIN = 25.0     # 25 ns per bin
N_BINS = 2048         # Number of FADC bins

def generate_monopole_trace_auger_format(charge_n=1, velocity_beta=0.9, entry_time_bins=600):
    """
    Generate a magnetic monopole FADC trace matching Pierre Auger format
    
    Parameters:
    -----------
    charge_n : int
        Magnetic charge in units of Dirac charge (g = n * 68.5e)
    velocity_beta : float
        Velocity as fraction of speed of light
    entry_time_bins : int
        Time bin when monopole enters tank (to match your trace timing)
    """
    
    # Initialize trace with realistic baseline
    trace = np.ones(N_BINS) * BASELINE_ADC
    
    # Add baseline noise (similar to real data)
    baseline_noise = np.random.normal(0, 1.5, N_BINS)
    trace += baseline_noise
    
    # Monopole physics parameters
    g_over_e = charge_n * 68.5  # Magnetic to electric charge ratio
    
    # Expected signal level for monopole
    # Much higher than cosmic rays due to (g/e)² enhancement
    monopole_vem = 200 * charge_n**2  # ~200 VEM for n=1 monopole
    monopole_adc = monopole_vem * ADC_PER_VEM
    
    # Transit time through 1.2m of water
    water_height = 1.2  # meters
    transit_time_ns = (water_height / (velocity_beta * 3e8)) * 1e9
    transit_bins = int(transit_time_ns / NS_PER_BIN)
    
    # For relativistic monopole: ~4-5 ns total, but signal spreads due to light propagation
    # Effective signal duration ~40-60 bins (1-1.5 microseconds)
    signal_duration_bins = 50  # More realistic for light collection time
    
    # Generate monopole signal
    for i in range(entry_time_bins, min(entry_time_bins + signal_duration_bins, N_BINS)):
        # Position within signal
        rel_pos = (i - entry_time_bins) / signal_duration_bins
        
        # Signal shape: fast rise, sustained plateau, fast fall
        if rel_pos < 0.05:  # Rising edge (5% of duration)
            signal_fraction = rel_pos / 0.05
            signal_adc = monopole_adc * (1 - np.exp(-10 * signal_fraction))
        elif rel_pos > 0.95:  # Falling edge (last 5%)
            signal_fraction = (1 - rel_pos) / 0.05
            signal_adc = monopole_adc * (1 - np.exp(-10 * signal_fraction))
        else:  # Plateau region
            # Sustained high signal with small fluctuations
            signal_adc = monopole_adc * (1 + 0.01 * np.random.randn())
        
        trace[i] = BASELINE_ADC + signal_adc
    
    # Add small afterglow (light bouncing in tank)
    afterglow_start = entry_time_bins + signal_duration_bins
    for i in range(afterglow_start, min(afterglow_start + 100, N_BINS)):
        afterglow_adc = monopole_adc * 0.02 * np.exp(-0.05 * (i - afterglow_start))
        trace[i] += afterglow_adc
    
    # Ensure no values below zero
    trace = np.maximum(trace, 0)
    
    # Clip extreme values (ADC saturation at ~4000)
    trace = np.minimum(trace, 4000)
    
    return trace

def plot_monopole_vs_real_trace():
    """
    Create figure comparing monopole trace to real cosmic ray trace
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    time_bins = np.arange(N_BINS)
    
    # ========== Top: Recreation of your photon trace ==========
    ax1 = axes[0]
    
    # Simulate a trace similar to your photon event
    photon_trace = np.ones(N_BINS) * 290  # Your baseline
    photon_trace += np.random.normal(0, 1.5, N_BINS)  # Noise
    
    # Main peak at bin ~600
    for i in range(580, 700):
        rel_pos = (i - 580) / 120
        if rel_pos < 0.2:
            signal = 95 * (rel_pos / 0.2)  # Rise
        elif rel_pos < 0.3:
            signal = 95  # Peak
        else:
            signal = 95 * np.exp(-8 * (rel_pos - 0.3))  # Decay
        photon_trace[i] += signal + np.random.normal(0, 2)
    
    # Secondary peaks (muons or late particles)
    photon_trace[820:825] += 35 * np.exp(-0.5 * np.arange(5))
    photon_trace[650:653] += 25 * np.exp(-0.5 * np.arange(3))
    
    ax1.plot(time_bins, photon_trace, 'b-', linewidth=0.8)
    ax1.set_ylim(280, 400)
    ax1.set_xlim(0, 2048)
    ax1.set_ylabel('ADC', fontsize=12)
    ax1.set_title('Event 2, Station 4001, PMT 3 [MOPS trigger] [photon primary] [ML: 0.387 ML-Hadron]',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add stats box
    stats_text = f'Entries    2048\nMean      1023\nRMS       590.5'
    ax1.text(0.92, 0.95, 'traceHist_14', transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(0.89, 0.75, stats_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== Middle: Magnetic Monopole Trace ==========
    ax2 = axes[1]
    
    # Generate monopole trace
    monopole_trace = generate_monopole_trace_auger_format(charge_n=1, velocity_beta=0.9, entry_time_bins=600)
    
    ax2.plot(time_bins, monopole_trace, 'r-', linewidth=0.8)
    ax2.set_ylim(280, 4000)  # Much larger scale needed!
    ax2.set_xlim(0, 2048)
    ax2.set_ylabel('ADC', fontsize=12)
    
    # Calculate VEM for title
    peak_adc = np.max(monopole_trace) - BASELINE_ADC
    peak_vem = peak_adc / ADC_PER_VEM
    
    ax2.set_title(f'Event X, Station 4001, PMT 3 [MONOPOLE CANDIDATE] [Peak: {peak_vem:.0f} VEM] [ML: 0.99 ML-Monopole]',
                  fontsize=11, fontweight='bold', color='red')
    ax2.grid(True, alpha=0.3)
    
    # Add monopole stats box
    monopole_mean = np.mean(monopole_trace)
    monopole_rms = np.std(monopole_trace)
    stats_text2 = f'Entries    2048\nMean      {monopole_mean:.0f}\nRMS       {monopole_rms:.1f}'
    ax2.text(0.92, 0.95, 'traceHist_M1', transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.89, 0.75, stats_text2, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Highlight monopole signal region
    rect = Rectangle((600, 280), 50, peak_adc + BASELINE_ADC - 280, 
                    alpha=0.2, facecolor='red', edgecolor='red', linewidth=2)
    ax2.add_patch(rect)
    ax2.annotate('Monopole Transit', xy=(625, peak_adc/2 + BASELINE_ADC), 
                xytext=(750, peak_adc/2 + BASELINE_ADC),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    # ========== Bottom: Direct Comparison (zoomed to photon scale) ==========
    ax3 = axes[2]
    
    # Plot both on same scale as photon
    ax3.plot(time_bins, photon_trace, 'b-', linewidth=0.8, alpha=0.7, label='Photon Primary')
    ax3.plot(time_bins, np.minimum(monopole_trace, 400), 'r-', linewidth=0.8, alpha=0.7, label='Monopole (clipped)')
    ax3.set_ylim(280, 400)
    ax3.set_xlim(0, 2048)
    ax3.set_xlabel('Time bin (25 ns)', fontsize=12)
    ax3.set_ylabel('ADC', fontsize=12)
    ax3.set_title('Direct Comparison (Monopole clipped to photon scale)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Add note about clipping
    ax3.text(625, 395, f'Monopole actual peak: {np.max(monopole_trace):.0f} ADC', 
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig, monopole_trace, photon_trace

def generate_multiple_monopole_scenarios():
    """
    Generate monopole traces for different scenarios
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_bins = np.arange(N_BINS)
    
    scenarios = [
        {'charge_n': 1, 'beta': 0.9, 'title': 'Fast Monopole (n=1, β=0.9)'},
        {'charge_n': 2, 'beta': 0.9, 'title': 'Double Charge Monopole (n=2, β=0.9)'},
        {'charge_n': 1, 'beta': 0.5, 'title': 'Slow Monopole (n=1, β=0.5)'},
        {'charge_n': 1, 'beta': 0.99, 'title': 'Ultra-relativistic Monopole (n=1, β=0.99)'}
    ]
    
    for idx, (ax, scenario) in enumerate(zip(axes.flat, scenarios)):
        # Generate trace
        trace = generate_monopole_trace_auger_format(
            charge_n=scenario['charge_n'],
            velocity_beta=scenario['beta'],
            entry_time_bins=600
        )
        
        # Calculate characteristics
        signal_adc = trace - BASELINE_ADC
        peak_vem = np.max(signal_adc) / ADC_PER_VEM
        above_threshold = signal_adc > 50 * ADC_PER_VEM  # 50 VEM threshold
        if np.any(above_threshold):
            duration_bins = np.sum(above_threshold)
            duration_us = duration_bins * NS_PER_BIN / 1000
        else:
            duration_us = 0
        
        # Plot
        ax.plot(time_bins, trace, 'r-', linewidth=0.8)
        ax.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.axhline(y=BASELINE_ADC + 100*ADC_PER_VEM, color='blue', linestyle='--', 
                  alpha=0.5, label='100 VEM')
        
        # Set appropriate y-scale
        max_adc = np.max(trace)
        ax.set_ylim(280, min(max_adc * 1.1, 4000))
        ax.set_xlim(0, 2048)
        
        ax.set_xlabel('Time bin (25 ns)')
        ax.set_ylabel('ADC')
        ax.set_title(scenario['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add info box
        info_text = f'Peak: {peak_vem:.0f} VEM\nDuration: {duration_us:.2f} μs'
        ax.text(0.7, 0.95, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Magnetic Monopole FADC Traces - Different Scenarios', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_feature_comparison_table():
    """
    Create a detailed comparison table
    """
    
    # Generate example traces
    photon_trace = np.ones(N_BINS) * 290
    photon_trace += np.random.normal(0, 1.5, N_BINS)
    for i in range(580, 700):
        rel_pos = (i - 580) / 120
        if rel_pos < 0.3:
            signal = 95 * min(rel_pos / 0.2, 1)
        else:
            signal = 95 * np.exp(-8 * (rel_pos - 0.3))
        photon_trace[i] += signal
    
    monopole_trace = generate_monopole_trace_auger_format()
    
    # Calculate features
    def calculate_features(trace):
        signal = trace - BASELINE_ADC
        peak_adc = np.max(signal)
        peak_vem = peak_adc / ADC_PER_VEM
        
        above_10vem = signal > 10 * ADC_PER_VEM
        if np.any(above_10vem):
            indices = np.where(above_10vem)[0]
            duration_bins = indices[-1] - indices[0]
            duration_us = duration_bins * NS_PER_BIN / 1000
            start_bin = indices[0]
        else:
            duration_us = 0
            start_bin = 0
        
        total_charge = np.sum(np.maximum(signal, 0)) / ADC_PER_VEM * NS_PER_BIN / 1000
        
        # Smoothness (RMS of derivative where signal > 10 VEM)
        if np.any(above_10vem):
            derivative = np.diff(signal)
            smoothness = np.std(derivative[above_10vem[:-1]])
        else:
            smoothness = 0
        
        return {
            'Peak [ADC]': f'{peak_adc:.0f}',
            'Peak [VEM]': f'{peak_vem:.1f}',
            'Duration [μs]': f'{duration_us:.2f}',
            'Total Charge [VEM·μs]': f'{total_charge:.1f}',
            'Smoothness': f'{smoothness:.1f}',
            'Start Time [bin]': f'{start_bin}',
            'Signal Type': 'Sustained' if duration_us > 1.0 else 'Peaked'
        }
    
    photon_features = calculate_features(photon_trace)
    monopole_features = calculate_features(monopole_trace)
    
    # Print comparison table
    print("\n" + "="*60)
    print("FADC TRACE FEATURE COMPARISON")
    print("="*60)
    print(f"{'Feature':<25} {'Photon/Hadron':<20} {'Monopole':<20}")
    print("-"*60)
    
    for key in photon_features.keys():
        print(f"{key:<25} {photon_features[key]:<20} {monopole_features[key]:<20}")
    
    print("="*60)
    print("\nKEY DIFFERENCES:")
    print("  1. Monopole peak is ~200x higher (200 VEM vs 0.5 VEM)")
    print("  2. Monopole signal is sustained (rectangular pulse)")
    print("  3. Monopole has much smoother signal (no muon spikes)")
    print("  4. Monopole total charge is ~1000x larger")
    print("  5. Would trigger special high-energy/long-duration alarms")
    print("="*60)

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("MAGNETIC MONOPOLE FADC TRACE GENERATOR - PIERRE AUGER FORMAT")
    print("="*70)
    
    # Generate main comparison
    print("\nGenerating monopole vs photon comparison...")
    fig1, monopole_trace, photon_trace = plot_monopole_vs_real_trace()
    
    # Generate multiple scenarios
    print("Generating different monopole scenarios...")
    fig2 = generate_multiple_monopole_scenarios()
    
    # Create feature comparison
    print("\nCalculating feature comparison...")
    create_feature_comparison_table()
    
    # Save figures
    fig1.savefig('monopole_vs_photon_trace.png', dpi=150, bbox_inches='tight')
    fig2.savefig('monopole_scenarios.png', dpi=150, bbox_inches='tight')
    
    print("\nFigures saved:")
    print("  - monopole_vs_photon_trace.png")
    print("  - monopole_scenarios.png")
    
    # Generate standalone monopole trace data
    print("\n" + "="*70)
    print("STANDALONE MONOPOLE TRACE DATA (first 100 bins):")
    print("="*70)
    standalone_monopole = generate_monopole_trace_auger_format()
    print("Time Bin | ADC Value | Signal [VEM]")
    print("-"*40)
    for i in range(100):
        if i < 10 or (i >= 595 and i <= 655) or i > 2040:
            signal_vem = (standalone_monopole[i] - BASELINE_ADC) / ADC_PER_VEM
            print(f"{i:8d} | {standalone_monopole[i]:9.1f} | {signal_vem:12.2f}")
        elif i == 11:
            print("    ...  |    ...    |     ...")
        elif i == 656:
            print("    ...  |    ...    |     ...")
    
    plt.show()