#!/usr/bin/env python3
"""
Generate separate FADC trace images for different particle types
Clean version without debug messages
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Pierre Auger SD constants
BASELINE_ADC = 290.0
ADC_PER_VEM = 180.0
NS_PER_BIN = 25.0
N_BINS = 2048

def generate_photon_trace():
    """Generate photon shower trace"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    trace += np.random.normal(0, 1.5, N_BINS)
    
    # Main EM peak
    for i in range(580, 700):
        rel_pos = (i - 580) / 120
        if rel_pos < 0.2:
            signal = 95 * (rel_pos / 0.2)
        elif rel_pos < 0.3:
            signal = 95
        else:
            signal = 95 * np.exp(-8 * (rel_pos - 0.3))
        trace[i] += signal + np.random.normal(0, 2)
    
    # Secondary peaks
    trace[820:825] += 35 * np.exp(-0.5 * np.arange(5))
    trace[650:653] += 25 * np.exp(-0.5 * np.arange(3))
    
    return trace

def generate_proton_trace():
    """Generate proton shower trace"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    trace += np.random.normal(0, 1.5, N_BINS)
    
    # Main EM component
    for i in range(590, 750):
        rel_pos = (i - 590) / 160
        if rel_pos < 0.15:
            signal = 80 * (rel_pos / 0.15)
        elif rel_pos < 0.25:
            signal = 80
        else:
            signal = 80 * np.exp(-6 * (rel_pos - 0.25))
        trace[i] += signal + np.random.normal(0, 2)
    
    # Muon spikes
    muon_times = [615, 680, 745, 850, 920]
    muon_amplitudes = [45, 65, 35, 40, 30]
    
    for time, amp in zip(muon_times, muon_amplitudes):
        width = np.random.randint(2, 5)
        for j in range(width):
            if time + j < N_BINS:
                trace[time + j] += amp * np.exp(-j/2)
    
    return trace

def generate_iron_trace():
    """Generate iron nucleus shower trace"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    trace += np.random.normal(0, 1.5, N_BINS)
    
    # Broader EM component
    for i in range(570, 850):
        rel_pos = (i - 570) / 280
        if rel_pos < 0.1:
            signal = 60 * (rel_pos / 0.1)
        elif rel_pos < 0.2:
            signal = 60
        else:
            signal = 60 * np.exp(-4 * (rel_pos - 0.2))
        trace[i] += signal + np.random.normal(0, 2)
    
    # Many muon spikes
    for _ in range(12):
        muon_time = 580 + np.random.randint(0, 400)
        muon_amp = np.random.exponential(30) + 20
        width = np.random.randint(2, 6)
        for j in range(width):
            if muon_time + j < N_BINS:
                trace[muon_time + j] += muon_amp * np.exp(-j/2.5)
    
    return trace

def generate_monopole_trace(charge_n=1, velocity_beta=0.9):
    """Generate magnetic monopole trace"""
    trace = np.ones(N_BINS) * BASELINE_ADC
    trace += np.random.normal(0, 1.5, N_BINS)
    
    monopole_vem = 200 * charge_n**2
    monopole_adc = monopole_vem * ADC_PER_VEM
    
    entry_time_bins = 600
    signal_duration_bins = 50
    
    # Rectangular pulse
    for i in range(entry_time_bins, min(entry_time_bins + signal_duration_bins, N_BINS)):
        rel_pos = (i - entry_time_bins) / signal_duration_bins
        
        if rel_pos < 0.05:
            signal_fraction = rel_pos / 0.05
            signal_adc = monopole_adc * (1 - np.exp(-10 * signal_fraction))
        elif rel_pos > 0.95:
            signal_fraction = (1 - rel_pos) / 0.05
            signal_adc = monopole_adc * (1 - np.exp(-10 * signal_fraction))
        else:
            signal_adc = monopole_adc * (1 + 0.01 * np.random.randn())
        
        trace[i] = BASELINE_ADC + signal_adc
    
    # Afterglow
    afterglow_start = entry_time_bins + signal_duration_bins
    for i in range(afterglow_start, min(afterglow_start + 100, N_BINS)):
        afterglow_adc = monopole_adc * 0.02 * np.exp(-0.05 * (i - afterglow_start))
        trace[i] += afterglow_adc
    
    trace = np.minimum(trace, 4000)
    return trace

def plot_trace(trace, title, filename, ylim=(280, 400), color='b', highlight_muons=False):
    """Plot and save a single trace"""
    fig, ax = plt.subplots(figsize=(14, 6))
    time_bins = np.arange(N_BINS)
    
    ax.plot(time_bins, trace, color=color, linewidth=0.8)
    ax.set_xlim(0, 2048)
    ax.set_ylim(ylim)
    ax.set_xlabel('Time bin (25 ns)', fontsize=12)
    ax.set_ylabel('ADC', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add baseline
    ax.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.3, label='Baseline')
    
    # Add stats box
    mean = np.mean(trace)
    rms = np.std(trace)
    stats_text = f'Entries    2048\nMean      {mean:.0f}\nRMS       {rms:.1f}'
    ax.text(0.89, 0.75, stats_text, transform=ax.transAxes, fontsize=9, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Highlight muons if requested
    if highlight_muons:
        signal = trace - BASELINE_ADC
        peak_indices = np.where(signal > 30)[0]
        if len(peak_indices) > 0:
            for peak in peak_indices[:3]:
                ax.plot(peak, trace[peak], 'ro', markersize=6)
    
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_monopole_special(trace, title, filename):
    """Plot monopole with special annotations"""
    fig, ax = plt.subplots(figsize=(14, 6))
    time_bins = np.arange(N_BINS)
    
    ax.plot(time_bins, trace, 'r-', linewidth=0.8)
    ax.set_xlim(0, 2048)
    ax.set_ylim(280, 4100)
    ax.set_xlabel('Time bin (25 ns)', fontsize=12)
    ax.set_ylabel('ADC', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold', color='red')
    ax.grid(True, alpha=0.3)
    
    # Reference lines
    ax.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.3, label='Baseline')
    ax.axhline(y=BASELINE_ADC + 100*ADC_PER_VEM, color='blue', linestyle='--', alpha=0.5, label='100 VEM')
    ax.axhline(y=4000, color='orange', linestyle='--', alpha=0.5, label='ADC Saturation')
    
    # Highlight monopole region
    rect = Rectangle((600, 280), 50, 3720, alpha=0.2, facecolor='red', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    # Stats box
    mean = np.mean(trace)
    rms = np.std(trace)
    stats_text = f'Entries    2048\nMean      {mean:.0f}\nRMS       {rms:.1f}'
    ax.text(0.89, 0.75, stats_text, transform=ax.transAxes, fontsize=9,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison():
    """Create comparison plot"""
    fig, ax = plt.subplots(figsize=(14, 6))
    time_bins = np.arange(N_BINS)
    
    photon = generate_photon_trace()
    proton = generate_proton_trace()
    iron = generate_iron_trace()
    monopole = generate_monopole_trace()
    
    ax.plot(time_bins, photon, 'b-', linewidth=0.8, alpha=0.7, label='Photon')
    ax.plot(time_bins, proton, 'g-', linewidth=0.8, alpha=0.7, label='Proton')
    ax.plot(time_bins, iron, 'purple', linewidth=0.8, alpha=0.7, label='Iron')
    ax.plot(time_bins, np.minimum(monopole, 400), 'r-', linewidth=1.2, alpha=0.8, label='Monopole (clipped)')
    
    ax.set_xlim(0, 2048)
    ax.set_ylim(280, 420)
    ax.set_xlabel('Time bin (25 ns)', fontsize=12)
    ax.set_ylabel('ADC', fontsize=12)
    ax.set_title('FADC Trace Comparison - All Particle Types', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=BASELINE_ADC, color='gray', linestyle='--', alpha=0.3)
    
    actual_peak = np.max(monopole)
    ax.text(625, 410, f'Monopole actual peak: {actual_peak:.0f} ADC',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('trace_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all trace images"""
    # Create output directory
    os.makedirs("fadc_traces", exist_ok=True)
    os.chdir("fadc_traces")
    
    # Generate cosmic ray traces
    photon = generate_photon_trace()
    plot_trace(photon, 'Event 2, Station 4001, PMT 3 [MOPS trigger] [photon primary]', 
               'trace_photon.png', color='b')
    
    proton = generate_proton_trace()
    plot_trace(proton, 'Event 3, Station 4001, PMT 3 [MOPS trigger] [proton primary]', 
               'trace_proton.png', color='g', highlight_muons=True)
    
    iron = generate_iron_trace()
    plot_trace(iron, 'Event 4, Station 4001, PMT 3 [MOPS trigger] [iron primary]', 
               'trace_iron.png', ylim=(280, 420), color='purple')
    
    # Generate monopole traces
    monopole = generate_monopole_trace()
    peak_vem = (np.max(monopole) - BASELINE_ADC) / ADC_PER_VEM
    plot_monopole_special(monopole, f'Event X, Station 4001, PMT 3 [MONOPOLE CANDIDATE] [Peak: {peak_vem:.0f} VEM]', 
                         'trace_monopole.png')
    
    # Saturated monopole
    monopole_sat = np.minimum(monopole, 1023)
    plot_trace(monopole_sat, 'Event X, Station 4001, PMT 3 [MONOPOLE - ADC SATURATED]', 
               'trace_monopole_saturated.png', ylim=(280, 1100), color='r')
    
    # Monopole variations
    monopole_slow = generate_monopole_trace(charge_n=1, velocity_beta=0.5)
    plot_trace(monopole_slow, 'Slow Monopole (n=1, β=0.5)', 
               'trace_monopole_slow.png', ylim=(280, 4100), color='r')
    
    monopole_double = generate_monopole_trace(charge_n=2, velocity_beta=0.9)
    plot_trace(np.minimum(monopole_double, 4000), 'Double Charge Monopole (n=2, β=0.9)', 
               'trace_monopole_double.png', ylim=(280, 4100), color='r')
    
    # Comparison plot
    plot_comparison()
    
    print("All trace images generated successfully in 'fadc_traces' directory:")
    print("  - trace_photon.png")
    print("  - trace_proton.png")
    print("  - trace_iron.png")
    print("  - trace_monopole.png")
    print("  - trace_monopole_saturated.png")
    print("  - trace_monopole_slow.png")
    print("  - trace_monopole_double.png")
    print("  - trace_comparison.png")

if __name__ == "__main__":
    main()