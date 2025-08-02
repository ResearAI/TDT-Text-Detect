#!/usr/bin/env python3
"""
TDT (Temporal Discrepancy Tomography) Demo with Professional Report Generation

This script demonstrates the TDT method for AI-generated text detection
with professional PDF report generation using real detection data.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for better quality
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import TDT components
from detectors import get_detector
from utils import load_json

# Import visualization dependencies
try:
    import pywt
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import norm
    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False
    print("Warning: PyWavelets not installed. Install with: pip install PyWavelets scipy")

# Import report generation dependencies
try:
    import markdown
    import weasyprint
    from PIL import Image
    HAS_REPORT_DEPS = True
except ImportError:
    HAS_REPORT_DEPS = False
    print("Warning: Report generation dependencies not installed.")
    print("Install with: pip install markdown weasyprint Pillow")


class TDTDemo:
    """Interactive TDT demonstration with professional report generation"""
    
    def __init__(self, enable_viz: bool = True):
        """Initialize TDT demo"""
        self.enable_viz = enable_viz and HAS_WAVELET
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Colors for consistent theming
        self.colors = {
            'primary': '#1976d2',
            'success': '#388e3c',
            'danger': '#d32f2f',
            'warning': '#f57c00',
            'info': '#0288d1',
            'morphological': '#FF6B6B',
            'syntactic': '#4ECDC4',
            'discourse': '#45B7D1'
        }
        
        # Initialize detector
        print("Initializing TDT detector...")
        self.detector = get_detector('tdt')
        print("TDT detector ready!")
        
        # Ensure the detector is TDT class for visualization
        from detectors.tdt import TDT
        if not isinstance(self.detector, TDT):
            print("Warning: Detector is not TDT class, visualization may be limited.")
        
    def create_signal_plot(self, token_discrepancies: np.ndarray, tokens: List[str], 
                          save_path: str) -> Dict:
        """Create token-level discrepancy signal plot using real data"""
        n_tokens = len(token_discrepancies)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot signal
        positions = np.arange(n_tokens)
        ax.plot(positions, token_discrepancies, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax.fill_between(positions, token_discrepancies, alpha=0.3, color='skyblue')
        
        # Add threshold line (z-score threshold)
        threshold = 2.0  # 2 standard deviations
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Anomaly Threshold (z={threshold})')
        ax.axhline(y=-threshold, color='red', linestyle='--', linewidth=2)
        
        # Highlight suspicious regions
        suspicious = np.abs(token_discrepancies) > threshold
        if np.any(suspicious):
            ax.fill_between(positions, -10, 10, where=suspicious, 
                           color='red', alpha=0.2, transform=ax.get_xaxis_transform(),
                           label='Suspicious Regions')
        
        # Add token labels for the first few tokens if space allows
        if n_tokens <= 30:
            for i, token in enumerate(tokens[:n_tokens]):
                if len(token) > 10:
                    token = token[:10] + '...'
                ax.text(i, -0.5, token, rotation=45, ha='right', va='top', fontsize=8)
        
        # Styling
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Normalized Discrepancy (z-score)', fontsize=12)
        ax.set_title('Token-Level Statistical Discrepancy Signal (Real Data)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='upper right')
        ax.set_xlim(-0.5, n_tokens-0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        return {
            'mean': float(np.mean(token_discrepancies)),
            'std': float(np.std(token_discrepancies)),
            'max': float(np.max(np.abs(token_discrepancies))),
            'suspicious_ratio': float(np.mean(suspicious)),
            'n_tokens': n_tokens
        }
    
    def create_scalogram_plot(self, continuous_signal: np.ndarray, wavelet_coeffs: np.ndarray,
                             save_path: str) -> Dict:
        """Create wavelet scalogram visualization using real wavelet coefficients"""
        power = np.abs(wavelet_coeffs) ** 2
        
        # Create figure with multiple subplots for better visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[4, 1])
        
        ax1 = fig.add_subplot(gs[0, :])  # Continuous signal
        ax2 = fig.add_subplot(gs[1, 0])  # Scalogram
        ax3 = fig.add_subplot(gs[1, 1])  # Scale power profile
        ax4 = fig.add_subplot(gs[2, 0])  # Time-averaged power per band
        ax5 = fig.add_subplot(gs[2, 1])  # Peak detection
        
        # Plot continuous signal
        signal_points = np.linspace(0, 1, len(continuous_signal))
        ax1.plot(signal_points, continuous_signal, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(signal_points, continuous_signal, alpha=0.3, color='skyblue')
        ax1.set_xlabel('Normalized Position', fontsize=11)
        ax1.set_ylabel('Signal Amplitude', fontsize=11)
        ax1.set_title('Continuous Signal (Interpolated with Gaussian Smoothing)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Add signal statistics
        signal_mean = np.mean(continuous_signal)
        signal_std = np.std(continuous_signal)
        ax1.axhline(y=signal_mean, color='red', linestyle=':', alpha=0.7, label=f'Mean: {signal_mean:.3f}')
        ax1.axhline(y=signal_mean + signal_std, color='orange', linestyle=':', alpha=0.5, label=f'±1 STD')
        ax1.axhline(y=signal_mean - signal_std, color='orange', linestyle=':', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=9)
        
        # Enhanced scalogram visualization
        # Use different normalization to preserve relative variations
        power_db = 10 * np.log10(power + 1e-10)  # Convert to decibel scale
        
        # Global normalization to preserve relative differences
        vmin, vmax = np.percentile(power_db, [5, 95])  # Use percentiles for robustness
        
        im = ax2.imshow(power_db, extent=[0, 1, 1, 12], 
                       cmap='jet', aspect='auto', interpolation='bilinear',
                       origin='lower', vmin=vmin, vmax=vmax)
        
        # Add band labels
        band_boundaries = [4.5, 8.5]
        for boundary in band_boundaries:
            ax2.axhline(y=boundary, color='white', linestyle='--', linewidth=2, alpha=0.7)
        
        # Label bands
        ax2.text(0.02, 2.5, 'Morphological', color='white', 
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        ax2.text(0.02, 6.5, 'Syntactic', color='white', 
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        ax2.text(0.02, 10.5, 'Discourse', color='white', 
                fontsize=11, fontweight='bold', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        
        # Styling
        ax2.set_xlabel('Normalized Position', fontsize=12)
        ax2.set_ylabel('Scale (Linguistic Level)', fontsize=12)
        ax2.set_title('Continuous Wavelet Transform - Time-Scale Representation (dB scale)', 
                     fontsize=14, fontweight='bold')
        
        # Colorbar with dB label
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Power (dB)', fontsize=11)
        
        # Add scale power profile (right panel)
        scale_power = np.mean(power, axis=1)  # Average power across time for each scale
        scales = np.arange(1, 13)
        ax3.plot(scale_power, scales, 'r-', linewidth=2)
        ax3.fill_betweenx(scales, 0, scale_power, alpha=0.3, color='red')
        ax3.set_ylabel('Scale', fontsize=11)
        ax3.set_xlabel('Average Power', fontsize=11)
        ax3.set_title('Scale Power Profile', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.set_ylim(1, 12)
        
        # Add band boundaries
        for boundary in band_boundaries:
            ax3.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Time-averaged power per band (bottom left)
        morph_power = power[0:4, :]
        syn_power = power[4:8, :]
        disc_power = power[8:12, :]
        
        # Calculate time series of band powers
        time_points = np.linspace(0, 1, morph_power.shape[1])
        morph_ts = np.mean(morph_power, axis=0)
        syn_ts = np.mean(syn_power, axis=0)
        disc_ts = np.mean(disc_power, axis=0)
        
        ax4.plot(time_points, morph_ts, color=self.colors['morphological'], 
                label='Morphological', linewidth=2, alpha=0.8)
        ax4.plot(time_points, syn_ts, color=self.colors['syntactic'], 
                label='Syntactic', linewidth=2, alpha=0.8)
        ax4.plot(time_points, disc_ts, color=self.colors['discourse'], 
                label='Discourse', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Normalized Position', fontsize=11)
        ax4.set_ylabel('Band Power', fontsize=11)
        ax4.set_title('Time-varying Band Power', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3, linestyle=':')
        
        # Peak detection and statistics (bottom right)
        # Find peaks in each band
        from scipy.signal import find_peaks
        
        morph_peaks, _ = find_peaks(morph_ts, height=np.mean(morph_ts))
        syn_peaks, _ = find_peaks(syn_ts, height=np.mean(syn_ts))
        disc_peaks, _ = find_peaks(disc_ts, height=np.mean(disc_ts))
        
        peak_counts = [len(morph_peaks), len(syn_peaks), len(disc_peaks)]
        band_names = ['Morph', 'Syn', 'Disc']
        colors = [self.colors['morphological'], self.colors['syntactic'], self.colors['discourse']]
        
        bars = ax5.bar(band_names, peak_counts, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Number of Peaks', fontsize=11)
        ax5.set_title('Peak Count per Band', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        # Add value labels on bars
        for bar, count in zip(bars, peak_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate band energies
        morph_energy = np.sqrt(np.sum(power[0:4, :]))
        syn_energy = np.sqrt(np.sum(power[4:8, :]))
        disc_energy = np.sqrt(np.sum(power[8:12, :]))
        
        # Normalize energies
        total_energy = morph_energy + syn_energy + disc_energy
        
        # Additional statistics
        morph_var = np.var(morph_ts)
        syn_var = np.var(syn_ts)
        disc_var = np.var(disc_ts)
        
        return {
            'morphological_energy': float(morph_energy),
            'syntactic_energy': float(syn_energy),
            'discourse_energy': float(disc_energy),
            'morph_ratio': float(morph_energy / total_energy),
            'syn_ratio': float(syn_energy / total_energy),
            'disc_ratio': float(disc_energy / total_energy),
            'morph_peaks': len(morph_peaks),
            'syn_peaks': len(syn_peaks),
            'disc_peaks': len(disc_peaks),
            'morph_variance': float(morph_var),
            'syn_variance': float(syn_var),
            'disc_variance': float(disc_var)
        }
    
    def create_wavelet_detail_plot(self, wavelet_coeffs: np.ndarray, save_path: str):
        """Create detailed wavelet coefficient visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Band-wise coefficient visualization
        morph_coeffs = wavelet_coeffs[0:4, :]
        syn_coeffs = wavelet_coeffs[4:8, :]
        disc_coeffs = wavelet_coeffs[8:12, :]
        
        bands = [morph_coeffs, syn_coeffs, disc_coeffs]
        band_names = ['Morphological (Scales 1-4)', 'Syntactic (Scales 5-8)', 'Discourse (Scales 9-12)']
        colors_map = ['Reds', 'Blues', 'Greens']
        
        for idx, (band, name, cmap) in enumerate(zip(bands, band_names, colors_map)):
            ax = axes[idx]
            
            # Use absolute values for visualization
            band_abs = np.abs(band)
            
            # Create heatmap
            im = ax.imshow(band_abs, aspect='auto', cmap=cmap, 
                          extent=[0, 1, 0, band.shape[0]], 
                          interpolation='bilinear', origin='lower')
            
            ax.set_xlabel('Normalized Position', fontsize=11)
            ax.set_ylabel('Scale within Band', fontsize=11)
            ax.set_title(f'{name} Wavelet Coefficients', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Coefficient Magnitude', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle=':', color='white')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_detail.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_plot(self, features: Dict, save_path: str):
        """Create multi-scale feature extraction visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute energy values
        names = ['Morphological\n(Word-level)', 
                'Syntactic\n(Phrase-level)', 
                'Discourse\n(Paragraph-level)']
        values = [features['morphological_energy'], 
                 features['syntactic_energy'],
                 features['discourse_energy']]
        colors = [self.colors['morphological'], 
                 self.colors['syntactic'],
                 self.colors['discourse']]
        
        # Create bars for absolute values
        x = np.arange(len(names))
        bars = ax1.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Styling for absolute values
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylabel('Energy (Frobenius Norm)', fontsize=12)
        ax1.set_title('Absolute Energy Values', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle=':')
        ax1.set_ylim(0, max(values) * 1.2)
        
        # Energy distribution pie chart
        ratios = [features['morph_ratio'], features['syn_ratio'], features['disc_ratio']]
        wedges, texts, autotexts = ax2.pie(ratios, labels=names, colors=colors, 
                                           autopct='%1.1f%%', startangle=90)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_weight('bold')
            autotext.set_fontsize(11)
        
        ax2.set_title('Energy Distribution', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_markdown_report(self, text: str, is_ai: bool, score: float,
                               signal_stats: Dict, scalogram_features: Dict,
                               timestamp: str, details: Dict) -> str:
        """Generate markdown report content with real detection details"""
        
        result_text = "AI-Generated" if is_ai else "Human-Written"
        result_color = "red" if is_ai else "green"
        confidence = abs(score - 0.0023) / 0.0023 * 100
        
        # Get wavelet features if available
        wavelet_features = details.get('wavelet_features', [score, score, score])
        
        markdown_content = f"""
# TDT Analysis Report

**Generated on:** {timestamp}

---

## Executive Summary

<div style="background-color: #f0f4f8; padding: 20px; border-radius: 10px; margin: 20px 0;">

Detection Result: <span style="color: {result_color}; font-size: 24px; font-weight: bold;">{result_text}</span>

</div>

## 1. Input Text Analysis

<div style="background-color: #fafafa; padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 20px 0;">
<p style="font-family: monospace; line-height: 1.6;">
{text[:500]}{'...' if len(text) > 500 else ''}
</p>
</div>

### Text Statistics
- **Words:** {len(text.split())}
- **Characters:** {len(text)}
- **Tokens Processed:** {signal_stats['n_tokens']}
- **Average Token Discrepancy:** {signal_stats['mean']:.4f}

## 2. Token-Level Discrepancy Analysis

![Token-Level Discrepancy Signal](signal_plot.png)

### Signal Statistics
- **Mean Discrepancy (z-score):** {signal_stats['mean']:.6f}
- **Standard Deviation:** {signal_stats['std']:.6f}
- **Maximum Absolute Value:** {signal_stats['max']:.6f}
- **Suspicious Token Ratio:** {signal_stats['suspicious_ratio']*100:.1f}%

The token-level analysis shows how each token's statistical properties deviate from expected patterns.
Tokens with |z-score| > 2 are considered statistically anomalous.

## 3. Wavelet Transform Analysis

![Wavelet Scalogram](scalogram_plot.png)

The Continuous Wavelet Transform (CWT) decomposes the discrepancy signal into multiple scales:

- **Morphological Band (Scales 1-4):** Captures word-level anomalies and local patterns
- **Syntactic Band (Scales 5-8):** Reveals phrase and sentence structure irregularities
- **Discourse Band (Scales 9-12):** Shows paragraph-level coherence and long-range dependencies

### Energy Distribution
- **Morphological:** {scalogram_features['morph_ratio']*100:.1f}% of total energy
- **Syntactic:** {scalogram_features['syn_ratio']*100:.1f}% of total energy
- **Discourse:** {scalogram_features['disc_ratio']*100:.1f}% of total energy

### Temporal Analysis
- **Morphological Peaks:** {scalogram_features.get('morph_peaks', 'N/A')} detected anomaly regions
- **Syntactic Peaks:** {scalogram_features.get('syn_peaks', 'N/A')} detected anomaly regions
- **Discourse Peaks:** {scalogram_features.get('disc_peaks', 'N/A')} detected anomaly regions

### Band Variance (Signal Stability)
- **Morphological Variance:** {scalogram_features.get('morph_variance', 0):.6f}
- **Syntactic Variance:** {scalogram_features.get('syn_variance', 0):.6f}
- **Discourse Variance:** {scalogram_features.get('disc_variance', 0):.6f}

## 4. Multi-Scale Feature Extraction

![Multi-Scale Features](feature_plot.png)

### Detailed Wavelet Coefficient Analysis

![Wavelet Coefficient Details](feature_plot_detail.png)

The detailed coefficient visualization shows the wavelet transform magnitude at each linguistic band separately, revealing:
- **Morphological Band**: Fine-grained word-level patterns and local anomalies
- **Syntactic Band**: Medium-scale phrase and sentence structure variations
- **Discourse Band**: Large-scale paragraph and document-level coherence patterns

### Extracted Features (Normalized)
- **Morphological Energy:** {wavelet_features[0]:.4f}
- **Syntactic Energy:** {wavelet_features[1]:.4f}
- **Discourse Energy:** {wavelet_features[2]:.4f}

### Raw Energy Values
- **Morphological (Frobenius Norm):** {scalogram_features['morphological_energy']:.4f}
- **Syntactic (Frobenius Norm):** {scalogram_features['syntactic_energy']:.4f}
- **Discourse (Frobenius Norm):** {scalogram_features['discourse_energy']:.4f}

## 5. Technical Details

### Method: Temporal Discrepancy Tomography (TDT)

TDT performs real-time analysis using the following pipeline:

#### **Token-Level Discrepancy Computation:**
   - For each token, compute: $z_i = \\frac{{\\log p_{{\\text{{score}}}}(x_i) - \\mathbb{{E}}[\\log p_{{\\text{{score}}}}(x_i)]}}{{\\sqrt{{\\text{{Var}}[\\log p_{{\\text{{score}}}}(x_i)]}}}}$
   - Uses Student's t-distribution normalization for robustness

#### **Signal Transformation:**
   - Gaussian KDE converts discrete token discrepancies to continuous signal
   - Bandwidth selected using Scott's rule for optimal smoothing

#### **Wavelet Analysis:**
   - Complex Morlet wavelet (cmor1.5-1.0) for time-frequency localization
   - 12 scales logarithmically spaced to capture linguistic hierarchy

#### **Feature Extraction:**
   - Frobenius norm of wavelet coefficients in each linguistic band
   - Features normalized to match original detection score magnitude

### Key Insights from This Analysis

- **Non-stationarity Score:** {abs(signal_stats['std'] / (signal_stats['mean'] + 1e-6)):.2f}
- **Energy Concentration:** {'Morphological' if scalogram_features['morph_ratio'] > 0.4 else 'Distributed'} pattern
- **Anomaly Type:** {'Local' if signal_stats['suspicious_ratio'] < 0.2 else 'Global'} anomalies detected

---

<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #e8f0fe; border-radius: 10px;">
<p style="font-size: 14px; color: #666;">
Report generated by TDT (Temporal Discrepancy Tomography)<br>
Real-time detection using Falcon-7B models<br>
For more information, visit: https://github.com/anonymous/TDT
</p>
</div>
"""
        
        return markdown_content
    
    def generate_pdf_report(self, text: str, is_ai: bool, score: float, 
                           details: Dict, output_path: str = None) -> str:
        """Generate professional PDF report using real detection data"""
        
        if not HAS_REPORT_DEPS:
            print("Error: Report generation dependencies not installed.")
            print("Install with: pip install markdown weasyprint Pillow")
            return None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if output_path is None:
            output_path = f"tdt_report_{int(time.time())}.pdf"
        
        print("\n📊 Generating professional PDF report with real data...")
        
        # Extract real data
        token_discrepancies = details['token_discrepancies']
        continuous_signal = details['continuous_signal']
        wavelet_coeffs = details['wavelet_coeffs']
        tokens = self.detector.last_tokens[:len(token_discrepancies)]
        
        # Generate individual plots with real data
        print("  • Creating token-level discrepancy plot (real data)...")
        signal_plot_path = os.path.join(self.cache_dir, 'signal_plot.png')
        signal_stats = self.create_signal_plot(token_discrepancies, tokens, signal_plot_path)
        
        print("  • Creating wavelet scalogram (real coefficients)...")
        scalogram_plot_path = os.path.join(self.cache_dir, 'scalogram_plot.png')
        scalogram_features = self.create_scalogram_plot(continuous_signal, wavelet_coeffs, 
                                                       scalogram_plot_path)
        
        print("  • Creating feature extraction plot...")
        feature_plot_path = os.path.join(self.cache_dir, 'feature_plot.png')
        self.create_feature_plot(scalogram_features, feature_plot_path)
        
        print("  • Creating detailed wavelet coefficient visualization...")
        self.create_wavelet_detail_plot(wavelet_coeffs, feature_plot_path)
        
        # Generate markdown content
        print("  • Generating report content...")
        markdown_content = self.generate_markdown_report(
            text, is_ai, score, signal_stats, scalogram_features, timestamp, details
        )
        
        # Save markdown file
        markdown_path = os.path.join(self.cache_dir, 'report.md')
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Convert to HTML
        print("  • Converting to PDF...")
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'codehilite'])
        
        # Add CSS styling
        css = """
        <style>
        body {
            font-family: 'DejaVu Sans', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #1976d2;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 10px;
        }
        h2 {
            color: #1565c0;
            margin-top: 30px;
        }
        h3 {
            color: #0d47a1;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .highlight {
            background-color: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
        }
        </style>
        """
        
        # Combine HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>TDT Analysis Report</title>
            {css}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Save HTML for debugging
        html_path = os.path.join(self.cache_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        # Convert HTML to PDF using WeasyPrint
        try:
            weasyprint.HTML(string=full_html, base_url=self.cache_dir).write_pdf(output_path)
            print(f"\n✅ Report saved to: {output_path}")
        except Exception as e:
            print(f"\n❌ Error generating PDF: {e}")
            print(f"   HTML report saved to: {html_path}")
            return html_path
        
        return output_path
    
    def detect_text(self, text: str, threshold: float = 0.0023, 
                   generate_report: bool = True, output_path: str = None) -> Tuple[bool, float]:
        """Perform detection on input text with real data visualization"""
        
        print("\n🔍 Running TDT detection with real data analysis...")
        
        # Run detector with details
        result = self.detector.compute_crit(text, return_details=True)
        
        # Handle different result formats
        if isinstance(result, tuple):
            crit, details = result
            # Check if crit is a list (wavelet features)
            if isinstance(crit, list):
                score = crit[0]  # Use morphological energy as primary score
                print(f"  • Morphological energy: {crit[0]:.4f}")
                print(f"  • Syntactic energy: {crit[1]:.4f}")
                print(f"  • Discourse energy: {crit[2]:.4f}")
                if 'wavelet_features' not in details:
                    details['wavelet_features'] = crit
            else:
                score = float(crit) if not isinstance(crit, float) else crit
                print(f"  • Detection score: {score:.4f}")
            
            # Check if wavelet features are available in details
            if 'wavelet_features' in details and isinstance(details['wavelet_features'], list) and len(details['wavelet_features']) >= 3:
                wf = details['wavelet_features']
                if not isinstance(crit, list):  # Don't print twice
                    print(f"  • Morphological energy: {wf[0]:.4f}")
                    print(f"  • Syntactic energy: {wf[1]:.4f}")
                    print(f"  • Discourse energy: {wf[2]:.4f}")
        elif isinstance(result, list) and len(result) >= 3:
            # Result is wavelet features without details
            score = result[0]  # Use morphological energy as primary score
            details = {'wavelet_features': result}
            print(f"  • Morphological energy: {result[0]:.4f}")
            print(f"  • Syntactic energy: {result[1]:.4f}")
            print(f"  • Discourse energy: {result[2]:.4f}")
        else:
            # Fallback for single scalar result
            score = float(result) if not isinstance(result, float) else result
            details = {}
            print(f"  • Detection score: {score:.4f}")
        
        # Detection decision
        is_ai_generated = score > threshold
        confidence = abs(score - threshold) / threshold * 100
        
        print(f"\n📊 Results:")
        print(f"  • Score: {score:.6f}")
        print(f"  • Threshold: {threshold:.6f}")
        print(f"  • Detection: {'AI-GENERATED' if is_ai_generated else 'HUMAN-WRITTEN'}")
        print(f"  • Confidence: {confidence:.1f}%")
        
        # Generate report if requested and we have details
        if generate_report and self.enable_viz and details:
            self.generate_pdf_report(text, is_ai_generated, score, details, output_path)
        elif generate_report and not details:
            print("\n⚠️  Cannot generate detailed report without wavelet analysis.")
            print("    The detector may not have wavelet features enabled.")
        
        return is_ai_generated, score
    
    def run_examples(self):
        """Run detection on example texts"""
        
        print("\n" + "="*60)
        print("RUNNING EXAMPLE DETECTIONS WITH REAL DATA")
        print("="*60)
        
        examples = [
            {
                'text': "The weather today is quite pleasant. I decided to take a walk in the park and enjoy the sunshine. The birds were singing and children were playing. It was a perfect day for outdoor activities. The trees provided shade from the warm sun, and a gentle breeze made everything feel fresh and alive.",
                'label': 'Human'
            },
            {
                'text': "Artificial intelligence continues to revolutionize various industries through innovative applications. Machine learning algorithms process vast amounts of data to identify complex patterns and relationships. Deep neural networks enable sophisticated image recognition capabilities that surpass human performance. Natural language processing facilitates seamless human-computer interaction through advanced computational linguistics and transformer architectures.",
                'label': 'AI'
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n--- Example {i} ({example['label']}) ---")
            output_path = f"tdt_example_{i}_report.pdf"
            self.detect_text(example['text'], output_path=output_path)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        
        print("\n" + "="*60)
        print("TDT INTERACTIVE MODE - REAL DATA ANALYSIS")
        print("="*60)
        print("\nEnter text to analyze (press Enter twice to submit):")
        
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                if lines:
                    break
        
        text = ' '.join(lines)
        if text.strip():
            self.detect_text(text)
        else:
            print("No text provided.")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='TDT (Temporal Discrepancy Tomography) Demo with Real Data Visualization'
    )
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--examples', action='store_true', help='Run example detections')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--output', type=str, help='Output path for PDF report')
    parser.add_argument('--no-report', action='store_true', help='Disable PDF report generation')
    parser.add_argument('--threshold', type=float, default=-0.09633050877193311, 
                       help='Detection threshold (default: -0.09633050877193311)')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = TDTDemo(enable_viz=not args.no_report)
    
    # Run appropriate mode
    if args.examples:
        demo.run_examples()
    elif args.text:
        demo.detect_text(args.text, args.threshold, 
                        not args.no_report, args.output)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            demo.detect_text(text, args.threshold, 
                           not args.no_report, args.output)
        except Exception as e:
            print(f"Error reading file: {e}")
    elif args.interactive:
        demo.interactive_mode()
    else:
        # Default to interactive mode
        demo.interactive_mode()


if __name__ == '__main__':
    main()