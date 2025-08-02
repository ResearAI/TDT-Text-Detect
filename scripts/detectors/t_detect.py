# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import pywt
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def transform_discrete_sequence(discrepancy_sequence, bandwidth='scott'):
    """
    Transform discrete token-level discrepancies to continuous signal.
    Uses Gaussian KDE with optimal bandwidth selection.
    
    Args:
        discrepancy_sequence: Numpy array of token-level discrepancies
        bandwidth: Bandwidth selection method for KDE. Options:
            - 'scott': Scott's rule (default)
            - 'silverman': Silverman's rule
            - float: Fixed bandwidth value
            - 'cv': Cross-validated bandwidth (uses leave-one-out)
    
    Returns:
        continuous_signal: Continuous representation of the sequence
    """
    # Handle edge cases
    if len(discrepancy_sequence) == 0:
        return np.zeros(1000)
    
    if len(discrepancy_sequence) == 1:
        # For single token, create a narrow Gaussian centered at that position
        t = np.linspace(0, 1, 1000)
        signal = np.exp(-0.5 * ((t - 0.5) / 0.1)**2) * discrepancy_sequence[0]
        return signal
    
    # Handle cross-validation bandwidth selection
    if bandwidth == 'cv':
        # Implement leave-one-out cross-validation for bandwidth selection
        from sklearn.model_selection import LeaveOneOut
        from sklearn.neighbors import KernelDensity
        
        # Test a range of bandwidths
        bandwidths = np.logspace(-2, 1, 20)
        scores = []
        
        loo = LeaveOneOut()
        data = discrepancy_sequence.reshape(-1, 1)
        
        for bw in bandwidths:
            log_likelihood = 0
            for train_idx, test_idx in loo.split(data):
                train_data = data[train_idx]
                test_data = data[test_idx]
                
                kde_cv = KernelDensity(bandwidth=bw, kernel='gaussian')
                kde_cv.fit(train_data)
                log_likelihood += kde_cv.score(test_data)
            
            scores.append(log_likelihood / len(data))
        
        # Select bandwidth with highest cross-validated likelihood
        optimal_bw = bandwidths[np.argmax(scores)]
        kde = gaussian_kde(discrepancy_sequence, bw_method=optimal_bw)
    else:
        # Apply kernel density estimation with specified method
        kde = gaussian_kde(discrepancy_sequence, bw_method=bandwidth)
    
    # Create continuous representation
    t = np.linspace(0, len(discrepancy_sequence), 1000)
    continuous_signal = kde(t)
    
    return continuous_signal


def get_wavelet_features(continuous_signal, wavelet='cmor1.5-1.0', max_scales=12, 
                        scale_boundaries=None, energy_method='frobenius', boundary_mode='zero', return_coeffs=False):
    """
    Extract multi-scale wavelet features from continuous signal with configurable boundary handling.
    Returns weighted energy at morphological, syntactic, and discourse scales.
    
    Args:
        continuous_signal: Continuous signal from KDE
        wavelet: Wavelet function to use
        max_scales: Maximum number of scales
        scale_boundaries: Tuple of (morph_end, syn_end) or None for default (4, 8)
        energy_method: Energy calculation method ('frobenius', 'l2', 'shannon', 'wavelet_packet', 'local_maxima')
        boundary_mode: Boundary handling strategy ('zero', 'symmetric', 'periodic', 'reflect', 'constant', 'truncate')
        return_coeffs: Whether to return the coefficient matrix for visualization
    
    Returns:
        features: List of [morphological, syntactic, discourse] weighted energies
        boundary_info: Dict with boundary effect analysis information
        coeffs: (Optional) Wavelet coefficient matrix if return_coeffs=True
    """
    # Handle edge cases
    if np.all(continuous_signal == 0) or len(continuous_signal) == 0:
        return [0.0, 0.0, 0.0], {'boundary_mode': boundary_mode, 'signal_length': 0, 'edge_to_center_ratio': 1.0, 'edge_correlation': 0.0, 'left_edge_magnitude': 0.0, 'right_edge_magnitude': 0.0, 'avg_magnitude_profile': []}
    
    # Store original signal for boundary effect analysis
    original_signal = continuous_signal.copy()
    original_length = len(continuous_signal)
    
    # Apply boundary handling strategy
    if boundary_mode == 'zero':
        # Default behavior - no padding needed as CWT handles this internally
        processed_signal = continuous_signal
        pad_length = 0
    elif boundary_mode == 'symmetric':
        # Symmetric padding (reflection)
        pad_length = min(original_length // 4, 50)  # Limit padding to prevent excessive computation
        processed_signal = np.pad(continuous_signal, pad_length, mode='symmetric')
    elif boundary_mode == 'periodic':
        # Periodic padding (wrap around)
        pad_length = min(original_length // 4, 50)
        processed_signal = np.pad(continuous_signal, pad_length, mode='wrap')
    elif boundary_mode == 'reflect':
        # Reflection without repeating edge values
        pad_length = min(original_length // 4, 50)
        processed_signal = np.pad(continuous_signal, pad_length, mode='reflect')
    elif boundary_mode == 'constant':
        # Constant padding with edge values
        pad_length = min(original_length // 4, 50)
        processed_signal = np.pad(continuous_signal, pad_length, mode='edge')
    elif boundary_mode == 'truncate':
        # No padding, accept boundary artifacts
        processed_signal = continuous_signal
        pad_length = 0
    else:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")
    
    # Use default boundaries if not specified
    if scale_boundaries is None:
        morph_end, syn_end = 4, 8
    else:
        morph_end, syn_end = scale_boundaries
    
    # Validate boundaries
    if not (1 <= morph_end < syn_end <= max_scales):
        raise ValueError(f"Invalid scale boundaries: morph_end={morph_end}, syn_end={syn_end}, max_scales={max_scales}")
    
    # Compute continuous wavelet transform on processed signal
    scales = np.arange(1, max_scales + 1)
    
    try:
        coeffs_full, freqs = pywt.cwt(processed_signal, scales, wavelet)
        
        # Extract coefficients corresponding to original signal region
        if pad_length > 0:
            coeffs = coeffs_full[:, pad_length:pad_length + original_length]
        else:
            coeffs = coeffs_full
            
    except Exception as e:
        # Fallback for any wavelet transform errors
        print(f"Wavelet transform error: {e}")
        return [0.0, 0.0, 0.0], {'boundary_mode': boundary_mode, 'signal_length': 0, 'edge_to_center_ratio': 1.0, 'edge_correlation': 0.0, 'left_edge_magnitude': 0.0, 'right_edge_magnitude': 0.0, 'avg_magnitude_profile': []}
    
    # Compute energy at each scale using different methods
    if energy_method == 'frobenius':
        # Original Frobenius norm (L2 norm of coefficient matrix)
        energy = np.sqrt(np.sum(np.abs(coeffs)**2, axis=1))
    elif energy_method == 'l2':
        # L2 norm of coefficients at each scale
        energy = np.linalg.norm(coeffs, axis=1)
    elif energy_method == 'shannon':
        # Shannon entropy of wavelet coefficients
        energy = []
        for scale_coeffs in coeffs:
            # Normalize coefficients to probabilities
            abs_coeffs = np.abs(scale_coeffs)
            if np.sum(abs_coeffs) == 0:
                energy.append(0.0)
            else:
                p = abs_coeffs / np.sum(abs_coeffs)
                # Add small epsilon to avoid log(0)
                p = p + 1e-12
                entropy = -np.sum(p * np.log2(p))
                energy.append(entropy)
        energy = np.array(energy)
    elif energy_method == 'wavelet_packet':
        # Wavelet packet energy using decomposition
        try:
            from scipy.signal import find_peaks
            energy = []
            for scale_coeffs in coeffs:
                # Use wavelet packet-like energy measure
                # Compute energy in frequency bands
                packet_energy = np.sum(np.abs(scale_coeffs)**2) / len(scale_coeffs)
                energy.append(packet_energy)
            energy = np.array(energy)
        except:
            # Fallback to L2 norm if wavelet packet fails
            energy = np.linalg.norm(coeffs, axis=1)
    elif energy_method == 'local_maxima':
        # Energy based on local maxima of coefficients
        try:
            from scipy.signal import find_peaks
            energy = []
            for scale_coeffs in coeffs:
                # Find local maxima
                peaks, _ = find_peaks(np.abs(scale_coeffs))
                if len(peaks) > 0:
                    # Energy as sum of local maxima values
                    maxima_energy = np.sum(np.abs(scale_coeffs[peaks]))
                else:
                    # Fallback to max value if no peaks found
                    maxima_energy = np.max(np.abs(scale_coeffs))
                energy.append(maxima_energy)
            energy = np.array(energy)
        except:
            # Fallback to standard method if peak detection fails
            energy = np.sqrt(np.sum(np.abs(coeffs)**2, axis=1))
    else:
        raise ValueError(f"Unknown energy method: {energy_method}")
    
    # Preserve signal strength while capturing multi-scale patterns
    signal_strength = np.std(continuous_signal)  # Overall signal variability
    
    # Extract weighted features at different linguistic scales with dynamic boundaries
    morph_energy = np.mean(energy[0:morph_end])             # Scales 1-morph_end: morphological
    syn_energy = np.mean(energy[morph_end:syn_end])         # Scales morph_end+1-syn_end: syntactic  
    disc_energy = np.mean(energy[syn_end:max_scales])       # Scales syn_end+1-max_scales: discourse
    
    # Weight by signal strength to preserve discriminative power
    # Use log transform to handle wide range of values
    eps = 1e-6
    features = [
        float(np.log(morph_energy * signal_strength + eps)),
        float(np.log(syn_energy * signal_strength + eps)), 
        float(np.log(disc_energy * signal_strength + eps))
    ]
    
    # Compute boundary effect analysis
    boundary_info = compute_boundary_effects(coeffs, original_length, boundary_mode)
    
    if return_coeffs:
        return features, boundary_info, coeffs
    else:
        return features, boundary_info


def compute_boundary_effects(coeffs, signal_length, boundary_mode):
    """
    Analyze boundary effects in wavelet coefficients.
    
    Args:
        coeffs: Wavelet coefficients matrix (scales x signal_length)
        signal_length: Length of original signal
        boundary_mode: Boundary handling strategy used
    
    Returns:
        boundary_info: Dict with boundary effect analysis
    """
    if coeffs.shape[1] != signal_length:
        # Handle case where coeffs might be different size
        signal_length = min(signal_length, coeffs.shape[1])
        coeffs = coeffs[:, :signal_length]
    
    # Compute mean coefficient magnitude at different positions
    position_indices = np.arange(signal_length)
    edge_region_size = min(signal_length // 10, 20)  # 10% of signal or max 20 points
    
    # Compute average coefficient magnitude across all scales
    avg_magnitude = np.mean(np.abs(coeffs), axis=0)
    
    # Analyze edge effects by comparing edge vs center regions
    if signal_length > 2 * edge_region_size:
        # Edge regions (beginning and end)
        left_edge = avg_magnitude[:edge_region_size]
        right_edge = avg_magnitude[-edge_region_size:]
        center_region = avg_magnitude[edge_region_size:-edge_region_size]
        
        # Compute edge effect metrics
        left_edge_mean = np.mean(left_edge)
        right_edge_mean = np.mean(right_edge)
        center_mean = np.mean(center_region)
        
        # Edge-to-center ratio
        edge_to_center_ratio = (left_edge_mean + right_edge_mean) / (2 * center_mean + 1e-8)
        
        # Correlation between coefficient magnitude and distance from edges
        distances_from_edges = np.minimum(position_indices, signal_length - 1 - position_indices)
        correlation_coeff = np.corrcoef(avg_magnitude, distances_from_edges)[0, 1]
    else:
        # For very short signals, compute simplified metrics
        edge_to_center_ratio = 1.0
        correlation_coeff = 0.0
        left_edge_mean = np.mean(avg_magnitude[:len(avg_magnitude)//2])
        right_edge_mean = np.mean(avg_magnitude[len(avg_magnitude)//2:])
    
    boundary_info = {
        'boundary_mode': boundary_mode,
        'signal_length': signal_length,
        'edge_to_center_ratio': float(edge_to_center_ratio),
        'edge_correlation': float(correlation_coeff) if not np.isnan(correlation_coeff) else 0.0,
        'left_edge_magnitude': float(left_edge_mean),
        'right_edge_magnitude': float(right_edge_mean),
        'avg_magnitude_profile': avg_magnitude.tolist()
    }
    
    return boundary_info


def generate_wavelet_heatmap(coeffs, tokens, text, save_path=None, title="Wavelet Coefficient Heatmap"):
    """
    Generate a heatmap visualization of wavelet coefficients aligned with text tokens.
    
    Args:
        coeffs: Wavelet coefficients matrix (scales x signal_length)
        tokens: List of token strings
        text: Original text string
        save_path: Path to save the heatmap image
        title: Title for the heatmap
        
    Returns:
        fig: Matplotlib figure object
        heatmap_data: Dictionary with heatmap data for further analysis
    """
    # Prepare data
    n_scales, signal_length = coeffs.shape
    
    # Normalize coefficients for visualization
    coeffs_norm = np.abs(coeffs)
    coeffs_norm = (coeffs_norm - np.min(coeffs_norm)) / (np.max(coeffs_norm) - np.min(coeffs_norm) + 1e-8)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(15, len(tokens)*0.5), 8), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main heatmap
    im = ax1.imshow(coeffs_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set scale labels (morphological, syntactic, discourse)
    scale_labels = []
    for i in range(n_scales):
        if i < 4:
            scale_labels.append(f'Morph-{i+1}')
        elif i < 8:
            scale_labels.append(f'Syn-{i+1}')
        else:
            scale_labels.append(f'Disc-{i+1}')
    
    ax1.set_yticks(range(n_scales))
    ax1.set_yticklabels(scale_labels)
    ax1.set_ylabel('Wavelet Scales')
    ax1.set_title(title)
    
    # Token positions (sample if too many tokens)
    if len(tokens) > 50:
        # Sample token positions for readability
        token_indices = np.linspace(0, len(tokens)-1, 50, dtype=int)
        display_tokens = [tokens[i] for i in token_indices]
        token_positions = np.linspace(0, signal_length-1, len(display_tokens), dtype=int)
    else:
        display_tokens = tokens
        token_positions = np.linspace(0, signal_length-1, len(tokens), dtype=int)
    
    ax1.set_xticks(token_positions)
    ax1.set_xticklabels(display_tokens, rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Normalized Coefficient Magnitude')
    
    # Token-level aggregated scores (mean across scales)
    token_scores = np.mean(coeffs_norm, axis=0)
    
    # Token score bar plot
    ax2.bar(range(signal_length), token_scores, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Signal Position')
    ax2.set_ylabel('Avg Coefficient')
    ax2.set_title('Token-level Aggregated Wavelet Activity')
    
    # Align x-axis with main heatmap
    ax2.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Prepare heatmap data for analysis
    heatmap_data = {
        'coefficients': coeffs.tolist(),
        'normalized_coefficients': coeffs_norm.tolist(),
        'token_scores': token_scores.tolist(),
        'tokens': tokens,
        'text': text,
        'scale_labels': scale_labels,
        'morphological_activity': np.mean(coeffs_norm[:4], axis=0).tolist(),
        'syntactic_activity': np.mean(coeffs_norm[4:8], axis=0).tolist() if n_scales >= 8 else [],
        'discourse_activity': np.mean(coeffs_norm[8:], axis=0).tolist() if n_scales > 8 else []
    }
    
    return fig, heatmap_data


def generate_baseline_saliency(logits, labels, tokens, method='gradient'):
    """
    Generate baseline saliency maps for comparison.
    
    Args:
        logits: Model logits (requires_grad=True)
        labels: Ground truth labels
        tokens: List of token strings
        method: Saliency method ('gradient', 'integrated_gradient')
        
    Returns:
        saliency_scores: Token-level saliency scores
    """
    if method == 'gradient':
        # Simple gradient-based saliency
        logits.retain_grad()
        
        # Compute loss
        lprobs = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.nll_loss(
            lprobs.view(-1, lprobs.size(-1)), 
            labels.view(-1), 
            reduction='mean'
        )
        
        # Backward pass to get gradients
        loss.backward(retain_graph=True)
        
        # Get gradients as saliency
        saliency = torch.abs(logits.grad).sum(dim=-1).squeeze().detach().cpu().numpy()
        
    else:
        # Fallback to simple attention-based saliency
        attention_weights = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        saliency = entropy.squeeze().detach().cpu().numpy()
    
    return saliency


def create_comparison_visualization(wavelet_data, saliency_data, tokens, text, save_path=None):
    """
    Create side-by-side comparison of wavelet and saliency visualizations.
    
    Args:
        wavelet_data: Wavelet heatmap data dictionary
        saliency_data: Saliency scores array
        tokens: List of token strings  
        text: Original text
        save_path: Path to save comparison image
        
    Returns:
        fig: Matplotlib figure object
        comparison_data: Dictionary with comparison metrics
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # Wavelet heatmap (top)
    coeffs_norm = np.array(wavelet_data['normalized_coefficients'])
    im1 = ax1.imshow(coeffs_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Wavelet Coefficient Heatmap')
    ax1.set_ylabel('Wavelet Scales')
    ax1.set_yticks(range(len(wavelet_data['scale_labels'])))
    ax1.set_yticklabels(wavelet_data['scale_labels'])
    
    # Token scores comparison (middle)
    positions = range(len(wavelet_data['token_scores']))
    
    ax2.bar(positions, wavelet_data['token_scores'], alpha=0.7, label='Wavelet Activity', color='orange')
    ax2_twin = ax2.twinx()
    
    # Ensure saliency data matches token scores length
    saliency_len = min(len(saliency_data), len(wavelet_data['token_scores']))
    saliency_positions = range(saliency_len)
    ax2_twin.bar([p+0.4 for p in saliency_positions], saliency_data[:saliency_len], alpha=0.7, label='Baseline Saliency', color='blue', width=0.4)
    
    ax2.set_title('Token-level Scores Comparison')
    ax2.set_ylabel('Wavelet Activity', color='orange')
    ax2_twin.set_ylabel('Saliency Score', color='blue')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Text with highlighting (bottom)
    ax3.text(0.02, 0.5, text[:200] + "..." if len(text) > 200 else text, 
             transform=ax3.transAxes, fontsize=10, verticalalignment='center', wrap=True)
    ax3.set_title('Original Text')
    ax3.axis('off')
    
    # Token positions for x-axis (sample if too many)
    if len(tokens) > 30:
        token_indices = np.linspace(0, len(tokens)-1, 30, dtype=int)
        display_tokens = [tokens[i] for i in token_indices]
        display_positions = [positions[i] for i in token_indices]
    else:
        display_tokens = tokens
        display_positions = positions
    
    ax1.set_xticks(display_positions)
    ax1.set_xticklabels(display_tokens, rotation=45, ha='right')
    ax2.set_xticks(display_positions)
    ax2.set_xticklabels(display_tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Calculate comparison metrics
    correlation = np.corrcoef(wavelet_data['token_scores'][:len(saliency_data)], saliency_data)[0, 1]
    
    comparison_data = {
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'wavelet_entropy': float(np.std(wavelet_data['token_scores'])),
        'saliency_entropy': float(np.std(saliency_data)),
        'max_wavelet_position': int(np.argmax(wavelet_data['token_scores'])),
        'max_saliency_position': int(np.argmax(saliency_data))
    }
    
    return fig, comparison_data


def get_t_discrepancy_analytic(logits_ref, logits_score, labels, nu=5, extract_wavelet_features=False, max_scales=12, scale_boundaries=None, bandwidth='scott', energy_method='frobenius', boundary_mode='zero', wavelet_type='cmor1.5-1.0', preserve_parameter_sensitivity=True, feature_expansion=None, n_features=3, random_seed=42, return_token_stats=False, return_visualization_data=False):
    """
    Compute discrepancy using heavy-tailed Student's t-distribution normalization.
    
    Args:
        logits_ref: Reference model logits
        logits_score: Scoring model logits  
        labels: Ground truth labels
        nu: Degrees of freedom for t-distribution (default 5 for heavy tails)
        extract_wavelet_features: Whether to extract wavelet features
        max_scales: Maximum number of wavelet scales to use
        scale_boundaries: Boundaries for wavelet scale grouping
        bandwidth: Bandwidth selection method for KDE
        feature_expansion: Type of feature expansion ('random_3d', 'random_12d', None)
        n_features: Number of features for random expansion
        random_seed: Random seed for reproducible feature expansion
        return_token_stats: Whether to return token-level statistics for MI calculation
        return_visualization_data: Whether to return data for visualization generation
    
    Returns:
        t_discrepancy: Discrepancy score using t-distribution normalization, wavelet features, or random expansion
        If return_token_stats=True, returns dict with additional token-level information
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    
    # Avoid division by zero and negative variances
    var_ref = torch.clamp(var_ref, min=1e-6)
    
    # Compute t-discrepancy with heavy-tailed normalization
    # Scale factor for Student's t-distribution: sqrt(nu/(nu-2) * variance)
    scale = torch.sqrt(var_ref * nu / (nu - 2))
    if extract_wavelet_features:
        # Compute token-level discrepancies for wavelet analysis
        # Don't normalize yet to preserve signal strength
        token_discrepancies_raw = (log_likelihood - mean_ref).squeeze(0)
        token_var = var_ref.squeeze(0)
        
        # Compute the original t-discrepancy for reference
        t_discrepancy_scalar = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / scale.sum(dim=-1)
        t_discrepancy_scalar = t_discrepancy_scalar.mean().item()
        
        # Normalize token discrepancies
        token_discrepancies = (token_discrepancies_raw / torch.sqrt(token_var)).detach().cpu().numpy()
        
        # Handle NaN/Inf values
        token_discrepancies = np.nan_to_num(token_discrepancies, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Transform to continuous signal with specified bandwidth
        continuous_signal = transform_discrete_sequence(token_discrepancies, bandwidth=bandwidth)
        
        # Extract wavelet features with boundary handling
        if return_visualization_data:
            wavelet_features, boundary_info, coeffs = get_wavelet_features(
                continuous_signal, 
                wavelet=wavelet_type,
                max_scales=max_scales,
                scale_boundaries=scale_boundaries, 
                energy_method=energy_method,
                boundary_mode=boundary_mode,
                return_coeffs=True
            )
        else:
            wavelet_features, boundary_info = get_wavelet_features(
                continuous_signal, 
                wavelet=wavelet_type,
                max_scales=max_scales,
                scale_boundaries=scale_boundaries, 
                energy_method=energy_method,
                boundary_mode=boundary_mode
            )
        
        # Scale features to preserve relative differences while maintaining reasonable magnitude
        # Use log-scale normalization to handle wide dynamic range
        # This preserves the discriminative power of different parameter configurations
        if preserve_parameter_sensitivity:
            # New scaling: preserve relative differences
            # Apply gentle scaling to keep features in reasonable range [-10, 10]
            max_abs_feature = max(abs(f) for f in wavelet_features)
            if max_abs_feature > 10:
                scale_factor = 10.0 / max_abs_feature
                wavelet_features = [f * scale_factor for f in wavelet_features]
        else:
            # Original scaling: match t-discrepancy magnitude (destroys parameter sensitivity)
            scale_factor = abs(t_discrepancy_scalar) / (np.mean(np.abs(wavelet_features)) + 1e-6)
            wavelet_features = [f * scale_factor for f in wavelet_features]
        
        # Return visualization data if requested
        if return_visualization_data:
            return {
                'features': wavelet_features,
                'boundary_info': boundary_info,
                'token_discrepancies': token_discrepancies.tolist(),
                'continuous_signal': continuous_signal.tolist(),
                'scalar_discrepancy': t_discrepancy_scalar,
                'wavelet_coeffs': coeffs.tolist() if 'coeffs' in locals() else None,
                'coeffs_matrix': coeffs.tolist() if 'coeffs' in locals() else None  # For visualization
            }
        
        # Return token-level statistics if requested for MI calculation
        if return_token_stats:
            return {
                'features': wavelet_features,
                'boundary_info': boundary_info,
                'token_discrepancies': token_discrepancies.tolist(),
                'continuous_signal': continuous_signal.tolist(),
                'scalar_discrepancy': t_discrepancy_scalar,
                'wavelet_coeffs': None  # Could add raw coefficients if needed
            }
        
        # Return features with boundary information for analysis
        return {'features': wavelet_features, 'boundary_info': boundary_info}
    
    elif feature_expansion is not None:
        # Random feature expansion variants for theoretical validation
        t_discrepancy_scalar = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / scale.sum(dim=-1)
        t_discrepancy_scalar = t_discrepancy_scalar.mean().item()
        
        # Create random projections of the scalar t-discrepancy
        np.random.seed(random_seed)
        
        if feature_expansion == 'random_3d':
            # 3D random projection to match wavelet feature dimensionality
            random_matrix = np.random.randn(3, 1)  # Project scalar to 3D
            base_features = np.array([t_discrepancy_scalar])
            expanded_features = (random_matrix @ base_features.reshape(1, 1)).flatten()
            
        elif feature_expansion == 'random_12d':
            # 12D random projection to test high-dimensional effects
            random_matrix = np.random.randn(12, 1)  # Project scalar to 12D
            base_features = np.array([t_discrepancy_scalar])
            expanded_features = (random_matrix @ base_features.reshape(1, 1)).flatten()
            
        else:
            raise ValueError(f"Unknown feature expansion type: {feature_expansion}")
        
        # Add noise for variation while preserving base signal
        noise_scale = np.abs(t_discrepancy_scalar) * 0.1  # 10% noise
        noise = np.random.normal(0, noise_scale, len(expanded_features))
        expanded_features = expanded_features + noise
        
        return expanded_features.tolist()
    
    # Original scalar computation
    t_discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / scale.sum(dim=-1)
    t_discrepancy = t_discrepancy.mean()
    
    # Return token-level statistics if requested for MI calculation
    if return_token_stats:
        token_discrepancies_raw = (log_likelihood - mean_ref).squeeze(0)
        token_var = var_ref.squeeze(0)
        token_discrepancies = (token_discrepancies_raw / torch.sqrt(token_var)).detach().cpu().numpy()
        token_discrepancies = np.nan_to_num(token_discrepancies, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return {
            'scalar_discrepancy': t_discrepancy.item(),
            'token_discrepancies': token_discrepancies.tolist()
        }
    
    return t_discrepancy.item()

def compute_perplexity(logits, labels):
    """
    Compute perplexity from logits and labels.
    
    Args:
        logits: Model logits
        labels: Ground truth labels
        
    Returns:
        perplexity: Perplexity value
    """
    lprobs = torch.log_softmax(logits, dim=-1)
    nll = torch.nn.functional.nll_loss(lprobs.view(-1, lprobs.size(-1)), labels.view(-1), reduction='none')
    perplexity = torch.exp(nll.mean())
    return perplexity.item()

class TDetect(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.criterion_fn = get_t_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(self.config.scoring_model_name, self.config.cache_dir)
        self.scoring_model = load_model(self.config.scoring_model_name, self.config.device, self.config.cache_dir)
        self.scoring_model.eval()
        if self.config.reference_model_name != self.config.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.config.reference_model_name, self.config.cache_dir)
            self.reference_model = load_model(self.config.reference_model_name, self.config.device, self.config.cache_dir)
            self.reference_model.eval()
        
        # Initialize parameters for dynamic threshold calibration
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.beta = getattr(self.config, 'beta', 0.1)
        self.ref_entropy = getattr(self.config, 'ref_entropy', 5.0)
        self.enable_dynamic_threshold = getattr(self.config, 'enable_dynamic_threshold', True)
        
        # Initialize wavelet feature extraction parameter
        self.extract_wavelet_features = getattr(self.config, 'extract_wavelet_features', False)
        
        # Initialize scale boundaries for wavelet features (morph_end, syn_end)
        self.scale_boundaries = getattr(self.config, 'scale_boundaries', None)
        
        # Initialize maximum number of scales
        self.max_scales = getattr(self.config, 'max_scales', 12)
        
        # Initialize bandwidth selection method
        self.bandwidth = getattr(self.config, 'bandwidth', 'scott')
        
        # Initialize energy calculation method
        self.energy_method = getattr(self.config, 'energy_method', 'frobenius')
        
        # Initialize boundary handling mode
        self.boundary_mode = getattr(self.config, 'boundary_mode', 'zero')
        
        # Initialize feature expansion parameters for theoretical validation
        self.feature_expansion = getattr(self.config, 'feature_expansion', None)
        self.n_features = getattr(self.config, 'n_features', 3)
        self.random_seed = getattr(self.config, 'random_seed', 42)
        
        # Store perplexity for dynamic thresholding
        self.last_perplexity = None
        
        # Store boundary effect analysis
        self.last_boundary_info = None

    def compute_crit(self, text, return_token_stats=False, return_visualization_data=False):
        tokenized = self.scoring_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                           return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
        labels = tokenized.input_ids[:, 1:]
        
        # Store tokens for visualization
        tokens = self.scoring_tokenizer.convert_ids_to_tokens(tokenized.input_ids[0, 1:])
        
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.config.reference_model_name == self.config.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                                        return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
                assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]
            
            # Compute t-discrepancy, wavelet features, or random feature expansion
            wavelet_type = getattr(self, 'wavelet_type', getattr(self.config, 'wavelet_type', 'cmor1.5-1.0'))
            crit = self.criterion_fn(logits_ref, logits_score, labels, 
                                    extract_wavelet_features=self.extract_wavelet_features,
                                    max_scales=self.max_scales,
                                    scale_boundaries=self.scale_boundaries,
                                    bandwidth=self.bandwidth,
                                    energy_method=self.energy_method,
                                    boundary_mode=self.boundary_mode,
                                    wavelet_type=wavelet_type,
                                    feature_expansion=self.feature_expansion,
                                    n_features=self.n_features,
                                    random_seed=self.random_seed,
                                    return_token_stats=return_token_stats,
                                    return_visualization_data=return_visualization_data)
            
            # Store perplexity for dynamic thresholding
            if self.enable_dynamic_threshold:
                self.last_perplexity = compute_perplexity(logits_score, labels)
            
            # Handle visualization data return
            if return_visualization_data and isinstance(crit, dict) and 'coeffs_matrix' in crit:
                # Add tokens and text for visualization
                crit['tokens'] = tokens
                crit['text'] = text
                crit['logits_score'] = logits_score.detach()  # For baseline saliency
                crit['labels'] = labels.detach()
                return crit
            
            # Handle different return formats from wavelet features
            if self.extract_wavelet_features and isinstance(crit, dict):
                # Store boundary info for analysis and return features
                self.last_boundary_info = crit.get('boundary_info', {})
                return crit['features']
            else:
                return crit
    
    def generate_visualization(self, text, save_path=None, include_baseline=True):
        """
        Generate visualization for a given text.
        
        Args:
            text: Input text to analyze
            save_path: Path to save visualization (optional)
            include_baseline: Whether to include baseline saliency comparison
            
        Returns:
            visualization_data: Dictionary with all visualization components
        """
        # Get visualization data
        viz_data = self.compute_crit(text, return_visualization_data=True)
        
        if not isinstance(viz_data, dict) or 'coeffs_matrix' not in viz_data:
            raise ValueError("Visualization data not available. Ensure extract_wavelet_features=True.")
        
        # Generate wavelet heatmap
        coeffs_matrix = np.array(viz_data['coeffs_matrix'])
        tokens = viz_data['tokens']
        
        wavelet_fig, heatmap_data = generate_wavelet_heatmap(
            coeffs_matrix, tokens, text,
            save_path=save_path.replace('.png', '_wavelet.png') if save_path else None,
            title=f"Wavelet Analysis - T-Detect"
        )
        
        visualization_data = {
            'wavelet_figure': wavelet_fig,
            'heatmap_data': heatmap_data,
            'visualization_metadata': viz_data
        }
        
        # Generate baseline comparison if requested
        if include_baseline:
            try:
                # Enable gradients for saliency computation
                logits_score = viz_data['logits_score'].requires_grad_(True)
                labels = viz_data['labels']
                
                saliency_scores = generate_baseline_saliency(logits_score, labels, tokens)
                
                comparison_fig, comparison_data = create_comparison_visualization(
                    heatmap_data, saliency_scores, tokens, text,
                    save_path=save_path.replace('.png', '_comparison.png') if save_path else None
                )
                
                visualization_data.update({
                    'comparison_figure': comparison_fig,
                    'comparison_data': comparison_data,
                    'baseline_saliency': saliency_scores.tolist()
                })
                
            except Exception as e:
                print(f"Warning: Could not generate baseline comparison: {e}")
        
        return visualization_data
    
    def get_dynamic_threshold(self, base_threshold):
        """
        Compute dynamic threshold based on perplexity.
        
        Args:
            base_threshold: Base threshold from training
            
        Returns:
            dynamic_threshold: Perplexity-adjusted threshold
        """
        if not self.enable_dynamic_threshold or self.last_perplexity is None:
            return base_threshold
            
        # Convert perplexity to entropy: H(p) = log(perplexity)
        entropy = np.log(self.last_perplexity)
        
        # Dynamic threshold scaling: threshold = alpha * exp(beta * (entropy - ref_entropy))
        scale_factor = self.alpha * np.exp(self.beta * (entropy - self.ref_entropy))
        dynamic_threshold = base_threshold * scale_factor
        
        return dynamic_threshold

    def detect_boundaries_in_signal(self, continuous_signal, coeffs_matrix, detection_method='energy_gradient', threshold_percentile=75):
        """
        Detect boundaries in continuous signal using wavelet analysis.
        
        Args:
            continuous_signal: Continuous signal from KDE transformation
            coeffs_matrix: Wavelet coefficients matrix (scales x signal_length)
            detection_method: Method for boundary detection ('energy_gradient', 'wavelet_modulus', 'multiscale_energy')
            threshold_percentile: Percentile threshold for boundary detection
            
        Returns:
            boundary_positions: List of detected boundary positions
            boundary_strengths: List of boundary strength values
            detection_metadata: Dictionary with detection method metadata
        """
        if coeffs_matrix is None or len(continuous_signal) == 0:
            return [], [], {'method': detection_method, 'threshold_percentile': threshold_percentile}
        
        try:
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d
            
            boundary_positions = []
            boundary_strengths = []
            
            if detection_method == 'energy_gradient':
                # Method 1: Energy gradient across scales
                # Compute total energy at each position
                total_energy = np.sum(np.abs(coeffs_matrix)**2, axis=0)
                
                # Smooth the energy signal to reduce noise
                smoothed_energy = gaussian_filter1d(total_energy, sigma=2.0)
                
                # Compute gradient to find rapid changes
                energy_gradient = np.abs(np.gradient(smoothed_energy))
                
                # Find peaks in gradient (high change points)
                threshold = np.percentile(energy_gradient, threshold_percentile)
                peaks, properties = find_peaks(energy_gradient, height=threshold, distance=5)
                
                boundary_positions = peaks.tolist()
                boundary_strengths = energy_gradient[peaks].tolist()
                
            elif detection_method == 'wavelet_modulus':
                # Method 2: Wavelet modulus maxima
                # Focus on mid-range scales (syntactic level)
                mid_scales = coeffs_matrix[4:8, :]  # Syntactic scales
                modulus = np.sqrt(np.sum(mid_scales**2, axis=0))
                
                # Find local maxima in modulus
                threshold = np.percentile(modulus, threshold_percentile)
                peaks, properties = find_peaks(modulus, height=threshold, distance=3)
                
                boundary_positions = peaks.tolist()
                boundary_strengths = modulus[peaks].tolist()
                
            elif detection_method == 'multiscale_energy':
                # Method 3: Multi-scale energy analysis
                # Analyze energy changes across different scales
                n_scales, signal_length = coeffs_matrix.shape
                
                # Compute energy at each scale
                scale_energies = np.sqrt(np.sum(np.abs(coeffs_matrix)**2, axis=1))
                
                # Find positions where multiple scales show high activity
                multiscale_activity = np.zeros(signal_length)
                
                for i in range(n_scales):
                    scale_coeff = np.abs(coeffs_matrix[i, :])
                    scale_threshold = np.percentile(scale_coeff, threshold_percentile)
                    high_activity = scale_coeff > scale_threshold
                    multiscale_activity += high_activity.astype(float)
                
                # Find positions with high multi-scale activity
                activity_threshold = n_scales * 0.3  # At least 30% of scales active
                boundary_candidates = np.where(multiscale_activity >= activity_threshold)[0]
                
                if len(boundary_candidates) > 0:
                    # Cluster nearby candidates and take centroids
                    clusters = []
                    current_cluster = [boundary_candidates[0]]
                    
                    for pos in boundary_candidates[1:]:
                        if pos - current_cluster[-1] <= 5:  # Within 5 positions
                            current_cluster.append(pos)
                        else:
                            clusters.append(current_cluster)
                            current_cluster = [pos]
                    clusters.append(current_cluster)
                    
                    # Take centroid of each cluster
                    for cluster in clusters:
                        centroid = int(np.mean(cluster))
                        strength = multiscale_activity[centroid]
                        boundary_positions.append(centroid)
                        boundary_strengths.append(float(strength))
                
            else:
                raise ValueError(f"Unknown boundary detection method: {detection_method}")
            
            detection_metadata = {
                'method': detection_method,
                'threshold_percentile': threshold_percentile,
                'total_boundaries_detected': len(boundary_positions),
                'signal_length': len(continuous_signal),
                'scales_analyzed': coeffs_matrix.shape[0] if coeffs_matrix is not None else 0
            }
            
            return boundary_positions, boundary_strengths, detection_metadata
            
        except Exception as e:
            print(f"Error in boundary detection: {e}")
            return [], [], {'method': detection_method, 'error': str(e)}
    
    def compute_boundary_preservation_quality(self, detected_boundaries, true_boundaries, signal_length, tolerance=5):
        """
        Compute quality metrics for boundary preservation.
        
        Args:
            detected_boundaries: List of detected boundary positions
            true_boundaries: List of true boundary positions
            signal_length: Length of the analyzed signal
            tolerance: Tolerance for matching boundaries (in positions)
            
        Returns:
            quality_metrics: Dictionary with boundary preservation quality metrics
        """
        if len(true_boundaries) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'localization_error': 0.0,
                'detection_rate': 0.0,
                'false_positive_rate': 0.0,
                'mean_distance_error': 0.0,
                'boundary_correlation': 0.0
            }
        
        # Convert to numpy arrays for easier computation
        detected = np.array(detected_boundaries) if detected_boundaries else np.array([])
        true = np.array(true_boundaries)
        
        # Find matches within tolerance
        matches = []
        matched_true = set()
        matched_detected = set()
        distance_errors = []
        
        for i, det_pos in enumerate(detected):
            best_match = None
            best_distance = float('inf')
            
            for j, true_pos in enumerate(true):
                if j not in matched_true:
                    distance = abs(det_pos - true_pos)
                    if distance <= tolerance and distance < best_distance:
                        best_match = j
                        best_distance = distance
            
            if best_match is not None:
                matches.append((i, best_match))
                matched_detected.add(i)
                matched_true.add(best_match)
                distance_errors.append(best_distance)
        
        # Compute metrics
        true_positives = len(matches)
        false_positives = len(detected) - true_positives
        false_negatives = len(true) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        detection_rate = true_positives / len(true) if len(true) > 0 else 0.0
        false_positive_rate = false_positives / signal_length if signal_length > 0 else 0.0
        
        mean_distance_error = np.mean(distance_errors) if distance_errors else 0.0
        
        # Compute boundary correlation (spatial correlation between detected and true boundaries)
        boundary_correlation = 0.0
        if len(detected) > 0 and len(true) > 0:
            # Create binary vectors for detected and true boundaries
            detected_vector = np.zeros(signal_length)
            true_vector = np.zeros(signal_length)
            
            for pos in detected:
                if 0 <= pos < signal_length:
                    detected_vector[pos] = 1
            
            for pos in true:
                if 0 <= pos < signal_length:
                    true_vector[pos] = 1
            
            # Compute correlation
            if np.sum(detected_vector) > 0 and np.sum(true_vector) > 0:
                correlation_coeff = np.corrcoef(detected_vector, true_vector)[0, 1]
                boundary_correlation = correlation_coeff if not np.isnan(correlation_coeff) else 0.0
        
        quality_metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'localization_error': float(mean_distance_error),
            'detection_rate': float(detection_rate),
            'false_positive_rate': float(false_positive_rate),
            'mean_distance_error': float(mean_distance_error),
            'boundary_correlation': float(boundary_correlation),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'total_detected': len(detected),
            'total_true': len(true)
        }
        
        return quality_metrics
    
    def analyze_boundary_preservation(self, text, true_boundaries=None, detection_method='energy_gradient', threshold_percentile=75, tolerance=5):
        """
        Comprehensive boundary preservation analysis for a given text.
        
        Args:
            text: Input text to analyze
            true_boundaries: List of true boundary positions (token indices)
            detection_method: Method for boundary detection
            threshold_percentile: Percentile threshold for detection
            tolerance: Tolerance for boundary matching
            
        Returns:
            analysis_results: Dictionary with comprehensive boundary analysis results
        """
        # Get wavelet analysis data
        viz_data = self.compute_crit(text, return_visualization_data=True)
        
        if not isinstance(viz_data, dict) or 'coeffs_matrix' not in viz_data:
            return {'error': 'Wavelet features not available. Ensure extract_wavelet_features=True.'}
        
        # Extract analysis components
        continuous_signal = np.array(viz_data['continuous_signal'])
        coeffs_matrix = np.array(viz_data['coeffs_matrix'])
        tokens = viz_data['tokens']
        
        # Map true boundaries from token indices to signal positions
        mapped_true_boundaries = []
        if true_boundaries:
            signal_length = len(continuous_signal)
            token_count = len(tokens)
            
            for boundary_token_idx in true_boundaries:
                # Map token index to signal position
                if token_count > 0:
                    signal_pos = int(boundary_token_idx * signal_length / token_count)
                    signal_pos = max(0, min(signal_pos, signal_length - 1))
                    mapped_true_boundaries.append(signal_pos)
        
        # Detect boundaries using wavelet analysis
        detected_positions, detected_strengths, detection_metadata = self.detect_boundaries_in_signal(
            continuous_signal, coeffs_matrix, detection_method, threshold_percentile
        )
        
        # Compute boundary preservation quality
        quality_metrics = self.compute_boundary_preservation_quality(
            detected_positions, mapped_true_boundaries, len(continuous_signal), tolerance
        )
        
        # Generate scalar baseline for comparison
        scalar_discrepancy = viz_data.get('scalar_discrepancy', 0.0)
        
        # Create comprehensive analysis results
        analysis_results = {
            'wavelet_analysis': {
                'detected_boundaries': detected_positions,
                'boundary_strengths': detected_strengths,
                'detection_metadata': detection_metadata,
                'quality_metrics': quality_metrics
            },
            'scalar_baseline': {
                'scalar_score': scalar_discrepancy,
                'boundary_detection': 'not_applicable'  # Scalar methods don't detect boundaries
            },
            'text_metadata': {
                'text_length': len(text),
                'token_count': len(tokens),
                'signal_length': len(continuous_signal),
                'true_boundaries': true_boundaries,
                'mapped_boundaries': mapped_true_boundaries
            },
            'wavelet_features': viz_data.get('features', []),
            'boundary_info': viz_data.get('boundary_info', {}),
            'visualization_data': viz_data
        }
        
        return analysis_results