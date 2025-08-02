# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import pywt
from scipy.stats import gaussian_kde
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def transform_discrete_sequence(discrepancy_sequence, bandwidth='scott'):
    """
    Transform discrete token-level discrepancies to continuous signal.
    Uses Gaussian KDE with optimal bandwidth selection.
    
    Args:
        discrepancy_sequence: Numpy array of token-level discrepancies
        bandwidth: Bandwidth selection method for KDE
    
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
    
    # Apply kernel density estimation
    kde = gaussian_kde(discrepancy_sequence, bw_method=bandwidth)
    
    # Create continuous representation
    t = np.linspace(0, len(discrepancy_sequence), 1000)
    continuous_signal = kde(t)
    
    return continuous_signal


def get_wavelet_features(continuous_signal, wavelet='cmor1.5-1.0', max_scales=12):
    """
    Extract multi-scale wavelet features from continuous signal.
    Returns weighted energy at morphological, syntactic, and discourse scales.
    
    Args:
        continuous_signal: Continuous signal from KDE
        wavelet: Wavelet function to use
        max_scales: Maximum number of scales
    
    Returns:
        features: List of [morphological, syntactic, discourse] weighted energies
    """
    # Handle edge cases
    if np.all(continuous_signal == 0) or len(continuous_signal) == 0:
        return [0.0, 0.0, 0.0]
    
    # Compute continuous wavelet transform
    scales = np.arange(1, max_scales + 1)
    
    try:
        coeffs, freqs = pywt.cwt(continuous_signal, scales, wavelet)
    except Exception as e:
        # Fallback for any wavelet transform errors
        print(f"Wavelet transform error: {e}")
        return [0.0, 0.0, 0.0]
    
    # Compute energy at each scale (L2 norm)
    energy = np.sqrt(np.sum(np.abs(coeffs)**2, axis=1))
    
    # Preserve signal strength while capturing multi-scale patterns
    signal_strength = np.std(continuous_signal)  # Overall signal variability
    
    # Extract weighted features at different linguistic scales
    morph_energy = np.mean(energy[0:4])    # Scales 1-4: morphological
    syn_energy = np.mean(energy[4:8])      # Scales 5-8: syntactic  
    disc_energy = np.mean(energy[8:12])    # Scales 9-12: discourse
    
    # Weight by signal strength to preserve discriminative power
    # Use log transform to handle wide range of values
    eps = 1e-6
    features = [
        float(np.log(morph_energy * signal_strength + eps)),
        float(np.log(syn_energy * signal_strength + eps)), 
        float(np.log(disc_energy * signal_strength + eps))
    ]
    
    return features


def get_t_discrepancy_analytic(logits_ref, logits_score, labels, nu=5, extract_wavelet_features=False, return_details=False):
    """
    Compute discrepancy using heavy-tailed Student's t-distribution normalization.
    
    Args:
        logits_ref: Reference model logits
        logits_score: Scoring model logits  
        labels: Ground truth labels
        nu: Degrees of freedom for t-distribution (default 5 for heavy tails)
        extract_wavelet_features: Whether to extract wavelet features
        return_details: Whether to return internal details for visualization
    
    Returns:
        t_discrepancy: Discrepancy score using t-distribution normalization
        details (optional): Dictionary with internal features if return_details=True
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
    
    # Always compute token-level features for potential visualization
    token_discrepancies_raw = (log_likelihood - mean_ref).squeeze(0)
    token_var = var_ref.squeeze(0)
    
    # Compute the original t-discrepancy
    t_discrepancy_scalar = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / scale.sum(dim=-1)
    t_discrepancy_scalar = t_discrepancy_scalar.mean().item()
    
    # Normalize token discrepancies
    token_discrepancies = (token_discrepancies_raw / torch.sqrt(token_var)).detach().cpu().numpy()
    
    # Handle NaN/Inf values
    token_discrepancies = np.nan_to_num(token_discrepancies, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Transform to continuous signal
    continuous_signal = transform_discrete_sequence(token_discrepancies)
    
    # Compute wavelet transform for visualization
    scales = np.arange(1, 13)
    wavelet_coeffs, _ = pywt.cwt(continuous_signal, scales, 'cmor1.5-1.0')
    
    if extract_wavelet_features:
        # Extract wavelet features
        wavelet_features = get_wavelet_features(continuous_signal)
        
        # Scale features to match the magnitude of original t-discrepancy
        # This preserves the discriminative range
        scale_factor = abs(t_discrepancy_scalar) / (np.mean(np.abs(wavelet_features)) + 1e-6)
        wavelet_features = [f * scale_factor for f in wavelet_features]
        
        if return_details:
            details = {
                'token_discrepancies': token_discrepancies,
                'continuous_signal': continuous_signal,
                'wavelet_coeffs': wavelet_coeffs,
                'wavelet_features': wavelet_features,
                't_discrepancy_scalar': t_discrepancy_scalar
            }
            return wavelet_features, details
        
        return wavelet_features
    
    # Original scalar computation
    if return_details:
        details = {
            'token_discrepancies': token_discrepancies,
            'continuous_signal': continuous_signal,
            'wavelet_coeffs': wavelet_coeffs,
            'wavelet_features': [t_discrepancy_scalar, t_discrepancy_scalar, t_discrepancy_scalar],
            't_discrepancy_scalar': t_discrepancy_scalar
        }
        return t_discrepancy_scalar, details
    
    return t_discrepancy_scalar

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

class TDT(DetectorBase):
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
        
        # Store perplexity for dynamic thresholding
        self.last_perplexity = None

    def compute_crit(self, text, return_details=False):
        tokenized = self.scoring_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                           return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
        labels = tokenized.input_ids[:, 1:]
        
        # Store tokens for visualization
        self.last_tokens = self.scoring_tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
        
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.config.reference_model_name == self.config.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                                     return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            
            # Compute t-discrepancy or wavelet features
            result = self.criterion_fn(logits_ref, logits_score, labels, 
                                    extract_wavelet_features=self.extract_wavelet_features,
                                    return_details=return_details)
            
            # Store perplexity for dynamic thresholding
            if self.enable_dynamic_threshold:
                self.last_perplexity = compute_perplexity(logits_score, labels)
            
        return result
    
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