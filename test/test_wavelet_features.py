#!/usr/bin/env python3
"""
Unit tests for wavelet-based feature extraction in t_detect.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
from scripts.detectors.t_detect import (
    transform_discrete_sequence, 
    get_wavelet_features,
    get_t_discrepancy_analytic
)


class TestWaveletFeatures(unittest.TestCase):
    """Test suite for wavelet feature extraction functionality"""
    
    def test_transform_discrete_sequence_empty(self):
        """Test handling of empty sequence"""
        result = transform_discrete_sequence(np.array([]))
        self.assertEqual(len(result), 1000)
        self.assertTrue(np.all(result == 0))
    
    def test_transform_discrete_sequence_single(self):
        """Test handling of single value sequence"""
        result = transform_discrete_sequence(np.array([5.0]))
        self.assertEqual(len(result), 1000)
        # Should create a Gaussian-like peak
        self.assertGreater(np.max(result), 0)
        self.assertAlmostEqual(np.sum(result > 0.1 * np.max(result)), 100, delta=50)
    
    def test_transform_discrete_sequence_normal(self):
        """Test normal sequence transformation"""
        sequence = np.random.randn(100)
        result = transform_discrete_sequence(sequence)
        self.assertEqual(len(result), 1000)
        # Should be smooth and continuous
        self.assertTrue(np.all(np.isfinite(result)))
        
    def test_transform_discrete_sequence_extreme(self):
        """Test handling of extreme values"""
        sequence = np.array([1e10, -1e10, 0, 1e-10])
        result = transform_discrete_sequence(sequence)
        self.assertEqual(len(result), 1000)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_get_wavelet_features_zero(self):
        """Test wavelet features on zero signal"""
        signal = np.zeros(1000)
        features = get_wavelet_features(signal)
        self.assertEqual(len(features), 3)
        self.assertEqual(features, [0.0, 0.0, 0.0])
    
    def test_get_wavelet_features_constant(self):
        """Test wavelet features on constant signal"""
        signal = np.ones(1000) * 5.0
        features = get_wavelet_features(signal)
        self.assertEqual(len(features), 3)
        # Constant signal should have most energy at low frequencies
        self.assertGreater(features[0], 0.5)
    
    def test_get_wavelet_features_sine(self):
        """Test wavelet features on sine wave"""
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * t)
        features = get_wavelet_features(signal)
        self.assertEqual(len(features), 3)
        # Features should sum to approximately 1 (normalized)
        self.assertAlmostEqual(sum(features), 1.0, delta=0.1)
    
    def test_get_wavelet_features_noise(self):
        """Test wavelet features on random noise"""
        signal = np.random.randn(1000)
        features = get_wavelet_features(signal)
        self.assertEqual(len(features), 3)
        # Noise should have energy distributed across scales
        self.assertTrue(all(f > 0.1 for f in features))
        self.assertTrue(all(f < 0.9 for f in features))
    
    def test_get_wavelet_features_edge_cases(self):
        """Test wavelet features edge cases"""
        # Very small signal
        signal = np.random.randn(1000) * 1e-15
        features = get_wavelet_features(signal)
        self.assertEqual(features, [0.0, 0.0, 0.0])
        
        # Signal with NaN
        signal = np.ones(1000)
        signal[500] = np.nan
        features = get_wavelet_features(signal)
        self.assertEqual(len(features), 3)
    
    def test_t_discrepancy_wavelet_mode(self):
        """Test t_discrepancy computation in wavelet mode"""
        # Create dummy logits
        batch_size = 1
        seq_len = 50
        vocab_size = 100
        
        logits_ref = torch.randn(batch_size, seq_len, vocab_size)
        logits_score = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test wavelet mode
        result = get_t_discrepancy_analytic(
            logits_ref, logits_score, labels, 
            extract_wavelet_features=True
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(f, float) for f in result))
        self.assertTrue(all(0 <= f <= 1 for f in result))
    
    def test_t_discrepancy_scalar_mode(self):
        """Test t_discrepancy computation in scalar mode"""
        # Create dummy logits
        batch_size = 1
        seq_len = 50
        vocab_size = 100
        
        logits_ref = torch.randn(batch_size, seq_len, vocab_size)
        logits_score = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test scalar mode
        result = get_t_discrepancy_analytic(
            logits_ref, logits_score, labels, 
            extract_wavelet_features=False
        )
        
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        # Very large discrepancies
        large_seq = np.array([1e6, -1e6, 1e6, -1e6])
        signal = transform_discrete_sequence(large_seq)
        features = get_wavelet_features(signal)
        
        self.assertTrue(all(np.isfinite(f) for f in features))
        self.assertTrue(all(0 <= f <= 1 for f in features))
        
        # Very small discrepancies
        small_seq = np.array([1e-10, -1e-10, 1e-10, -1e-10])
        signal = transform_discrete_sequence(small_seq)
        features = get_wavelet_features(signal)
        
        self.assertTrue(all(np.isfinite(f) for f in features))
    
    def test_reproducibility(self):
        """Test that results are reproducible"""
        sequence = np.random.randn(100)
        
        # Run transformation twice
        signal1 = transform_discrete_sequence(sequence)
        signal2 = transform_discrete_sequence(sequence)
        
        np.testing.assert_array_almost_equal(signal1, signal2)
        
        # Run wavelet extraction twice
        features1 = get_wavelet_features(signal1)
        features2 = get_wavelet_features(signal2)
        
        self.assertEqual(features1, features2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full wavelet pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from token discrepancies to features"""
        # Simulate token-level discrepancies
        token_discrepancies = np.random.randn(100) * 2.0
        
        # Transform to continuous signal
        continuous_signal = transform_discrete_sequence(token_discrepancies)
        
        # Extract wavelet features
        features = get_wavelet_features(continuous_signal)
        
        # Validate results
        self.assertEqual(len(features), 3)
        self.assertAlmostEqual(sum(features), 1.0, delta=0.1)
        self.assertTrue(all(0 <= f <= 1 for f in features))
    
    def test_different_text_lengths(self):
        """Test handling of different text lengths"""
        for length in [10, 50, 100, 500, 1000]:
            token_discrepancies = np.random.randn(length)
            signal = transform_discrete_sequence(token_discrepancies)
            features = get_wavelet_features(signal)
            
            self.assertEqual(len(signal), 1000)
            self.assertEqual(len(features), 3)
            self.assertAlmostEqual(sum(features), 1.0, delta=0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)