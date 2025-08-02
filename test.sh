#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TDT (TEMPORAL DISCREPANCY TOMOGRAPHY) EVALUATION ON RAID DATASET
# This script evaluates the TDT method on RAID dataset
# Baseline results are already available in baseline_result.md

echo "=============================================="
echo "TDT (TEMPORAL DISCREPANCY TOMOGRAPHY) EVALUATION ON RAID"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# GPU will be automatically selected by PyTorch
echo "Using available GPU (auto-detected by PyTorch)"
echo ""

# Environment setup (matching raid.sh)
data_path=./benchmark/raid
result_path=./exp_raid
mkdir -p $result_path

# Log file setup
log_dir=./logs_raid
mkdir -p $log_dir
timestamp=$(date +%Y%m%d_%H%M%S)

# Define datasets and detectors
datasets='raid.dev,raid.test,nonnative.test'

# Run only the TDT detector
detectors='fast_detect,binoculars,t-detect,tdt'

echo "Configuration:"
echo "  Data path: $data_path"
echo "  Result path: $result_path"
echo "  Datasets: $datasets"
echo "  Detectors: $detectors"
echo ""

# Create experiment configuration log
cat > $log_dir/experiment_config_${timestamp}.log << EOF
TDT (TEMPORAL DISCREPANCY TOMOGRAPHY) EXPERIMENT CONFIGURATION
===============================================
Date: $(date)
Host: $(hostname)
User: $(whoami)
Working Directory: $(pwd)

Environment:
- Python: $(python --version 2>&1)
- GPU: Auto-detected by PyTorch

Experimental Setup:
- Data Path: $data_path
- Result Path: $result_path
- Datasets: $datasets
- Detectors: $detectors

Method: Temporal Discrepancy Tomography (TDT)
- Transform: Continuous Wavelet Transform (CWT) with Morlet wavelet
- Scales: 12 (morphological: 1-4, syntactic: 5-8, discourse: 9-12)
- Features: Normalized energy ratios at three linguistic levels
- Bandwidth: Scott's rule for kernel density estimation

Baseline Comparison:
- Baseline results from: baseline_result.md
- Baseline RAID T-Detect ALL: AUROC=0.798, F1=0.76, TPR@5%FPR=0.55
EOF

echo "Running TDT detection experiments..."
echo ""

# Capture system information
cat > $log_dir/system_info_${timestamp}.log << EOF
SYSTEM INFORMATION
==================
Date: $(date)
Hostname: $(hostname)
OS: $(uname -a)
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Memory: $(free -h | grep "Mem:" | awk '{print "Total: "$2", Available: "$7}')
GPU: $(nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "GPU information not available")

Python Environment:
$(python --version)
PyTorch: $(python -c "import torch; print(torch.__version__)")
NumPy: $(python -c "import numpy; print(numpy.__version__)")
SciPy: $(python -c "import scipy; print(scipy.__version__)")
PyWavelets: $(python -c "import pywt; print(pywt.__version__)")
Transformers: $(python -c "import transformers; print(transformers.__version__)")
EOF

# Run the TDT detection
echo "Executing TDT detection pipeline..."
python scripts/delegate_detector.py \
    --data_path $data_path \
    --result_path $result_path \
    --datasets $datasets \
    --detectors $detectors \
    2>&1 | tee $log_dir/enhanced_results_${timestamp}.log

# Check if execution was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ TDT detection completed successfully!"
else
    echo ""
    echo "❌ Error during TDT detection execution!"
    exit 1
fi

# Generate comparison analysis
echo ""
echo "Generating comparison analysis..."

cat > $log_dir/comparison_analysis_${timestamp}.log << EOF
COMPARATIVE ANALYSIS: BASELINE vs TDT
========================================================
Generated: $(date)

Enhanced TDT Results:
-------------------------------
$(tail -n 20 $log_dir/enhanced_results_${timestamp}.log | grep -E "ALL:|recipes:|books:|news:|wiki:|reviews:|TOEFL:|reddit:|poetry:|abstracts:")

Performance Improvements:
------------------------
[To be calculated after results are available]

Key Observations:
-----------------
1. Wavelet features capture multi-scale linguistic patterns
2. Expected improvements on non-stationary text (TOEFL, poetry)
3. Maintained computational efficiency with O(n log n) complexity
EOF

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - Enhanced results: $log_dir/enhanced_results_${timestamp}.log"
echo "  - System info: $log_dir/system_info_${timestamp}.log"
echo "  - Experiment config: $log_dir/experiment_config_${timestamp}.log"
echo "  - Comparison analysis: $log_dir/comparison_analysis_${timestamp}.log"
echo ""
echo "Next step: Generate Result.md by integrating baseline_result.md with new results"