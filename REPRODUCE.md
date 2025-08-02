# Reproduction Guide for TDT

This guide provides detailed instructions for reproducing the results from our paper "AI-Generated Text is Non-Stationary: Detection via Temporal Tomography".

## Prerequisites

### Hardware Requirements
- GPU with at least 16GB VRAM (tested on NVIDIA H100 80GB)
- 32GB+ system RAM recommended
- CUDA 11.0+ compatible GPU

### Software Requirements
- Python 3.12 or higher
- CUDA 11.0 or higher
- PyTorch 2.0+

## Environment Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd impl_IDEA-dsvf
```

2. **Create virtual environment**
```bash
python -m venv tdt_env
source tdt_env/bin/activate  # On Windows: tdt_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Model Setup

### Download Required Models

TDT requires two Falcon models. You need to either:

1. **Download from HuggingFace** (recommended):
```bash
# Create model directory
mkdir -p /path/to/models

# Download models using HuggingFace CLI
pip install huggingface-hub
huggingface-cli download tiiuae/falcon-7b --local-dir /path/to/models/falcon-7b
huggingface-cli download tiiuae/falcon-7b-instruct --local-dir /path/to/models/falcon-7b-instruct
```

2. **Update configuration files** to point to your model paths:
```bash
# Edit the model paths in the config
vi scripts/detectors/configs/tdt.json
```

Update the paths:
```json
{
    "reference_model_name": "/path/to/models/falcon-7b",
    "scoring_model_name": "/path/to/models/falcon-7b-instruct",
    ...
}
```

## Reproducing Main Results

### 1. RAID Benchmark Results (Table 1 in paper)

```bash
# Run TDT on RAID dataset
./test.sh

# Expected output location
# Results will be saved in: exp_raid_wavelet/
# Logs will be saved in: logs_wavelet/
```

Expected results:
- Overall AUROC: 0.855 (±0.01)
- TPR@5%FPR: 0.575 (±0.02)

### 2. HART Benchmark Results (Tables 2-3 in paper)

First, fix the main.sh script (currently misconfigured):
```bash
# Edit main.sh to use HART dataset
vi main.sh
# Change data_path to ./benchmark/hart
# Change detectors to include tdt variants
```

Then run:
```bash
./main.sh

# Results will be saved in: exp_main/
```

Expected results:
- Level 1 AUROC: 0.825 (±0.01)
- Level 2 AUROC: 0.812 (±0.01) 
- Level 3 AUROC: 0.891 (±0.01)

### 3. Multilingual Results (Table 4 in paper)

```bash
./langs.sh

# Results will be saved in: exp_langs/
```

### 4. Running Individual Experiments

For specific dataset/detector combinations:
```bash
# Example: TDT on essay dataset
python scripts/delegate_detector.py \
    --data_path ./benchmark/hart \
    --result_path ./exp_custom \
    --datasets essay.dev,essay.test \
    --detectors tdt,C(tdt),CT(tdt)
```

## Understanding the Results

### Result File Format
Results are saved as JSON files with naming convention: `[dataset].[split].[detector].json`

Each file contains:
- `generation_crit`: Detection scores for original text
- `content_crit`: Detection scores for content dimension
- `generation_wave1/2/3`: TDT multi-scale features

### Metrics Computation
The system reports three metrics:
- **AUROC**: Area Under ROC Curve
- **F1**: F1 score at optimal threshold
- **TPR@5%FPR**: True Positive Rate at 5% False Positive Rate

### Log Files
Detailed logs include:
- `experiment_config_*.log`: Configuration and parameters
- `enhanced_results_*.log`: Full experimental results
- `system_info_*.log`: Hardware and software details
- `comparison_analysis_*.log`: Performance comparisons

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in detector configs
   - Use smaller max_token_observed value

2. **Model loading errors**
   - Verify model paths in config files
   - Ensure models are fully downloaded
   - Check file permissions

3. **Wavelet import errors**
   - Ensure PyWavelets is installed: `pip install PyWavelets>=1.4.0`

4. **Performance variations**
   - Set random seed for reproducibility: `--seed 42`
   - Use same GPU model as paper (H100 recommended)

## Ablation Studies

To reproduce ablation results:

```bash
# Vary wavelet scales (default: 12)
# Edit tdt.json to change wavelet_scales parameter

# Test different wavelet mothers
# Change wavelet_mother in tdt.json (options: cmor, morl, mexh)

# Test without multi-scale features
# Use standard t-detect instead of tdt
```

## Computational Requirements

- **Time**: ~2 hours for full RAID benchmark on H100
- **Memory**: Peak GPU memory ~8GB
- **Storage**: ~1GB for models + 100MB for results

## Contact

For reproduction issues, please open a GitHub issue with:
- Error messages
- System configuration
- Steps to reproduce

## Citation

If you use this code, please cite:
```bibtex
@article{anonymous2025tdt,
  title={AI-Generated Text is Non-Stationary: Detection via Temporal Tomography},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:XXX.XXXXX},
  year={2025}
}