# Temporal Discrepancy Tomography (TDT)

**Official implementation of "AI-Generated Text is Non-Stationary: Detection via Temporal Tomography"**

[Paper](https://arxiv.org/abs/XXX.XXXXX) | [Demo](demo.py)

## Abstract

The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity—statistical properties vary by 73.8% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. 

We introduce **Temporal Discrepancy Tomography (TDT)**, a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies.

## Key Results

### RAID Benchmark (Table 1 in paper)
- **Overall**: 0.855 AUROC (7.1% improvement over best baseline)
- **Recipes**: 0.875 AUROC (15.3% improvement)
- **Poetry**: 0.894 AUROC (8.1% improvement)
- **News**: 0.869 AUROC (13.3% improvement)

### HART Benchmark (Tables 2-3 in paper)
- **Level 1 (Simple Detection)**: 0.825 AUROC (5.8% improvement)
- **Level 2 (Adversarial Paraphrasing)**: 0.812 AUROC (14.1% improvement)
- **Level 3 (Humanization)**: 0.891 AUROC (2.4% improvement)

### Cross-Model Generalization (Tables 4-5 in paper)
- **QWEN-3-0.6B English**: 0.724 AUROC (6.3% improvement)
- **Spanish News**: 0.638 AUROC (11.4% improvement at Level 1)
- **Arabic News**: 0.674 AUROC (33.1% improvement at Level 2)

### Efficiency
- Only 13% computational overhead compared to scalar methods
- Maintains O(n log n) complexity

## Method

TDT reformulates AI text detection as a signal processing task through three key stages:

### 1. Signal Generation
Token-level discrepancy scores Z(x) = [z₁, z₂, ..., zₙ] are converted to a continuous signal using Gaussian Kernel Density Estimation:

```
Z̃(x,t) = (1/nh) ∑ᵢ₌₁ⁿ K((t-i)/h) zᵢ
```

### 2. Continuous Wavelet Transform
The signal is decomposed using the Morlet wavelet to create a 2D time-scale representation:

```
W(a,b) = (1/√a) ∫ Z̃(x,t) ψ*((t-b)/a) dt
```

### 3. Multi-Scale Feature Extraction
Energy is extracted from three linguistically-motivated bands:
- **Morphological** (scales 1-4): Word-level anomalies
- **Syntactic** (scales 5-8): Phrase-level patterns  
- **Discourse** (scales 9-12): Paragraph-level coherence

The final representation is:
```
S_TDT(x) = [‖W_morph‖_F, ‖W_syn‖_F, ‖W_disc‖_F]
```

## Installation

```bash
# Python 3.12+ required
pip install -r requirements.txt

# For PDF report generation (optional)
# Ubuntu/Debian:
sudo apt-get install python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0

# macOS:
brew install pango

# Note: WeasyPrint requires system dependencies for PDF generation
```

## Quick Start

### Interactive Demo with Professional PDF Reports

The demo script provides a comprehensive TDT analysis system with professional PDF report generation capabilities.

#### Usage Examples

```bash
# Run interactive mode (default)
python demo.py

# Analyze specific text and generate PDF report
python demo.py --text "Your text to analyze here" --output analysis_report.pdf

# Analyze text from file
python demo.py --file input.txt --output report.pdf

# Run example detections with automatic report generation
python demo.py --examples

# Run in interactive mode
python demo.py --interactive

# Analyze without generating PDF report
python demo.py --text "Your text" --no-report

# Custom detection threshold
python demo.py --text "Your text" --threshold -0.096
```

#### Command Line Options

- `--text`: Text to analyze directly
- `--file`: Path to file containing text to analyze
- `--examples`: Run example detections with professional reports
- `--interactive`: Run in interactive mode
- `--output`: Custom path for PDF report (default: tdt_report_<timestamp>.pdf)
- `--no-report`: Disable PDF report generation
- `--threshold`: Detection threshold (default: -0.096)

#### Generated Report Features

The demo generates comprehensive PDF reports including:
- **Executive Summary**: Detection result, confidence score, and key metrics
- **Text Analysis**: Input text visualization with statistics
- **Token-Level Visualization**: Discrepancy signal with suspicious regions highlighted
- **2D Wavelet Scalogram**: Time-scale representation showing linguistic anomalies
- **Multi-Scale Features**: Energy extracted from morphological, syntactic, and discourse bands
- **Professional Layout**: Two-page report with aligned visualizations

Reports are automatically saved to the specified output path or `tdt_report_<timestamp>.pdf` by default.

### Run Experiments

```bash
# RAID benchmark
./test.sh

# HART benchmark
./main.sh

# Multilingual experiments
./langs.sh
```

## Project Structure

```
TDT/
├── demo.py                    # Interactive TDT demonstration
├── requirements.txt           # Dependencies
├── scripts/                   
│   ├── delegate_detector.py   # Main detection pipeline
│   └── detectors/            
│       ├── t_detect.py       # TDT implementation
│       └── configs/          
│           └── tdt.json      # TDT configuration
├── benchmark/                 # Datasets (not included)
├── test.sh                   # RAID experiments
├── main.sh                   # HART experiments
└── langs.sh                  # Multilingual experiments
```

## Models

TDT uses the following pre-trained models from HuggingFace:
- **Reference Model**: `tiiuae/falcon-7b`
- **Scoring Model**: `tiiuae/falcon-7b-instruct`

These will be automatically downloaded on first use.

## Citation

```bibtex
@article{anonymous2025tdt,
  title={AI-Generated Text is Non-Stationary: Detection via Temporal Tomography},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:XXX.XXXXX},
  year={2025}
}
```

## Key Contributions

1. **Empirical Evidence**: First to demonstrate that AI-generated text exhibits 73.8% higher non-stationarity than human writing
2. **Novel Paradigm**: Introduces temporal analysis to preserve positional information discarded by scalar methods
3. **State-of-the-Art Performance**: Achieves significant improvements across all benchmarks while maintaining efficiency

## License


MIT License - see [LICENSE](LICENSE) file for details.
