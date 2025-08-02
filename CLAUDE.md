# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Truth Mirror is a research project implementing two-dimensional AI-generated text detection by decoupling text into content and expression dimensions. The system detects three levels of AI risk and performs binary classification within a two-dimensional space.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running Experiments
- **Main English experiments (HART dataset)**: `./main.sh` (Note: Currently misconfigured to run RAID experiments)
- **Alternative detector experiments**: `./main2.sh`  
- **Multilingual experiments (5 languages)**: `./langs.sh`
- **RAID dataset experiments**: `./raid.sh`
- **Wavelet-enhanced experiments**: `./test.sh`

### Direct Python Execution
```bash
python scripts/delegate_detector.py --data_path ./benchmark/hart --result_path ./exp_main \
                                   --datasets essay.dev,essay.test --detectors fast_detect,binoculars

# Common parameters:
# --model: SVR (default), LogisticRegression, LinearSVR
# --ndev: Number of development samples (-1 for all)
# --seed: Random seed (default: 1)
```

### Running Tests
```bash
# Run wavelet feature tests
python test/test_wavelet_features.py

# Run delegate wavelet tests
python test/test_delegate_wavelet.py

# Simple wavelet test
python test_wavelet_simple.py
```

## Architecture Overview

### Core Components

1. **DelegateDetector** (`scripts/delegate_detector.py`): Main orchestrator that manages the detection pipeline
   - Handles 1D (threshold-based) and 2D (model-based) detection modes
   - Supports feature combinations: T (text), C (content), E (expression)
   - Manages training/evaluation across multiple datasets and detectors

2. **Detector Framework** (`scripts/detectors/`):
   - **DetectorBase**: Abstract base class for all detectors
   - **Available detectors**: fast_detect, binoculars, glimpse, radar, roberta, lrr, log_perplexity, log_rank, t-detect, tdt
   - **Configuration-driven**: Each detector uses JSON configs in `configs/` directory

3. **Two-Dimensional Detection**:
   - **1D Mode**: Single feature (e.g., `fast_detect`) with threshold-based classification
   - **2D Mode**: Multi-feature combinations (e.g., `C(fast_detect)`, `CT(binoculars)`) with ML models
   - **Feature Types**: 
     - T: Original text (generation)
     - C: Content dimension  
     - E: Expression/Language dimension (disabled in final version)

### Data Flow

1. **Preparation Phase**: Compute detection criteria for each text using specified detectors
2. **Training Phase**: Fit thresholds (1D) or ML models (2D) on dev sets
3. **Evaluation Phase**: Test on test sets and generate performance metrics (AUROC, F1, TPR@5%FPR)

### Key Files

- `scripts/delegate_detector.py`: Main detection pipeline
- `scripts/detectors/detector_base.py`: Base detector interface
- `scripts/detectors/__init__.py`: Detector factory and registry
- `scripts/utils.py`: Utility functions for data handling and labeling
- `scripts/content_extractor.py`: LLM-based content dimension extraction
- `scripts/language_extractor.py`: Expression dimension extraction
- `benchmark/`: Contains HART and RAID datasets
- `exp_*/`: Experiment output directories

### Configuration System

Detectors use JSON configuration files in `scripts/detectors/configs/` with support for environment variable substitution using `${VAR_NAME}` syntax. Model paths are typically configured here.

**Common Model Requirements**:
- Falcon-7B models: `/home/wyx/model/falcon-7b` and `/home/wyx/model/falcon-7b-instruct`
- GPU with CUDA support required for transformer-based detectors
- Cache directory: `./cache` for tokenizer caching

### Dataset Structure

The system expects JSON datasets with fields:
- `generation`: Original text
- `content`: Content representation
- `language`: Expression representation (optional)
- `source`: Label for human/machine classification
- `domain`: Dataset domain for grouping
- `task_level1/2/3`: Multi-level detection targets

### Performance Evaluation

Results are grouped by domain and reported with three metrics:
- **AUROC**: Area under ROC curve
- **F1**: F1 score
- **TPR@5%FPR**: True positive rate at 5% false positive rate

## Detector Algorithm Implementations

### 1. FastDetectGPT (`scripts/detectors/fast_detect_gpt.py`)

**Algorithm**: Sampling Discrepancy Analysis

**LaTeX Formulation**:
```latex
\text{Discrepancy} = \frac{\sum_{i=1}^{n} \log p_{\text{score}}(x_i) - \sum_{i=1}^{n} \mathbb{E}_{p_{\text{ref}}}[\log p_{\text{score}}(X_i)]}{\sqrt{\sum_{i=1}^{n} \text{Var}_{p_{\text{ref}}}[\log p_{\text{score}}(X_i)]}}
```

**Core Implementation**:
```python
def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy.mean().item()
```

**Key Features**:
- Uses two models: reference model and scoring model (can be the same)
- Computes discrepancy between actual log-likelihood and expected log-likelihood
- Normalizes by variance for statistical significance
- Higher discrepancy indicates machine-generated text

### 2. Binoculars (`scripts/detectors/binoculars.py`)

**Algorithm**: Cross-Perplexity Analysis

**LaTeX Formulation**:
```latex
\text{Binoculars Score} = \frac{\text{PPL}_{\text{performer}}(x)}{\text{CrossPPL}_{\text{observer} \to \text{performer}}(x)}
```

Where:
```latex
\text{PPL}_{\text{performer}}(x) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n} \log p_{\text{performer}}(x_i | x_{<i})\right)
```

```latex
\text{CrossPPL}_{\text{observer} \to \text{performer}}(x) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n} \log p_{\text{performer}}(\tilde{x}_i | x_{<i})\right)
```

**Core Implementation**:
```python
def compute_crit(self, text):
    encodings = self._tokenize([text])
    observer_logits, performer_logits = self._get_logits(encodings)
    ppl = perplexity(encodings, performer_logits)  # Standard perplexity
    x_ppl = entropy(observer_logits, performer_logits, encodings, pad_token_id)  # Cross-perplexity
    binoculars_scores = ppl / x_ppl  # Ratio
    return binoculars_scores[0]
```

**Key Features**:
- Uses two models: observer (reference) and performer (scoring)
- Computes ratio of standard perplexity to cross-perplexity
- Cross-perplexity measures how well performer predicts observer's preferred tokens
- Lower ratios indicate machine-generated text

### 3. Glimpse (`scripts/detectors/glimpse.py`)

**Algorithm**: OpenAI API-based FastDetectGPT with Geometric Distribution

**LaTeX Formulation**:
```latex
\text{Geometric Distribution}: P(X = k) = p_K \cdot \lambda^{k-K}, \quad k > K
```

**Core Implementation**:
```python
class PdeFastDetectGPT(PdeBase):
    def __call__(self, item):
        logprobs = np.array(item["logprobs"])
        probs = self.estimate_distrib_sequence(item)
        lprobs = np.nan_to_num(np.log(probs))
        mean_ref = (probs * lprobs).sum(axis=-1)
        var_ref = (probs * lprobs**2).sum(axis=-1) - mean_ref**2
        discrepancy = (logprobs.sum() - mean_ref.sum()) / np.sqrt(var_ref.sum())
        return discrepancy.item()
```

**Key Features**:
- Uses OpenAI API for token probabilities
- Estimates full vocabulary distribution using geometric approximation
- Applies same discrepancy calculation as FastDetectGPT
- Requires OpenAI API key and supports Azure endpoints

### 4. Baseline Detectors (`scripts/detectors/baselines.py`)

**Algorithms**: Log-Perplexity, Log-Rank, Likelihood, Entropy

**LaTeX Formulations**:
```latex
\text{Log-Perplexity} = -\frac{1}{n}\sum_{i=1}^{n} \log p(x_i | x_{<i})
```

```latex
\text{Log-Rank} = -\frac{1}{n}\sum_{i=1}^{n} \log(\text{rank}(x_i))
```

```latex
\text{Likelihood} = \frac{1}{n}\sum_{i=1}^{n} \log p(x_i | x_{<i})
```

```latex
\text{Entropy} = -\frac{1}{n}\sum_{i=1}^{n} \sum_{v} p(v | x_{<i}) \log p(v | x_{<i})
```

### 5. LRR Detector (`scripts/detectors/detect_llm.py`)

**Algorithm**: Log-Likelihood to Log-Rank Ratio

**LaTeX Formulation**:
```latex
\text{LRR} = -\frac{\text{Likelihood}}{\text{Log-Rank}} = -\frac{\frac{1}{n}\sum_{i=1}^{n} \log p(x_i | x_{<i})}{\frac{1}{n}\sum_{i=1}^{n} \log(\text{rank}(x_i))}
```

### 6. Neural Classifiers (RoBERTa, Radar)

**Algorithm**: Fine-tuned transformer classification

**Implementation**: Direct classification using pre-trained models
- RoBERTa: Returns probability of class 1 (machine-generated)
- Radar: Returns probability of class 0 (human-generated)

### 7. T-Detect and TDT (Temporal Discrepancy Tomography)

**T-Detect**: Token-level detection using statistical features with robust t-distribution normalization
**TDT**: Temporal Discrepancy Tomography - a novel detection paradigm that preserves positional information by applying Continuous Wavelet Transform to token-level discrepancies, creating a 2D time-scale representation that captures both location and linguistic scale of statistical anomalies

## Two-Dimensional Detection (CT Algorithm)

### Algorithm Overview

The CT (Content + Text) algorithm implements the core two-dimensional detection by combining multiple feature dimensions:

**Feature Dimensions**:
- **T (Text)**: Original generation text
- **C (Content)**: Simplified/outlined content representation
- **E (Expression)**: Language expression variation (disabled in final version)

### Implementation in `delegate_detector.py`

**1D Detection (Threshold-based)**:
```python
def _fit_threshold(self, datasets, category, label_fn):
    # For single feature detectors like 'fast_detect'
    pairs = [(item[f'{field}_crit'], label_fn(item)) for item in results]
    # Find optimal threshold using F1 score
    precision, recall, thresholds = precision_recall_curve(labels, crits)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    threshold = thresholds[np.argmax(f1)]
```

**2D Detection (Model-based)**:
```python
def _fit_model(self, datasets, category, label_fn):
    # For multi-feature detectors like 'C(fast_detect)', 'CT(binoculars)'
    features = [[item[f'{field}_crit'] for field in self.feature_fields] for item in results]
    model = DelegateModel(self.args.model).fit(features, labels)  # SVR/LogisticRegression
    crits = model.predict(features)
    threshold = find_optimal_threshold(crits, labels)  # F1-based
```

**Feature Field Mapping**:
```python
def _get_feature_fields(self, name2d):
    abbr_fields = {
        'T': 'generation',  # Original text
        'C': 'content',     # Content dimension
        'E': 'language',    # Expression dimension
    }
    if name2d is None:
        return [abbr_fields['T']]  # 1D: only text
    return [abbr_fields[ch] for ch in name2d]  # 2D: multiple features
```

## Main.sh Processing Pipeline

### Execution Flow

1. **Dataset Loading**: `benchmark/hart/*.json` files containing:
   - `generation`: Original text
   - `content`: Content representation (extracted via LLM)
   - `language`: Expression variation (optional)
   - Labels: `task_level1`, `task_level2`, `task_level3`

2. **Detector Preparation**: For each detector and dataset:
   ```bash
   # Example: fast_detect on essay.dev
   python scripts/delegate_detector.py --datasets essay.dev --detectors fast_detect
   ```
   - Computes detection criteria for each text dimension
   - Saves results to `exp_main/essay.dev.fast_detect.json`

3. **Training Phase**: On `.dev` datasets:
   - Fits thresholds (1D) or ML models (2D) for each risk level
   - Saves models to `exp_main/temp/level{X}.{detector}.model.pkl`

4. **Evaluation Phase**: On `.test` datasets:
   - Loads trained models and applies to test data
   - Computes metrics: AUROC, F1, TPR@5%FPR
   - Groups results by domain for analysis

### Risk Level Detection

- **Level 1**: Basic human vs machine detection
- **Level 2**: Advanced detection including rephrased text
- **Level 3**: Comprehensive detection including humanized text

### Content Extraction Process

Content dimension is extracted using `scripts/content_extractor.py`:

**Prompts**:
- **Simplify**: "Simplify the text to make it clear and concise while preserving its meaning"
- **Outline**: "Outline the main points of the text to get a clear and concise picture of the content"

**Process**:
1. Load original text from dataset
2. Apply LLM-based content extraction
3. Store simplified/outlined version in `content` field
4. Use both `generation` and `content` for 2D detection

## Important Notes

- The system requires GPU access for transformer-based detectors
- Model paths in detector configs must point to valid HuggingFace model directories (currently set to `/home/wyx/model/`)
- Cache directory is used for tokenizer caching to improve performance
- Random seeds are set for reproducibility across experiments
- Content extraction requires OpenAI API access for LLM-based simplification
- CT algorithm effectiveness depends on quality of content extraction
- The `main.sh` script is currently misconfigured to run RAID experiments instead of HART experiments

## Experiment Result Structure

Results are saved in experiment directories with the following naming convention:
- **Result files**: `[dataset].[split].[detector].json`
- **Model files**: `temp/level[1-3].[detector].model.pkl` or `temp/level[1-3].[detector].config.json`
- **Log files** (for wavelet experiments): `logs_wavelet/[type]_[timestamp].log`

Each result file contains detection criteria computed for all samples, which are then used for training and evaluation across the three risk levels.