# Receipt Parser Training Pipeline

A comprehensive end-to-end pipeline for training T5/FLAN-T5 models to extract structured information from restaurant receipts using fine-tuning.

## Overview

This pipeline provides a complete workflow from raw receipt text to a production-ready model that generates valid JSON outputs. It includes ground truth generation, data preprocessing, model training, and inference testing.

## Pipeline Components

### 1. Ground Truth Generation (`ground_truth_gen.py`)
Uses the Groq API with Llama models to automatically generate high-quality training labels from raw receipt text.

**Features:**
- Splits multi-receipt files using `--- NEXT OCCURRENCE ---` separator
- Leverages Llama3-70B for accurate parsing of complex receipts
- Handles multiple text encodings (UTF-8, Latin-1, CP1252, etc.)
- Automatic retry logic with error recovery
- Generates both JSON ground truths and CSV summaries

**Usage:**
```bash
export GROQ_API_KEY='your_api_key'
python ground_truth_gen.py
```

**Input:** `tcp_data_mod1.txt` (or similar multi-receipt file)

**Output Structure:**
```
receipt_training_data/
├── inputs/
│   ├── input_1.txt
│   ├── input_2.txt
│   └── ...
├── ground_truths/
│   ├── ground_truth_1.json
│   ├── ground_truth_2.json
│   └── ...
├── ground_truth_summary.csv
└── order_items_detail.csv
```

### 2. Data Preprocessing (`finetuning_setup.py`)
Prepares raw training data for FLAN-T5 fine-tuning with comprehensive validation and analysis.

**Features:**
- Loads and pairs input texts with ground truth JSONs
- Text cleaning and normalization
- Token length analysis with configurable limits
- Train/validation/test split (70/15/15 by default)
- Dataset statistics and field coverage analysis
- Multiple output formats (JSON, JSONL)

**Usage:**
```python
from finetuning_setup import ReceiptDataPreprocessor

preprocessor = ReceiptDataPreprocessor(
    data_folder="receipt_training_data",
    model_name="google/flan-t5-small",
    max_input_length=512,
    max_output_length=512
)

# Run full pipeline
raw_data = preprocessor.load_raw_data()
preprocessor.analyze_dataset_statistics()
processed_data = preprocessor.process_for_training()
train, val, test = preprocessor.create_train_val_test_splits()
preprocessor.save_processed_data(train, val, test)
```

**Output:**
```
processed_receipt_data/
├── train.json / train.jsonl
├── val.json / val.jsonl
├── test.json / test.jsonl
├── dataset_stats.json
└── preprocessing_config.json
```

### 3. Model Fine-tuning (`ft.py`)
JSON-focused training script with advanced validation and error handling.

**Key Features:**
- **JSON-Optimized Training:** Specialized for valid JSON generation
- **Schema Validation:** Uses jsonschema for strict output validation
- **Custom Metrics:** JSON validity, schema compliance tracking
- **Multi-Strategy Generation:** Fallback mechanisms for robust outputs
- **macOS Compatibility:** OpenMP/MPS handling for Apple Silicon
- **Enhanced Token Management:** JSON-specific special tokens

**Configuration:**
```python
MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6
```

**Usage:**
```bash
python ft.py
```

**Training Features:**
- Automatic best model selection based on JSON validity
- Partial model saving on failures
- Comprehensive test suite with diverse receipt formats
- Real-time JSON quality metrics during training

**Output:** `./jf-parser/` (trained model + tokenizer)

### 4. Model Testing (`testing.py`)
Comprehensive inference and validation script with detailed analysis.

**Features:**
- **Schema Validation:** Ensures outputs match expected JSON structure
- **Quality Analysis:** Extraction quality scoring and field coverage
- **Error Handling:** Graceful degradation with retry logic
- **Detailed Reporting:** Per-example validation with confidence metrics
- **Platform Stability:** CPU-only mode with single-threading for macOS

**Usage:**
```bash
python testing.py
```

**Capabilities:**
- Load models from custom paths
- Test on JSON files or sample cases
- Validate against receipt schema
- Analyze extraction quality (fields found, confidence)
- Generate comprehensive performance reports

**Output Metrics:**
- Valid JSON percentage
- Schema compliance rate
- Average quality score
- Field-by-field extraction analysis
- Performance rating (EXCELLENT/GOOD/FAIR/POOR)

### 5. Debug Utilities (`debug.py`)
Quick comparison tool for validating data processing pipeline.

**Purpose:** Compare original ground truth files with processed training targets to ensure no data corruption during preprocessing.

## Receipt JSON Schema

All models are trained to generate outputs matching this schema:

```json
{
  "customer_name": "string or null",
  "date": "MM/DD/YY or null",
  "time": "HH:MMam/pm or null",
  "check_number": "string or null",
  "table_number": "string or null",
  "pickup_time": "string or null",
  "total_amount": "string or null",
  "restaurant_name": "string or null",
  "confidence_score": 0.0-1.0,
  "order_items": [
    {
      "item_name": "string",
      "quantity": integer,
      "modifiers": ["string"],
      "price": "string or null",
      "seat_number": "string (optional)"
    }
  ]
}
```

## Complete Workflow

### Step 1: Generate Ground Truth
```bash
# Set up API key
export GROQ_API_KEY='your_groq_api_key'

# Generate labeled data from raw receipts
python ground_truth_gen.py
```

### Step 2: Preprocess Data
```bash
# Create train/val/test splits
python finetuning_setup.py
```

### Step 3: Train Model
```bash
# Fine-tune FLAN-T5 on receipt data
python ft.py
```

### Step 4: Test Model
```bash
# Validate model performance
python testing.py
```

## System Requirements

**Python:** 3.8+

**Key Dependencies:**
```bash
pip install torch transformers datasets
pip install jsonschema requests pandas python-dotenv
pip install scikit-learn matplotlib seaborn
```

**Environment Variables:**
- `GROQ_API_KEY` - Required for ground truth generation

## File Structure

```
.
├── ground_truth_gen.py       # Generate training labels
├── finetuning_setup.py        # Preprocess data
├── ft.py                      # Train model
├── testing.py                 # Test/validate model
├── debug.py                   # Debugging utilities
├── receipt_training_data/     # Raw data
│   ├── inputs/
│   └── ground_truths/
├── processed_receipt_data/    # Preprocessed splits
│   ├── train.json
│   ├── val.json
│   └── test.json
└── jf-parser/                 # Trained model
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files
```

## Advanced Usage

### Custom Model Selection
```python
# In ft.py, change MODEL_NAME
MODEL_NAME = "google/flan-t5-base"  # Larger model
# or
MODEL_NAME = "google/flan-t5-large"  # Even better quality
```

### Adjust Training Parameters
```python
# More epochs for better convergence
NUM_EPOCHS = 10

# Larger batch size (if memory allows)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

# Higher learning rate for faster training
LEARNING_RATE = 2e-4
```