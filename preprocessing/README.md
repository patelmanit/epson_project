# Receipt Data Processing & Training Utilities

Quick utilities for extracting, processing, and training on receipt data from network packet captures.

## Files Overview

### 1. `decode_data.py` - Wireshark Packet Decoder
Extracts receipt data from Wireshark packet dissection files.

**Purpose:** Convert hex data from network captures to readable receipt text.

**Usage:**
```bash
python decode_data.py
```

**Input:** `full_dissection.txt` (Wireshark packet dissection)  
**Output:** `decoded_receipt_data.txt` (readable receipt text)

**Features:**
- Extracts hex data from packet "Data:" fields
- Decodes using multiple encoding strategies (UTF-8, ASCII, Latin-1, CP1252)
- Removes non-printable characters
- Two extraction modes:
  - Mode 1: Extract only "Data:" hex lines
  - Mode 2: Extract all hex patterns (comprehensive)

---

### 2. `separate_data.py` - Receipt Separator
Finds and extracts individual receipts from decoded data.

**Purpose:** Split large decoded files into individual receipt sections.

**Usage:**
```bash
python separate_data.py
```

**Input:** `decoded_receipt_data.txt`  
**Output:** `tcp_data_mod2.txt` (separated receipts)

**Extraction Logic:**
- Searches for "Date 0" pattern (receipt identifier)
- Extracts: 4 chars before + "Date 0" + 1000 chars after
- Separates sections with `--- NEXT OCCURRENCE ---`
- Handles multiple text encodings

**Configuration:**
```python
# In main()
search_string = "Date 0"    # Pattern to find
prev_chars = 4              # Characters before
next_chars = 1000           # Characters after
```

---

### 3. `tts_gen.py` - Training Data Generator
Generates structured training data (train/val/test splits) using Groq API.

**Purpose:** Create labeled training data from raw receipts.

**Usage:**
```bash
export GROQ_API_KEY='your_api_key'
python tts_gen.py
```

**Input:** `tcp_data_mod2.txt` (separated receipts)  
**Output:** `training_splits/` directory with:
- `train.json` (80% of data)
- `val.json` (10% of data)
- `test.json` (10% of data)

**Features:**
- Uses Llama3-70B for accurate ground truth generation
- Automatic 80/10/10 train/val/test split with shuffling
- Creates instruction-style prompts
- Generates valid JSON targets
- Handles encoding issues and retries on failures
- Adds metadata (split info, timestamps, seed)

**Output Format:**
```json
{
  "metadata": {
    "split_type": "train",
    "total_examples": 80,
    "split_ratio": "80/10/10",
    "created_at": "2025-01-01T12:00:00",
    "random_seed": 42
  },
  "data": [
    {
      "file_id": "1",
      "input": "Extract structured information from this receipt...",
      "target": "{\"customer_name\":\"John\",...}",
      "input_length": 245,
      "target_length": 156
    }
  ]
}
```

---

### 4. `main.py` - Model Training Script
Fine-tune T5-FLAN model on receipt parsing task.

**Purpose:** Train a neural model to extract structured data from receipts.

**Usage:**
```bash
python main.py
```

**Input:** `training_splits/` (train.json, val.json, test.json)  
**Output:** Trained models in:
- `./best_receipt_parser/` (best validation loss)
- `./final_receipt_parser/` (final epoch)

**Configuration:**
```python
num_epochs = 12
batch_size = 2
learning_rate = 3e-4
max_input_length = 512
max_target_length = 400
```

**Features:**
- JSON-optimized training with special tokens
- Gradient accumulation (effective batch size: 8)
- Automatic best model saving
- Validation-based early stopping
- JSON format post-processing
- Comprehensive accuracy metrics

**Training Metrics:**
- Exact match accuracy
- JSON validity rate
- Partial match accuracy (50%+ fields correct)

---

### 5. `debug.py` - Data Inspector
Quick utility to inspect training data structure.

**Purpose:** Verify JSON file format and structure.

**Usage:**
```bash
python debug.py
```

Checks:
- Data type (dict vs list)
- Nested structure (with 'data' key)
- First example preview
- Total examples count

---

## Complete Workflow

### Step 1: Decode Packet Data
```bash
# Extract receipt text from Wireshark capture
python decode_data.py
# Input: full_dissection.txt → Output: decoded_receipt_data.txt
```

### Step 2: Separate Individual Receipts
```bash
# Split into individual receipts
python separate_data.py
# Input: decoded_receipt_data.txt → Output: tcp_data_mod2.txt
```

### Step 3: Generate Training Data
```bash
# Create labeled train/val/test splits
export GROQ_API_KEY='your_key'
python tts_gen.py
# Input: tcp_data_mod2.txt → Output: training_splits/
```

### Step 4: Train Model
```bash
# Fine-tune T5 model
python main.py
# Input: training_splits/ → Output: best_receipt_parser/
```

### Step 5: Verify (Optional)
```bash
# Check data structure
python debug.py
```

---

## Requirements

```bash
pip install torch transformers
pip install requests pandas python-dotenv
```

**Environment Variables:**
```bash
export GROQ_API_KEY='your_groq_api_key'
```

---

## File Structure

```
.
├── decode_data.py              # 1. Decode Wireshark packets
├── separate_data.py            # 2. Separate receipts
├── tts_gen.py                  # 3. Generate training data
├── main.py                     # 4. Train model
├── debug.py                    # 5. Inspect data
│
├── full_dissection.txt         # Input: Wireshark capture
├── decoded_receipt_data.txt    # Decoded text
├── tcp_data_mod2.txt           # Separated receipts
│
├── training_splits/            # Generated training data
│   ├── train.json
│   ├── val.json
│   └── test.json
│
└── best_receipt_parser/        # Trained model
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files
```

---

## Key Features

**Data Processing:**
- Multi-encoding support (UTF-8, Latin-1, CP1252, etc.)
- Automatic retry logic with error recovery
- Clean text extraction from binary packets
- Configurable receipt extraction patterns

**Training:**
- JSON-specific tokenization
- Validation-based model selection
- Gradient accumulation for stability
- Post-processing for JSON validity

**Output Quality:**
- Structured JSON outputs
- Schema validation ready
- High accuracy on clean data
- Robust to encoding issues

---

## Quick Start

```bash
# Complete pipeline in 4 commands
python decode_data.py
python separate_data.py
export GROQ_API_KEY='your_key' && python tts_gen.py
python main.py
```

