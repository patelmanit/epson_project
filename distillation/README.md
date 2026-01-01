# Receipt Parser

A fine-tuned T5-FLAN model for extracting structured information from restaurant receipts.

## Overview

Train a small language model to parse unstructured receipt text into structured JSON containing customer details, order items, and metadata.

## Files

- `main.py` - Training pipeline and model evaluation
- `test.py` - Post-processing utilities to fix common JSON formatting issues
- `train.json`, `val.json`, `test.json` - Training, validation, and test datasets
- `test_predictions.json` - Model predictions on test set

## Quick Start

### Training
```bash
python main.py
```

### Testing with Post-Processing
```bash
python test.py
```

### Load Pre-trained Model
```python
from main import load_and_test_model
model, tokenizer, preds, metrics = load_and_test_model('./best_receipt_parser', 'test.json')
```

## Model Details

- **Base Model**: `google/flan-t5-small`
- **Task**: Sequence-to-sequence JSON generation
- **Max Input Length**: 512 tokens
- **Max Output Length**: 400 tokens

## Output Schema

```json
{
  "customer_name": "string or null",
  "date": "MM/DD/YY or null",
  "time": "HH:MMam/pm or null",
  "check_number": "string or null",
  "table_number": "string or null",
  "pickup_time": "string or null",
  "total_amount": "number or null",
  "restaurant_name": "string or null",
  "confidence_score": 0.0-1.0,
  "order_items": [
    {
      "item_name": "string",
      "quantity": 1,
      "modifiers": ["string"],
      "price": "number or null",
      "seat_number": "string (optional)"
    }
  ]
}
```

## Post-Processing

The `test.py` script applies rule-based fixes for:
- Malformed JSON structure (missing braces, incorrect array syntax)
- Date format normalization (ISO → MM/DD/YY)
- Common OCR errors (e.g., "IC LITE" → "Iron City Light")
- Invalid modifiers and items cleanup

## Training Configuration

- Epochs: 12
- Batch Size: 2 (effective 8 with gradient accumulation)
- Learning Rate: 3e-4
- Optimizer: AdamW with weight decay
- Max Gradient Norm: 1.0