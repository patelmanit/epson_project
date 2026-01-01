import json
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

class ReceiptDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_input_length: int = 512, max_target_length: int = 300):
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle nested JSON structure - extract the actual data list
        if isinstance(raw_data, dict) and 'data' in raw_data:
            self.data = raw_data['data']
        elif isinstance(raw_data, list):
            self.data = raw_data
        else:
            raise ValueError(f"Unexpected data structure in {data_file}. Expected dict with 'data' key or list.")
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        print(f"Loaded {len(self.data)} examples from {data_file}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            item['input'], 
            max_length=self.max_input_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target (ground truth JSON) - need to handle labels properly for T5
        targets = self.tokenizer(
            item['target'],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Replace padding token id's of the labels by -100 so it's ignored by loss
        labels = targets['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'file_id': item['file_id']
        }

def train_receipt_parser(train_file: str, val_file: str = None, num_epochs: int = 12, batch_size: int = 2, learning_rate: float = 3e-4):
    """Train the T5 model on receipt parsing data with JSON-optimized settings"""
    
    print("Loading T5-FLAN Small model...")
    # Load student model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for better JSON structure
    special_tokens = ["<JSON>", "</JSON>", "<ARRAY>", "</ARRAY>", "<OBJECT>", "</OBJECT>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets and dataloaders with JSON-optimized settings
    train_dataset = ReceiptDataset(train_file, tokenizer, max_input_length=512, max_target_length=400)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataloader = None
    if val_file:
        val_dataset = ReceiptDataset(val_file, tokenizer, max_input_length=512, max_target_length=400)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer with more conservative settings for JSON stability
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8)
    accumulation_steps = 4  # Larger effective batch size for more stable training
    
    print(f"Training setup (JSON-optimized):")
    print(f"- Training examples: {len(train_dataset)}")
    print(f"- Validation examples: {len(val_dataset) if val_file else 0}")
    print(f"- Batch size: {batch_size}")
    print(f"- Effective batch size: {batch_size * accumulation_steps}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {num_epochs}")
    print(f"- Max target length: 400 tokens")
    print()
    
    # Training loop with JSON-focused improvements
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_train_loss = 0
        model.train()
        
        # Training phase
        for batch_idx, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation with clipping for stability
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}")
        
        # Final gradient step if needed
        if len(train_dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        val_loss = 0
        if val_dataloader:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch in val_dataloader:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    total_val_loss += outputs.loss.item()
                
                val_loss = total_val_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        if val_dataloader:
            print(f"  Val Loss: {val_loss:.4f}")
        print()
        
        # Save best model
        if val_dataloader and val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss! Saving model...")
            model.save_pretrained("./best_receipt_parser")
            tokenizer.save_pretrained("./best_receipt_parser")
        elif not val_dataloader:
            # Save every few epochs if no validation
            if (epoch + 1) % 4 == 0:
                model.save_pretrained(f"./receipt_parser_epoch_{epoch+1}")
                tokenizer.save_pretrained(f"./receipt_parser_epoch_{epoch+1}")
    
    # Save final model
    model.save_pretrained("./final_receipt_parser")
    tokenizer.save_pretrained("./final_receipt_parser")
    
    print(f"Training completed!")
    print(f"Final model saved to ./final_receipt_parser")
    if val_dataloader:
        print(f"Best model (val loss: {best_val_loss:.4f}) saved to ./best_receipt_parser")
    
    return model, tokenizer

def evaluate_model(model, tokenizer, test_file: str, save_predictions: bool = True):
    """Evaluate the trained model on test data with JSON-optimized generation"""
    with open(test_file, 'r') as f:
        raw_test_data = json.load(f)
    
    # Handle nested JSON structure - same as in ReceiptDataset
    if isinstance(raw_test_data, dict) and 'data' in raw_test_data:
        test_data = raw_test_data['data']
    elif isinstance(raw_test_data, list):
        test_data = raw_test_data
    else:
        raise ValueError(f"Unexpected data structure in {test_file}. Expected dict with 'data' key or list.")
    
    model.eval()
    predictions = []
    
    print(f"Evaluating model on {len(test_data)} test examples...")
    print("-" * 80)
    
    for i, item in enumerate(test_data):
        # Tokenize input
        inputs = tokenizer(
            item['input'],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Generate prediction with JSON-optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=400,     # Increased for complex JSON
                num_beams=2,            # Reduced for more diverse output
                early_stopping=True,
                do_sample=False,        # Deterministic for JSON consistency
                temperature=1.0,
                repetition_penalty=1.1, # Prevent repetitive JSON keys
                length_penalty=1.0,     # Balanced length preference
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_bos_token_id=None  # Let model decide starting token
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to fix common JSON formatting issues
        prediction = fix_json_format(prediction)
        
        predictions.append({
            'file_id': item['file_id'],
            'input': item['input'][:100] + "..." if len(item['input']) > 100 else item['input'],
            'expected': item['target'],
            'predicted': prediction
        })
        
        print(f"Example {i+1}/{len(test_data)} (ID: {item['file_id']}):")
        print(f"Expected: {item['target']}")
        print(f"Predicted: {prediction}")
        
        # Quick similarity check
        if prediction.strip() == item['target'].strip():
            print("✅ EXACT MATCH")
        elif item['target'][:50] in prediction or prediction[:50] in item['target']:
            print("🟡 PARTIAL MATCH")
        else:
            print("❌ NO MATCH")
        print("-" * 80)
    
    # Save predictions
    if save_predictions:
        with open('test_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to test_predictions.json")
    
    return predictions

def fix_json_format(text):
    """Post-process model output to fix common JSON formatting issues"""
    # Remove any text before the first {
    if '{' in text:
        text = '{' + text.split('{', 1)[1]
    
    # Ensure it starts with {
    if not text.strip().startswith('{'):
        text = '{' + text
    
    # Fix common array formatting issues
    text = text.replace('"order_items":["item_name":', '"order_items":[{"item_name":')
    text = text.replace('","item_name":', '},{"item_name":')
    text = text.replace('"seat_number":"1"]', '"seat_number":"1"}]')
    text = text.replace('"seat_number":"2"]', '"seat_number":"2"}]')
    text = text.replace('"seat_number":"3"]', '"seat_number":"3"}]')
    text = text.replace('"seat_number":"4"]', '"seat_number":"4"}]')
    
    # Ensure proper closing
    if not text.strip().endswith('}'):
        # Find the last complete field and close properly
        if '"order_items":[' in text and not text.strip().endswith(']}'):
            text = text.rstrip() + '}]}'
        else:
            text = text.rstrip() + '}'
    
    return text

def calculate_accuracy_metrics(predictions):
    """Calculate various accuracy metrics"""
    total = len(predictions)
    exact_matches = 0
    json_parseable = 0
    partial_matches = 0
    
    for pred in predictions:
        expected = pred['expected'].strip()
        predicted = pred['predicted'].strip()
        
        # Exact match
        if expected == predicted:
            exact_matches += 1
        
        # Check if predicted output is valid JSON
        try:
            json.loads(predicted)
            json_parseable += 1
            
            # Check for partial matches in JSON structure
            try:
                exp_json = json.loads(expected)
                pred_json = json.loads(predicted)
                
                # Count matching top-level keys
                matching_keys = 0
                for key in exp_json.keys():
                    if key in pred_json and exp_json[key] == pred_json[key]:
                        matching_keys += 1
                
                if matching_keys >= len(exp_json) * 0.5:  # At least 50% of keys match
                    partial_matches += 1
            except:
                pass
                
        except json.JSONDecodeError:
            pass
    
    print(f"\n📊 ACCURACY METRICS:")
    print(f"{'='*50}")
    print(f"Total examples: {total}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total:.1%})")
    print(f"Valid JSON outputs: {json_parseable} ({json_parseable/total:.1%})")
    print(f"Partial matches: {partial_matches} ({partial_matches/total:.1%})")
    print(f"{'='*50}")
    
    return {
        'exact_accuracy': exact_matches / total,
        'json_validity': json_parseable / total,
        'partial_accuracy': partial_matches / total
    }

def load_and_test_model(model_path: str, test_file: str):
    """Load a saved model and test it"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    predictions = evaluate_model(model, tokenizer, test_file)
    metrics = calculate_accuracy_metrics(predictions)
    
    return model, tokenizer, predictions, metrics

# Main execution pipeline
if __name__ == "__main__":
    print("🍽️  RECEIPT PARSER TRAINING")
    print("=" * 50)
    
    # Check if data files exist
    required_files = ["training_splits/train.json", "training_splits/test.json", "training_splits/val.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Please ensure train.json, test.json, and val.json are in the current directory.")
        exit(1)
    
    print("✅ All data files found!")
    print()
    
    # Train the model with JSON-optimized settings
    print("🚀 Starting training...")
    model, tokenizer = train_receipt_parser(
        train_file="training_splits/train.json",
        val_file="training_splits/val.json",
        num_epochs=12,        # Slightly fewer epochs for stability
        batch_size=2,         # Smaller batch for JSON precision
        learning_rate=3e-4    # More conservative learning rate
    )
    
    print("\n" + "="*50)
    print("🧪 TESTING FINAL MODEL")
    print("="*50)
    
    # Test the final model
    predictions = evaluate_model(model, tokenizer, "training_splits/test.json")
    metrics = calculate_accuracy_metrics(predictions)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print("Models saved:")
    print("  - ./final_receipt_parser (last epoch)")
    print("  - ./best_receipt_parser (best validation loss)")
    print("\nTo test saved models later:")
    print("  model, tokenizer, preds, metrics = load_and_test_model('./best_receipt_parser', 'test.json')")