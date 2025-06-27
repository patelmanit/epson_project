#!/usr/bin/env python3
"""
Quick test script for the fine-tuned receipt parser
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def quick_test():
    """Quick test of the fine-tuned model"""
    print("Testing fine-tuned receipt parser...")
    
    # Check if model exists
    model_path = "./json-receipt-parser-finetuned"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Make sure you've completed training first!")
        return
    
    print("✓ Model directory found")
    
    # Load model
    try:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test with sample receipt
    sample_receipt = """Date 06/03/25 Time 11:35am !!Reeper! Check#:424331 !!Table: 9 ! !!Pick up TimeN/A! !1[Seat 1]! !1 1 Tuna ! !1 Melt ! r!1 Italian ! rr!1 Grilled ! rr!1 Skinny Fry !"""
    
    # Format input
    input_text = f"Extract structured information from this restaurant receipt and return valid JSON.\n\nReceipt Text:\n{sample_receipt}\n\nReturn only the JSON structure:"
    
    print(f"\nTesting with sample receipt:")
    print(f"Input: {sample_receipt}")
    
    # Tokenize and generate
    try:
        inputs = tokenizer(input_text, max_length=256, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                early_stopping=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nModel output:")
        print(result)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(result)
            print(f"\n✓ Valid JSON generated!")
            print("Parsed structure:")
            print(json.dumps(parsed, indent=2))
            
            # Check if key fields are present
            key_fields = ["customer_name", "date", "time", "check_number"]
            found_fields = [field for field in key_fields if field in parsed]
            print(f"\nKey fields found: {found_fields}")
            
            if len(found_fields) >= 3:
                print("✓ Model appears to be working well!")
            else:
                print("⚠️  Model might need more training")
                
        except json.JSONDecodeError:
            print("❌ Output is not valid JSON")
            print("Model might need more training or different hyperparameters")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    quick_test()