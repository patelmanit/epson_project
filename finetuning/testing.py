#!/usr/bin/env python3
"""
Inference script for trained receipt parser model
Tests the saved model with various receipt formats and shows detailed results
"""

import os
# CRITICAL: Set environment variables BEFORE any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
import warnings
import re
from typing import Dict, List, Tuple
from collections import OrderedDict
import jsonschema
from jsonschema import validate, ValidationError

# Suppress warnings
warnings.filterwarnings("ignore")

# Force CPU usage and single threading - ENHANCED
torch.backends.mps.is_available = lambda: False
torch.backends.cuda.is_available = lambda: False
torch.set_num_threads(1)

# Additional stability settings
if hasattr(torch.backends, 'openmp') and hasattr(torch.backends.openmp, 'is_available'):
    torch.backends.openmp.is_available = lambda: False

# Set CPU affinity if available
try:
    import psutil
    psutil.Process().cpu_affinity([0])  # Use only first CPU core
except:
    pass

# Your receipt schema
RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_name": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "time": {"type": ["string", "null"]},
        "check_number": {"type": ["string", "null"]},
        "table_number": {"type": ["string", "null"]},
        "pickup_time": {"type": ["string", "null"]},
        "total_amount": {"type": ["string", "null"]},
        "restaurant_name": {"type": ["string", "null"]},
        "confidence_score": {"type": ["number", "null"]},
        "order_items": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {"type": ["string", "null"]},
                    "quantity": {"type": ["integer", "null"]},
                    "modifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"}
                    },
                    "price": {"type": ["string", "null"]},
                    "seat_number": {"type": ["string", "null"]}
                },
                "required": ["item_name", "quantity", "modifiers", "price", "seat_number"],
                "additionalProperties": False
            }
        }
    },
    "required": ["customer_name", "date", "time", "check_number", "table_number", "pickup_time", "total_amount", "restaurant_name", "confidence_score", "order_items"],
    "additionalProperties": False
}

def validate_json_output(json_str: str) -> Tuple[bool, str, Dict]:
    """Comprehensive JSON validation"""
    try:
        # Parse JSON
        parsed = json.loads(json_str.strip())
        
        # Schema validation
        validate(instance=parsed, schema=RECEIPT_SCHEMA)
        
        return True, "Valid JSON with correct schema", parsed
        
    except json.JSONDecodeError as e:
        return False, f"JSON Parse Error: {e}", {}
    except ValidationError as e:
        return False, f"Schema Validation Error: {e.message}", {}
    except Exception as e:
        return False, f"Validation Error: {str(e)}", {}

def analyze_extraction_quality(parsed_json: Dict, original_text: str) -> Dict:
    """Analyze the quality of information extraction"""
    analysis = {
        "fields_extracted": 0,
        "fields_with_data": 0,
        "extraction_details": {},
        "quality_score": 0.0
    }
    
    total_fields = len(RECEIPT_SCHEMA["properties"])
    
    for field, value in parsed_json.items():
        analysis["fields_extracted"] += 1
        
        if value is not None and value != "":
            analysis["fields_with_data"] += 1
            
            # Check if extracted value appears in original text
            if isinstance(value, str):
                found_in_text = value.lower() in original_text.lower()
                analysis["extraction_details"][field] = {
                    "value": value,
                    "found_in_original": found_in_text,
                    "confidence": "high" if found_in_text else "low"
                }
            elif isinstance(value, list) and field == "order_items":
                # Analyze order items
                items_analysis = []
                for item in value:
                    if isinstance(item, dict):
                        item_name = item.get("item_name", "")
                        found_in_text = item_name.lower() in original_text.lower() if item_name else False
                        items_analysis.append({
                            "item_name": item_name,
                            "found_in_original": found_in_text,
                            "modifiers_count": len(item.get("modifiers", []))
                        })
                
                analysis["extraction_details"][field] = {
                    "items_count": len(value),
                    "items_analysis": items_analysis
                }
            else:
                analysis["extraction_details"][field] = {
                    "value": value,
                    "type": type(value).__name__
                }
    
    # Calculate quality score
    extraction_ratio = analysis["fields_with_data"] / total_fields
    analysis["quality_score"] = extraction_ratio
    
    return analysis

def load_model_and_tokenizer(model_path: str):
    """Load the trained model and tokenizer with enhanced error handling"""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load with CPU-only settings
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
        
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            local_files_only=False,
            torch_dtype=torch.float32,  # Explicit float32 for CPU
            device_map=None  # No device mapping
        )
        
        # Force model to CPU
        model = model.to('cpu')
        model.eval()
        
        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False
        
        # FIXED: Ensure proper token configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✓ Set pad_token to eos_token")
        
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            print("✓ Set bos_token to eos_token")
        
        # Configure model generation tokens
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
            print(f"✓ Set decoder_start_token_id to {model.config.decoder_start_token_id}")
        
        if model.config.bos_token_id is None:
            model.config.bos_token_id = tokenizer.bos_token_id
            print(f"✓ Set bos_token_id to {model.config.bos_token_id}")
        
        # Try to load generation config
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
            # Ensure required tokens are set
            generation_config.decoder_start_token_id = model.config.decoder_start_token_id
            generation_config.bos_token_id = model.config.bos_token_id
            generation_config.pad_token_id = tokenizer.pad_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id
            print("✓ Loaded and fixed custom generation config")
        except Exception as config_error:
            print(f"⚠️  Could not load generation config: {config_error}")
            generation_config = GenerationConfig(
                max_length=128,
                do_sample=False,
                num_beams=1,
                decoder_start_token_id=model.config.decoder_start_token_id,
                bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
            print("✓ Created default generation config with proper tokens")
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Vocabulary size: {len(tokenizer)}")
        print(f"✓ Model device: {next(model.parameters()).device}")
        print(f"✓ Decoder start token: {model.config.decoder_start_token_id}")
        print(f"✓ BOS token: {model.config.bos_token_id}")
        print(f"✓ EOS token: {tokenizer.eos_token_id}")
        print(f"✓ PAD token: {tokenizer.pad_token_id}")
        
        return model, tokenizer, generation_config
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"❌ Make sure the model path exists: {model_path}")
        
        # Try alternative paths
        alternative_paths = [
            "./json-focused-receipt-parser_partial",
            "./stable-receipt-parser", 
            "./enhanced-json-receipt-parser"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"🔄 Trying alternative path: {alt_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(alt_path)
                    model = AutoModelForSeq2SeqLM.from_pretrained(alt_path)
                    model = model.to('cpu')
                    model.eval()
                    
                    # Fix token configuration for alternative model too
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    if tokenizer.bos_token is None:
                        tokenizer.bos_token = tokenizer.eos_token
                    
                    if model.config.decoder_start_token_id is None:
                        model.config.decoder_start_token_id = tokenizer.pad_token_id
                    if model.config.bos_token_id is None:
                        model.config.bos_token_id = tokenizer.bos_token_id
                    
                    generation_config = GenerationConfig(
                        max_length=128,
                        do_sample=False,
                        num_beams=1,
                        decoder_start_token_id=model.config.decoder_start_token_id,
                        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                    
                    print(f"✅ Successfully loaded from: {alt_path}")
                    return model, tokenizer, generation_config
                    
                except Exception as alt_error:
                    print(f"❌ Alternative path failed: {alt_error}")
                    continue
        
        return None, None, None

def generate_json_from_receipt(model, tokenizer, generation_config, receipt_text: str) -> str:
    """Generate JSON from receipt text using the trained model - FIXED TOKEN ISSUES"""
    
    try:
        # Create simple prompt (matching training format)
        if "Receipt Text:" in receipt_text:
            # Extract just the receipt part
            receipt_part = receipt_text.split("Receipt Text:")[1].split("\n\nInstructions:")[0].strip()
            prompt = f"Convert receipt to JSON: {receipt_part[:150]}"  # Limit length
        else:
            prompt = f"Convert receipt to JSON: {receipt_text[:150]}"  # Limit length
        
        print(f"🔤 Prompt: {prompt[:80]}...")
        
        # Tokenize input with error handling
        try:
            inputs = tokenizer(
                prompt, 
                max_length=200,  # Reduced for stability 
                truncation=True, 
                return_tensors="pt",
                padding=False
            )
            
            print(f"📝 Input tokens: {inputs['input_ids'].shape[1]}")
            
        except Exception as tokenize_error:
            print(f"❌ Tokenization error: {tokenize_error}")
            return '{"error": "tokenization_failed"}'
        
        # Verify tokens are configured properly
        decoder_start_token_id = getattr(model.config, 'decoder_start_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
            model.config.decoder_start_token_id = decoder_start_token_id
            print(f"🔧 Fixed decoder_start_token_id: {decoder_start_token_id}")
        
        bos_token_id = getattr(model.config, 'bos_token_id', None)
        if bos_token_id is None:
            bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id
            model.config.bos_token_id = bos_token_id
            print(f"🔧 Fixed bos_token_id: {bos_token_id}")
        
        # Generate with enhanced error handling and explicit token configuration
        try:
            with torch.no_grad():
                # Clear any cached states
                if hasattr(model, 'clear_cache'):
                    model.clear_cache()
                
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_length=100,  # Reduced for stability
                    max_new_tokens=80,  # Explicit limit
                    do_sample=False,
                    num_beams=1,
                    
                    # EXPLICIT TOKEN CONFIGURATION
                    decoder_start_token_id=decoder_start_token_id,
                    bos_token_id=bos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    
                    repetition_penalty=1.1,
                    use_cache=False,  # Disable cache for stability
                    output_scores=False,
                    return_dict_in_generate=False
                )
                
                print(f"📤 Generated tokens: {outputs.shape[1]}")
                
        except Exception as gen_error:
            print(f"❌ Generation error: {gen_error}")
            
            # Try a simpler generation approach
            print("🔄 Trying simplified generation...")
            try:
                with torch.no_grad():
                    # Very basic generation
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=80,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,  # Use EOS as pad
                        eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.eos_token_id,  # Use EOS as start
                        use_cache=False
                    )
                    print("✓ Simplified generation succeeded")
            except Exception as simple_gen_error:
                print(f"❌ Simplified generation also failed: {simple_gen_error}")
                return '{"error": "all_generation_methods_failed"}'
        
        # Decode result with error handling
        try:
            result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(f"🔤 Raw result: {result[:100]}...")
            
        except Exception as decode_error:
            print(f"❌ Decoding error: {decode_error}")
            return '{"error": "decoding_failed"}'
        
        # Remove prompt from result
        if prompt in result:
            json_part = result.replace(prompt, "").strip()
        else:
            json_part = result.strip()
        
        # Extract JSON if it's embedded in other text
        if '{' in json_part and '}' in json_part:
            start = json_part.find('{')
            end = json_part.rfind('}') + 1
            json_part = json_part[start:end]
        
        # Basic cleanup
        json_part = json_part.strip()
        
        # If no JSON found, create minimal structure based on input
        if not json_part or not (json_part.startswith('{') and json_part.endswith('}')):
            print("⚠️  No valid JSON structure found, creating fallback from input")
            
            # Try to extract some basic info from the receipt text
            fallback_json = {
                "customer_name": None,
                "date": None,
                "time": None,
                "check_number": None,
                "table_number": None,
                "pickup_time": None,
                "total_amount": None,
                "restaurant_name": None,
                "confidence_score": 0.1,
                "order_items": None,
                "extraction_status": "fallback_generated",
                "raw_model_output": json_part[:50] if json_part else "empty"
            }
            
            # Try to extract customer name from !! marks
            import re
            customer_match = re.search(r'!!([^!]+)!', receipt_text)
            if customer_match:
                fallback_json["customer_name"] = customer_match.group(1).strip()
                fallback_json["confidence_score"] = 0.3
            
            json_part = json.dumps(fallback_json, separators=(',', ':'))
        
        return json_part
        
    except Exception as overall_error:
        print(f"❌ Overall generation error: {overall_error}")
        
        # Ultimate fallback
        emergency_json = {
            "customer_name": None,
            "date": None,
            "time": None,
            "check_number": None,
            "table_number": None,
            "pickup_time": None,
            "total_amount": None,
            "restaurant_name": None,
            "confidence_score": 0.0,
            "order_items": None,
            "error": "generation_completely_failed",
            "error_details": str(overall_error)[:100]
        }
        
        return json.dumps(emergency_json, separators=(',', ':'))

def test_model_on_samples(model, tokenizer, generation_config, test_cases: List[Dict]):
    """Test model on multiple receipt samples"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*80)
    
    total_tests = len(test_cases)
    valid_json_count = 0
    schema_valid_count = 0
    quality_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}/{total_tests}")
        print(f"{'='*60}")
        
        receipt_text = test_case.get("input", test_case.get("text", ""))
        expected_output = test_case.get("target", None)
        
        print(f"📄 Receipt Text:")
        print(f"{receipt_text[:200]}{'...' if len(receipt_text) > 200 else ''}")
        
        if expected_output:
            print(f"\n🎯 Expected Output:")
            try:
                expected_parsed = json.loads(expected_output)
                print(json.dumps(expected_parsed, indent=2))
            except:
                print(expected_output[:200])
        
        print(f"\n🤖 Model Output:")
        
        try:
            # Generate JSON
            generated_json = generate_json_from_receipt(model, tokenizer, generation_config, receipt_text)
            print(f"Raw: {generated_json}")
            
            # Validate JSON
            is_valid, error_msg, parsed_json = validate_json_output(generated_json)
            
            print(f"\n📊 Validation Results:")
            print(f"✓ Valid JSON: {is_valid}")
            
            if is_valid:
                valid_json_count += 1
                schema_valid_count += 1
                
                print(f"✓ Schema Compliant: Yes")
                print(f"\n📋 Formatted Output:")
                print(json.dumps(parsed_json, indent=2))
                
                # Analyze extraction quality
                analysis = analyze_extraction_quality(parsed_json, receipt_text)
                quality_scores.append(analysis["quality_score"])
                
                print(f"\n🔍 Extraction Analysis:")
                print(f"Fields with data: {analysis['fields_with_data']}/{analysis['fields_extracted']}")
                print(f"Quality score: {analysis['quality_score']:.2f}")
                
                # Show key extractions
                for field, details in analysis["extraction_details"].items():
                    if field != "order_items" and details.get("value"):
                        confidence = details.get("confidence", "unknown")
                        print(f"  • {field}: '{details['value']}' ({confidence} confidence)")
                
                # Show order items
                if "order_items" in analysis["extraction_details"]:
                    items_info = analysis["extraction_details"]["order_items"]
                    print(f"  • order_items: {items_info['items_count']} items extracted")
                    for item_analysis in items_info.get("items_analysis", [])[:3]:  # Show first 3
                        item_name = item_analysis["item_name"]
                        found = "✓" if item_analysis["found_in_original"] else "❌"
                        print(f"    - {item_name} {found}")
                
            else:
                print(f"❌ Schema Compliant: No")
                print(f"❌ Error: {error_msg}")
                quality_scores.append(0.0)
            
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            quality_scores.append(0.0)
        
        print(f"\n{'-'*60}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Results Overview:")
    print(f"  • Total tests: {total_tests}")
    print(f"  • Valid JSON: {valid_json_count}/{total_tests} ({valid_json_count/total_tests*100:.1f}%)")
    print(f"  • Schema compliant: {schema_valid_count}/{total_tests} ({schema_valid_count/total_tests*100:.1f}%)")
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"  • Average quality score: {avg_quality:.2f}")
        print(f"  • Best quality score: {max(quality_scores):.2f}")
        print(f"  • Worst quality score: {min(quality_scores):.2f}")
    
    print(f"\n🎯 Model Performance Rating:")
    if valid_json_count / total_tests >= 0.8:
        print("🟢 EXCELLENT - Model generates valid JSON consistently")
    elif valid_json_count / total_tests >= 0.6:
        print("🟡 GOOD - Model generates valid JSON most of the time")
    elif valid_json_count / total_tests >= 0.3:
        print("🟠 FAIR - Model sometimes generates valid JSON")
    else:
        print("🔴 POOR - Model rarely generates valid JSON")

def load_test_data_from_file(file_path: str) -> List[Dict]:
    """Load test data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} test samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ Test file {file_path} not found")
        return []
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return []

def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases if no file is available"""
    return [
        {
            "input": "Date 06/03/25 Time 12:41pm !!Reeper! Check#:424360 !!Table: P 2 ! !!Pick up TimeN/A! !1[Seat 1]! !1 1 Yuengling !",
            "description": "Simple receipt with customer name and one item"
        },
        {
            "input": "Date 06/03/25 Time 12:29pm !!Ava! Check#:424358 !!Table: W-6 ! !!Pick up TimeN/A! !1[Seat 1]! !1 1 Crown Peach ! r!1 On Rocks ! r!1 1 Kahlua !",
            "description": "Receipt with modifiers"
        },
        {
            "input": "McDonald's Order #1234 Date: 06/26/25 Time: 12:30pm Big Mac Fries Coke Total: $10.47",
            "description": "Standard restaurant receipt format"
        },
        {
            "input": "Starbucks Receipt Date: 2025-06-26 Time: 8:15am Item: Latte Item: Croissant Total: $8.50",
            "description": "Coffee shop receipt"
        },
        {
            "input": "Pizza Hut Table: 5 Check: 789 Date: 06/26/25 Pepperoni Pizza Large Garlic Bread Total: $24.99",
            "description": "Pizza restaurant receipt"
        }
    ]

def main():
    """Main inference function with enhanced stability"""
    print("🚀 Receipt Parser Model Inference Script")
    print("="*60)
    print("🔧 macOS OpenMP compatibility mode enabled")
    
    # Configuration
    MODEL_PATH = "./jf-parser"  # Change this to your model path
    TEST_DATA_FILE = "test.json"  # Change this to your test file
    
    print(f"📂 Model path: {MODEL_PATH}")
    print(f"📂 Test data: {TEST_DATA_FILE}")
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model path not found: {MODEL_PATH}")
        print("📁 Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item) and any(keyword in item.lower() for keyword in ["receipt", "parser", "model", "json"]):
                print(f"  - {item}")
        
        # Try to find model automatically
        possible_paths = [
            "./json-focused-receipt-parser_partial",
            "./stable-receipt-parser",
            "./enhanced-json-receipt-parser",
            "./fixed-receipt-parser"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                MODEL_PATH = path
                print(f"🔄 Using found model: {MODEL_PATH}")
                break
        else:
            print("❌ No model found. Please train a model first.")
            return
    
    # Load model with retries
    model, tokenizer, generation_config = None, None, None
    
    for attempt in range(3):
        print(f"\n📥 Loading model (attempt {attempt + 1}/3)...")
        try:
            model, tokenizer, generation_config = load_model_and_tokenizer(MODEL_PATH)
            if model is not None:
                break
        except Exception as e:
            print(f"❌ Loading attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("🔄 Retrying...")
                import time
                time.sleep(2)
    
    if model is None:
        print("❌ Failed to load model after multiple attempts. Exiting.")
        return
    
    # Load test data
    test_cases = load_test_data_from_file(TEST_DATA_FILE)
    
    if not test_cases:
        print("📝 Using sample test cases...")
        test_cases = create_sample_test_cases()
    
    # Limit test cases to prevent memory issues
    if len(test_cases) > 10:
        print(f"⚠️  Limiting test cases to 10 (from {len(test_cases)}) for stability")
        test_cases = test_cases[:10]
    
    # Test the model
    try:
        test_model_on_samples(model, tokenizer, generation_config, test_cases)
    except Exception as test_error:
        print(f"❌ Testing error: {test_error}")
        print("🔄 Trying individual test cases...")
        
        # Try each test case individually
        for i, test_case in enumerate(test_cases[:3]):  # Just try first 3
            try:
                print(f"\n🧪 Individual test {i+1}:")
                receipt_text = test_case.get("input", test_case.get("text", ""))[:200]
                result = generate_json_from_receipt(model, tokenizer, generation_config, receipt_text)
                print(f"Result: {result}")
            except Exception as individual_error:
                print(f"❌ Individual test {i+1} failed: {individual_error}")
    
    print(f"\n✅ Inference testing completed!")
    print(f"📁 Model tested: {MODEL_PATH}")
    print(f"📊 Test cases processed: {len(test_cases)}")
    print(f"💡 Tip: If you see errors, try with a smaller model or reduce batch size")

if __name__ == "__main__":
    main()