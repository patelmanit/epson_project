import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
import time
import requests
import random
from datetime import datetime

load_dotenv()

class GroqReceiptProcessor:
    def __init__(self, model_name: str = "llama3-70b-8192", groq_base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize with Groq API for receipt processing
        
        Args:
            model_name: Groq model to use (using 70b for better complex parsing)
            groq_base_url: Base URL for Groq API
        """
        self.model_name = model_name
        self.groq_base_url = groq_base_url.rstrip('/')
        
        # Get Groq API key from environment
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            print("⚠️ Warning: GROQ_API_KEY not found in environment variables")
            print("💡 Set your Groq API token: export GROQ_API_KEY='your_token_here'")
            raise ValueError("Groq API key is required")
        
        print(f"🤖 Initializing Groq receipt processor...")
        print(f"🌐 Using Groq API endpoint: {self.groq_base_url}")
        print(f"🔧 Using model: {self.model_name}")
        
        # Test connection
        self._test_connection()
        
        print(f"✅ Groq API connection successful!")
    
    def _test_connection(self):
        """Test connection to Groq service"""
        try:
            print("🧪 Testing Groq API connection with simple prompt...")
            test_prompt = "Say 'Hello, I am working!' and nothing else."
            response = self._generate_text(test_prompt, max_new_tokens=50)
            print(f"✅ Test response: '{response}'")
            print(f"✅ Connected to Groq API successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Groq: {e}")
            print("Make sure your API key is valid and you have credits")
            raise
    
    def _generate_text(self, prompt: str, max_new_tokens: int = 800) -> str:
        """
        Generate text using Groq API with increased token limit
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated text
        """
        # Prepare request payload (OpenAI chat format)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.1,  # Low temperature for consistent parsing
            "top_p": 0.9
        }
        
        # Set headers with API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}"
        }
        
        try:
            print(f"🌐 Making request to Groq API...")
            # Make the request to chat/completions endpoint
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120  # Increased timeout for complex receipts
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"📦 Response structure: {list(result.keys())}")
            
            # Extract the generated text from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                print(f"✅ Generated {len(generated_text)} characters")
                return generated_text
            else:
                print(f"❌ Unexpected response format: {result}")
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request timed out - model may be slow or unavailable")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to {self.groq_base_url}")
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"❌ Response body: {e.response.text}")
            if e.response.status_code == 401:
                raise Exception("Invalid API key - check your Groq API key")
            elif e.response.status_code == 429:
                raise Exception("Rate limit exceeded - please wait before making more requests")
            elif e.response.status_code == 402:
                raise Exception("Insufficient credits - please add credits to your Groq account")
            else:
                raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def split_receipt_file(self, file_path: str) -> List[str]:
        """
        Split the input file by '--- NEXT OCCURRENCE ---' separator
        Handles various encoding issues common in receipt files
        
        Args:
            file_path: Path to the input text file containing multiple receipts
            
        Returns:
            List of individual receipt texts
        """
        content = None
        
        # Try multiple encoding strategies
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"✅ Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"❌ Failed to read with {encoding} encoding, trying next...")
                continue
        
        # If all encodings fail, try reading as binary and cleaning
        if content is None:
            print("⚠️ All text encodings failed, reading as binary and cleaning...")
            try:
                with open(file_path, 'rb') as f:
                    raw_bytes = f.read()
                
                # Replace problematic bytes and decode
                content = raw_bytes.decode('utf-8', errors='replace')
                
                # Clean up replacement characters
                content = content.replace('\ufffd', '')  # Remove replacement chars
                
            except Exception as e:
                raise Exception(f"Could not read file {file_path}: {e}")
        
        # Additional cleaning for common receipt file issues
        content = self._clean_raw_content(content)
        
        # Split by the separator
        receipts = content.split("--- NEXT OCCURRENCE ---")
        
        # Clean up each receipt (remove extra whitespace)
        cleaned_receipts = []
        for receipt in receipts:
            receipt = receipt.strip()
            if receipt:  # Only add non-empty receipts
                cleaned_receipts.append(receipt)
        
        print(f"📄 Found {len(cleaned_receipts)} individual receipts")
        return cleaned_receipts
    
    def _clean_raw_content(self, content: str) -> str:
        """
        Clean problematic characters commonly found in receipt files
        """
        # Remove null bytes
        content = content.replace('\x00', '')
        
        # Replace common problematic characters
        replacements = {
            '\x0c': '\n',  # Form feed to newline
            '\x0b': '\n',  # Vertical tab to newline
            '\x1a': '',    # Substitute character
            '\x7f': '',    # DEL character
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Remove other control characters except common ones (tab, newline, carriage return)
        cleaned_chars = []
        for char in content:
            if ord(char) < 32 and char not in '\t\n\r':
                continue  # Skip control characters
            cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    def create_extraction_prompt(self, receipt_text: str) -> str:
        """
        Create an enhanced prompt that matches the exact format from your training examples
        """
        # Don't truncate - let the model handle the full receipt
        if len(receipt_text) > 2000:
            receipt_text = receipt_text[:2000] + "..."
        
        prompt = f"""Extract structured information from this restaurant receipt and return valid JSON.

Receipt Text:
{receipt_text}

Instructions:
- Extract customer_name from text between !! marks
- Extract date in original format
- Extract time in original format  
- Extract check_number from "Check#:" field
- Extract table_number if present
- Extract pickup_time if mentioned
- Extract all order_items with their quantities, modifiers, and seat numbers
- Include seat_number for each item if specified
- Set total_amount and restaurant_name to null if not clearly stated
- Set confidence_score to 0.8

Return only the JSON structure with all extracted information:"""
        
        return prompt
    
    def generate_ground_truth(self, receipt_text: str, max_retries: int = 3) -> str:
        """
        Use Groq API to generate ground truth JSON string (not parsed)
        
        Args:
            receipt_text: Raw receipt text
            max_retries: Number of retry attempts if parsing fails
            
        Returns:
            Ground truth JSON as string
        """
        prompt = self.create_extraction_prompt(receipt_text)
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating ground truth with Groq (attempt {attempt + 1}/{max_retries})...")
                
                # Generate with Groq API
                response_text = self._generate_text(prompt, max_new_tokens=800)
                
                # Clean the response
                response_text = response_text.strip()
                
                # Remove any markdown formatting
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end <= json_start:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise Exception("No valid JSON found in response")
                
                json_text = response_text[json_start:json_end]
                
                # Validate it's parseable JSON
                parsed_json = json.loads(json_text)
                
                print(f"✅ Successfully generated ground truth")
                return json_text  # Return the JSON string, not parsed
                
            except Exception as e:
                print(f"❌ Generation error (attempt {attempt + 1}): {e}")
                
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate ground truth after {max_retries} attempts: {e}")
                
                time.sleep(10)
    
    def shuffle_and_split(self, data: List[Dict[str, Any]], seed: int = 42) -> tuple:
        """Shuffle data and split into train/val/test with 80/10/10 ratio."""
        print(f"🔀 Shuffling data with seed {seed}...")
        
        # Create a copy and shuffle
        shuffled_data = data.copy()
        random.seed(seed)
        random.shuffle(shuffled_data)
        
        total_size = len(shuffled_data)
        
        # Calculate split indices
        train_end = int(0.8 * total_size)
        val_end = int(0.9 * total_size)
        
        # Split the data
        train_data = shuffled_data[:train_end]
        val_data = shuffled_data[train_end:val_end]
        test_data = shuffled_data[val_end:]
        
        print(f"📊 Split summary:")
        print(f"   Train: {len(train_data)} examples ({len(train_data)/total_size*100:.1f}%)")
        print(f"   Val: {len(val_data)} examples ({len(val_data)/total_size*100:.1f}%)")
        print(f"   Test: {len(test_data)} examples ({len(test_data)/total_size*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def save_split(self, split_data: List[Dict[str, Any]], filename: str, split_type: str) -> None:
        """Save a data split to JSON file."""
        output_data = {
            "metadata": {
                "split_type": split_type,
                "total_examples": len(split_data),
                "split_ratio": "80/10/10",
                "created_at": datetime.now().isoformat(),
                "random_seed": 42
            },
            "data": split_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(split_data)} examples to {filename}")
    
    def process_receipt_file_to_training_splits(self, input_file: str, output_dir: str = "./training_data"):
        """
        Process receipts and generate train/test/val JSON files
        
        Args:
            input_file: Path to input file with receipts separated by "--- NEXT OCCURRENCE ---"
            output_dir: Directory to save the split JSON files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Split the input file
        receipts = self.split_receipt_file(input_file)
        
        # Process each receipt and create training examples
        training_examples = []
        failed_receipts = []
        
        for i, receipt_text in enumerate(receipts):
            print(f"\n[{i+1}/{len(receipts)}] Processing receipt {i+1}...")
            print(f"Receipt length: {len(receipt_text)} characters")
            
            try:
                # Create the input prompt (instruction + receipt text)
                extraction_prompt = self.create_extraction_prompt(receipt_text)
                
                # Generate ground truth JSON string
                target_json = self.generate_ground_truth(receipt_text)
                
                # Create training example in the format you showed
                training_example = {
                    "file_id": str(i+1),
                    "input": extraction_prompt,
                    "target": target_json,
                    "input_length": len(extraction_prompt),
                    "target_length": len(target_json)
                }
                
                training_examples.append(training_example)
                
                print(f"✅ Created training example {i+1}")
                print(f"   Input length: {training_example['input_length']}")
                print(f"   Target length: {training_example['target_length']}")
                
            except Exception as e:
                print(f"❌ Failed to process receipt {i+1}: {e}")
                failed_receipts.append(i+1)
                continue
        
        if not training_examples:
            raise Exception("No receipts were successfully processed!")
        
        # Shuffle and split the data
        train_data, val_data, test_data = self.shuffle_and_split(training_examples)
        
        # Save the three splits
        self.save_split(train_data, output_path / "train.json", "train")
        self.save_split(val_data, output_path / "val.json", "val")
        self.save_split(test_data, output_path / "test.json", "test")
        
        # Summary statistics
        successful = len(training_examples)
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"✅ Total receipts processed: {len(receipts)}")
        print(f"🎯 Successful extractions: {successful}/{len(receipts)}")
        if failed_receipts:
            print(f"❌ Failed receipts: {failed_receipts}")
        print(f"📁 Training files saved to: {output_path}")
        print(f"   - train.json: {len(train_data)} examples")
        print(f"   - val.json: {len(val_data)} examples") 
        print(f"   - test.json: {len(test_data)} examples")
        
        return len(training_examples)


def main():
    """
    Process receipts and generate train/test/val JSON files
    """
    # Initialize processor with Groq API
    processor = GroqReceiptProcessor(
        model_name="llama3-70b-8192",
    )
    
    # Process your receipt file and create training splits
    input_file = "tcp_data_mod2.txt"  # Your input file
    output_directory = "./training_splits"
    
    print("🚀 Starting receipt processing for training data generation...")
    
    try:
        # Process the receipts and create train/test/val splits
        total_examples = processor.process_receipt_file_to_training_splits(
            input_file=input_file,
            output_dir=output_directory
        )
        
        print(f"\n🎉 Processing complete! Generated {total_examples} training examples.")
        print(f"📁 Check '{output_directory}' for train.json, val.json, and test.json files.")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print("💡 Make sure you have:")
        print("   - Set GROQ_API_KEY environment variable")
        print("   - Valid Groq API key with sufficient credits")
        print("   - Internet connection for API access")


if __name__ == "__main__":
    main()