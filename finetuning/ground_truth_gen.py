import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
import time
import requests

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
    
    def _clean_receipt_for_saving(self, receipt_text: str) -> str:
        """
        Clean receipt text for saving to individual files
        """
        # Remove excessive whitespace and normalize line endings
        lines = receipt_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Skip completely empty lines or lines with only special characters
            if not line or all(c in '!���� \t' for c in line):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_extraction_prompt(self, receipt_text: str) -> str:
        """
        Create an enhanced prompt that can handle large receipts with better structure understanding
        """
        # Don't truncate - let the model handle the full receipt
        # Only truncate if extremely long (over 2000 chars)
        if len(receipt_text) > 2000:
            receipt_text = receipt_text[:2000] + "..."
        
        prompt = f"""You are an expert at parsing restaurant receipts. Extract information from this receipt and return ONLY valid JSON.

RECEIPT TEXT:
{receipt_text}

EXTRACTION RULES:
1. Customer name is enclosed in !! marks (e.g., !!CustomerName!)
2. Look for Date, Time, Check#, Table, Pick up Time
3. Parse order items by seat sections [Seat X]
4. Items have quantity, name, and modifiers (indented with 'r' prefix)
5. Extract prices when available
6. Restaurant name may be in header or footer

REQUIRED JSON FORMAT:
{{
  "customer_name": "string or null",
  "date": "string or null", 
  "time": "string or null",
  "check_number": "string or null",
  "table_number": "string or null", 
  "pickup_time": "string or null",
  "order_items": [
    {{
      "seat_number": "string or null",
      "item_name": "string",
      "quantity": "number", 
      "modifiers": ["array of modifier strings"],
      "price": "string or null"
    }}
  ],
  "total_amount": "string or null",
  "restaurant_name": "string or null",
  "confidence_score": 0.8
}}

Return ONLY the JSON object, no other text or explanation."""
        
        return prompt
    
    def generate_ground_truth(self, receipt_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Use Groq API to generate ground truth JSON with enhanced error handling
        
        Args:
            receipt_text: Raw receipt text
            max_retries: Number of retry attempts if parsing fails
            
        Returns:
            Parsed ground truth data as dictionary
        """
        prompt = self.create_extraction_prompt(receipt_text)
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating ground truth with Groq (attempt {attempt + 1}/{max_retries})...")
                
                # Generate with Groq API - increased token limit for complex receipts
                response_text = self._generate_text(prompt, max_new_tokens=800)
                
                # Debug: Show response length
                print(f"🔍 Response length: {len(response_text)} characters")
                
                # Clean the response
                response_text = response_text.strip()
                
                # Check if response is empty
                if not response_text:
                    print(f"❌ Empty response from Groq API (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        raise Exception("Received empty response from Groq API")
                
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
                
                if json_start == -1:
                    print(f"❌ No JSON found in response (attempt {attempt + 1})")
                    print(f"Response content: '{response_text[:300]}...'")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Longer wait for complex receipts
                        continue
                    else:
                        raise Exception("No JSON object found in response")
                
                if json_end <= json_start:
                    print(f"❌ Invalid JSON structure (attempt {attempt + 1})")
                    print(f"Response content: '{response_text[:300]}...'")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise Exception("Invalid JSON structure in response")
                
                json_text = response_text[json_start:json_end]
                
                # Clean up JSON formatting
                json_text = re.sub(r'\n\s*', ' ', json_text)
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                
                print(f"🔧 Extracted JSON length: {len(json_text)} chars")
                
                # Parse JSON
                parsed_json = json.loads(json_text)
                
                # Enhanced validation and defaults
                required_fields = ['customer_name', 'date', 'time', 'check_number', 
                                 'table_number', 'pickup_time', 'order_items', 
                                 'total_amount', 'restaurant_name', 'confidence_score']
                
                for field in required_fields:
                    if field not in parsed_json:
                        if field == 'order_items':
                            parsed_json[field] = []
                        elif field == 'confidence_score':
                            parsed_json[field] = 0.5
                        else:
                            parsed_json[field] = None
                
                # Ensure order_items is a list and validate structure
                if not isinstance(parsed_json.get('order_items'), list):
                    parsed_json['order_items'] = []
                
                # Validate each order item has required fields
                for item in parsed_json['order_items']:
                    if not isinstance(item, dict):
                        continue
                    
                    item_fields = ['item_name', 'quantity', 'modifiers', 'price']
                    for field in item_fields:
                        if field not in item:
                            if field == 'modifiers':
                                item[field] = []
                            elif field == 'quantity':
                                item[field] = 1
                            else:
                                item[field] = None
                    
                    # Ensure modifiers is a list
                    if not isinstance(item.get('modifiers'), list):
                        item['modifiers'] = []
                
                print(f"✅ Successfully generated ground truth with {len(parsed_json['order_items'])} items")
                return parsed_json
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error (attempt {attempt + 1}): {e}")
                print(f"Problematic JSON content (first 500 chars):")
                print(f"'{json_text[:500] if 'json_text' in locals() else 'N/A'}'")
                print("-" * 50)
                
                # If this is the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to parse JSON after {max_retries} attempts: {e}")
                
            except Exception as e:
                print(f"❌ Generation error (attempt {attempt + 1}): {e}")
                
                # If this is the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate ground truth after {max_retries} attempts: {e}")
                
            # Wait before retry - longer for complex receipts
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 10 seconds before retry...")
                time.sleep(10)
    
    def process_receipt_file(self, input_file: str, output_dir: str = "./receipt_data"):
        """
        Process a file containing multiple receipts and generate training data
        
        Args:
            input_file: Path to input file with receipts separated by "--- NEXT OCCURRENCE ---"
            output_dir: Directory to save outputs
        """
        # Create output directories
        output_path = Path(output_dir)
        inputs_dir = output_path / "inputs"
        ground_truths_dir = output_path / "ground_truths"
        
        inputs_dir.mkdir(parents=True, exist_ok=True)
        ground_truths_dir.mkdir(parents=True, exist_ok=True)
        
        # Split the input file
        receipts = self.split_receipt_file(input_file)
        
        # Process each receipt
        results = []
        failed_receipts = []
        
        for i, receipt_text in enumerate(receipts):
            print(f"\n[{i+1}/{len(receipts)}] Processing receipt {i+1}...")
            print(f"Receipt length: {len(receipt_text)} characters")
            
            try:
                # Save individual receipt to input file
                input_filename = f"input_{i+1}.txt"
                input_path = inputs_dir / input_filename
                
                # Clean the receipt text before saving
                cleaned_receipt = self._clean_receipt_for_saving(receipt_text)
                
                with open(input_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(cleaned_receipt)
                
                # Generate ground truth
                ground_truth = self.generate_ground_truth(receipt_text)
                
                # Save ground truth as JSON
                json_filename = f"ground_truth_{i+1}.json"
                json_path = ground_truths_dir / json_filename
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(ground_truth, f, indent=2)
                
                # Create DataFrame row
                df_row = {
                    'receipt_id': i+1,
                    'input_file': input_filename,
                    'customer_name': ground_truth['customer_name'],
                    'date': ground_truth['date'],
                    'time': ground_truth['time'],
                    'check_number': ground_truth['check_number'],
                    'table_number': ground_truth['table_number'],
                    'pickup_time': ground_truth['pickup_time'],
                    'num_items': len(ground_truth['order_items']),
                    'order_items_json': json.dumps(ground_truth['order_items']),
                    'total_amount': ground_truth['total_amount'],
                    'restaurant_name': ground_truth['restaurant_name'],
                    'confidence_score': ground_truth['confidence_score']
                }
                
                results.append(df_row)
                
                print(f"✅ Saved input_{i+1}.txt and ground_truth_{i+1}.json")
                print(f"   Customer: {ground_truth['customer_name']}")
                print(f"   Items: {len(ground_truth['order_items'])}")
                print(f"   Confidence: {ground_truth['confidence_score']}")
                
            except Exception as e:
                print(f"❌ Failed to process receipt {i+1}: {e}")
                failed_receipts.append(i+1)
                continue
        
        if not results:
            raise Exception("No receipts were successfully processed!")
        
        # Create summary DataFrame
        df = pd.DataFrame(results)
        df_path = output_path / "ground_truth_summary.csv"
        df.to_csv(df_path, index=False)
        
        # Create detailed items DataFrame
        items_data = []
        for i, receipt_result in enumerate(results):
            receipt_id = receipt_result['receipt_id']
            items = json.loads(receipt_result['order_items_json'])
            
            for j, item in enumerate(items):
                items_data.append({
                    'receipt_id': receipt_id,
                    'item_index': j+1,
                    'seat_number': item.get('seat_number'),
                    'item_name': item.get('item_name'),
                    'quantity': item.get('quantity'),
                    'modifiers': json.dumps(item.get('modifiers', [])),
                    'price': item.get('price')
                })
        
        items_df = pd.DataFrame(items_data)
        items_df_path = output_path / "order_items_detail.csv"
        items_df.to_csv(items_df_path, index=False)
        
        # Summary statistics
        successful = len(results)
        avg_confidence = sum(r['confidence_score'] for r in results) / len(results) if results else 0
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"✅ Total receipts processed: {len(receipts)}")
        print(f"🎯 Successful extractions: {successful}/{len(receipts)}")
        if failed_receipts:
            print(f"❌ Failed receipts: {failed_receipts}")
        print(f"📈 Average confidence: {avg_confidence:.2f}")
        print(f"📁 Files saved to: {output_path}")
        print(f"   - Individual inputs: {inputs_dir}")
        print(f"   - Ground truth JSONs: {ground_truths_dir}")
        print(f"   - Summary CSV: {df_path}")
        print(f"   - Items detail CSV: {items_df_path}")
        
        return df, items_df


def main():
    """
    Example usage of the enhanced Groq receipt processor
    """
    # Initialize processor with Groq API - using 70b model for better complex parsing
    processor = GroqReceiptProcessor(
        model_name="llama3-70b-8192",  # Better for complex parsing
        # Alternative: "llama-3.1-8b-instant" (faster but less accurate for complex receipts)
    )
    
    # Process your receipt file
    input_file = "tcp_data_mod1.txt"  # Your input file
    output_directory = "./receipt_training_data"
    
    print("🚀 Starting enhanced receipt processing with Groq API...")
    
    try:
        # Process the receipts
        summary_df, items_df = processor.process_receipt_file(
            input_file=input_file,
            output_dir=output_directory
        )
        
        print(f"\n📋 Preview of generated data:")
        print(summary_df.head())
        
        print(f"\n🍽️ Preview of items data:")
        print(items_df.head())
        
        print(f"\n🎉 Processing complete! Check '{output_directory}' for all generated files.")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print("💡 Make sure you have:")
        print("   - Set GROQ_API_KEY environment variable: export GROQ_API_KEY='your_token_here'")
        print("   - Valid Groq API key with sufficient credits")
        print("   - Internet connection for API access")
        print("   - Required libraries: pip install requests pandas python-dotenv")


if __name__ == "__main__":
    main()