import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import sqlite3
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

# Data Models (same as before)
class OrderItem(BaseModel):
    item_name: str
    quantity: int = 1
    modifiers: List[str] = []
    price: Optional[float] = None

class ReceiptData(BaseModel):
    customer_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    order_id: Optional[str] = None
    check_number: Optional[str] = None
    table_number: Optional[str] = None
    pickup_time: Optional[str] = None
    order_items: List[OrderItem] = []
    total_amount: Optional[float] = None
    restaurant_name: Optional[str] = None
    confidence_score: float = 0.0

class LocalReceiptParser:
    def __init__(self, 
                 model_name: str = "google/flan-t5-base",
                 use_pipeline: bool = True,
                 device: str = "auto"):
        """
        Initialize the receipt parser with local FLAN-T5 model
        
        Args:
            model_name: Hugging Face model to use locally (default: google/flan-t5-base)
            use_pipeline: Whether to use pipeline API (simpler) or direct model loading
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.model_name = model_name
        self.use_pipeline = use_pipeline
        self.db_path = "receipts.db"
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🚀 Initializing local receipt parser...")
        print(f"📱 Using device: {self.device}")
        print(f"🤖 Loading model: {self.model_name}")
        
        # Load model
        self._load_model()
        self._init_database()
        
        print(f"✅ Model loaded successfully!")
    
    def _load_model(self):
        """Load the local FLAN-T5 model"""
        try:
            if self.use_pipeline:
                # Use pipeline (simpler approach)
                self.pipe = pipeline(
                    "text2text-generation", 
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,  # 0 for GPU, -1 for CPU
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.tokenizer = None
                self.model = None
            else:
                # Load model and tokenizer directly (more control)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.pipe = None
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try installing: pip install transformers torch")
            raise
    
    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        try:
            if self.use_pipeline:
                # Use pipeline
                result = self.pipe(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=False,  # Deterministic for consistent parsing
                    temperature=0.1,
                    pad_token_id=self.pipe.tokenizer.eos_token_id
                )
                return result[0]['generated_text']
            else:
                # Use model directly
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def _init_database(self):
        """Initialize SQLite database for storing parsed receipts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create receipts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT,
                date TEXT,
                time TEXT,
                order_id TEXT,
                check_number TEXT,
                table_number TEXT,
                pickup_time TEXT,
                total_amount REAL,
                restaurant_name TEXT,
                confidence_score REAL,
                raw_text TEXT,
                parsed_data TEXT,
                processing_time_seconds REAL,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create order items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER,
                item_name TEXT,
                quantity INTEGER,
                modifiers TEXT,
                price REAL,
                FOREIGN KEY (receipt_id) REFERENCES receipts (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_extraction_prompt(self, receipt_text: str) -> str:
        """Create a structured prompt optimized for FLAN-T5 model"""
        
        # FLAN-T5 works better with clear, specific instructions
        prompt = f"""Extract structured information from this receipt and return only valid JSON:

Receipt: {receipt_text}

Extract these fields:
- customer_name: Name between !! marks (e.g., !!John!! → "John")
- date: Date in MM/DD/YY format
- time: Time in HH:MM am/pm format  
- check_number: Number after "Check#:" or similar
- table_number: Table number if mentioned
- pickup_time: Pickup time if specified
- order_items: List of food items with quantities and modifiers
- total_amount: Total cost if shown
- restaurant_name: Restaurant name if present

Return only this JSON structure:
{{"customer_name": null, "date": null, "time": null, "order_id": null, "check_number": null, "table_number": null, "pickup_time": null, "order_items": [{{"item_name": "", "quantity": 1, "modifiers": [], "price": null}}], "total_amount": null, "restaurant_name": null, "confidence_score": 0.8}}

JSON:"""
        
        return prompt
    
    def parse_receipt(self, receipt_text: str) -> ReceiptData:
        """
        Parse a receipt using local FLAN-T5 model
        
        Args:
            receipt_text: Raw receipt text to parse
            
        Returns:
            ReceiptData object with extracted information
        """
        start_time = time.time()
        
        try:
            # Create the prompt
            prompt = self._create_extraction_prompt(receipt_text)
            
            # Generate response with local model
            print(f"🤖 Processing with local {self.model_name}...")
            response_text = self._generate_text(prompt, max_length=512)
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing took {processing_time:.2f} seconds")
            
            # Clean the response
            response_text = response_text.strip()
            
            # Remove prompt echo (FLAN-T5 sometimes repeats the prompt)
            if "JSON:" in response_text:
                response_text = response_text.split("JSON:")[-1].strip()
            
            # Remove any markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text.strip()
            
            # Clean up JSON
            json_text = re.sub(r'\n\s*', ' ', json_text)
            json_text = re.sub(r',\s*}', '}', json_text)
            json_text = re.sub(r',\s*]', ']', json_text)
            
            print(f"🔍 Extracted JSON: {json_text[:200]}...")
            
            # Parse JSON response
            parsed_json = json.loads(json_text)
            
            # Validate and create ReceiptData object
            receipt_data = ReceiptData(**parsed_json)
            
            # Calculate confidence score based on extraction quality
            extracted_fields = sum([
                1 for field in [receipt_data.customer_name, receipt_data.date, 
                               receipt_data.time, receipt_data.check_number]
                if field is not None
            ])
            has_items = len(receipt_data.order_items) > 0
            
            if has_items:
                extracted_fields += 1
            
            # Adjust confidence based on extraction completeness
            if extracted_fields >= 4:
                receipt_data.confidence_score = max(receipt_data.confidence_score, 0.9)
            elif extracted_fields >= 2:
                receipt_data.confidence_score = max(receipt_data.confidence_score, 0.7)
            else:
                receipt_data.confidence_score = min(receipt_data.confidence_score, 0.5)
            
            print(f"✅ Parsed successfully (confidence: {receipt_data.confidence_score:.2f})")
            return receipt_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {response_text[:300]}...")
            return self._fallback_parsing(receipt_text, response_text)
            
        except ValidationError as e:
            print(f"❌ Data validation error: {e}")
            return ReceiptData(confidence_score=0.0)
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Parsing error after {processing_time:.2f}s: {e}")
            return ReceiptData(confidence_score=0.0)
    
    def _fallback_parsing(self, receipt_text: str, response_text: str) -> ReceiptData:
        """Fallback parsing method when JSON parsing fails"""
        print("🔄 Attempting fallback parsing...")
        
        # Basic regex-based extraction as fallback
        receipt_data = ReceiptData(confidence_score=0.3)
        
        # Extract customer name between !! marks
        customer_match = re.search(r'!!\s*([^!]+?)\s*!!', receipt_text, re.IGNORECASE)
        if customer_match:
            receipt_data.customer_name = customer_match.group(1).strip()
        
        # Extract date
        date_match = re.search(r'Date\s+(\d{2}/\d{2}/\d{2})', receipt_text)
        if date_match:
            receipt_data.date = date_match.group(1)
        
        # Extract time
        time_match = re.search(r'Time\s+(\d{1,2}:\d{2}\s*[ap]m)', receipt_text, re.IGNORECASE)
        if time_match:
            receipt_data.time = time_match.group(1)
        
        # Extract check number
        check_match = re.search(r'Check#?\s*:?\s*(\d+)', receipt_text, re.IGNORECASE)
        if check_match:
            receipt_data.check_number = check_match.group(1)
        
        # Extract pickup time
        pickup_match = re.search(r'Pick up Time\s*(\d{1,2}:\d{2}\s*[ap]m)', receipt_text, re.IGNORECASE)
        if pickup_match:
            receipt_data.pickup_time = pickup_match.group(1)
        
        print(f"🔄 Fallback parsing completed (confidence: {receipt_data.confidence_score})")
        return receipt_data
    
    def save_to_database(self, receipt_data: ReceiptData, raw_text: str, processing_time: float = 0.0) -> int:
        """
        Save parsed receipt data to database
        
        Args:
            receipt_data: Parsed receipt data
            raw_text: Original receipt text
            processing_time: Time taken to process
            
        Returns:
            Database ID of saved receipt
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert receipt record
            cursor.execute('''
                INSERT INTO receipts (
                    customer_name, date, time, order_id, check_number, 
                    table_number, pickup_time, total_amount, restaurant_name,
                    confidence_score, raw_text, parsed_data, processing_time_seconds, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                receipt_data.customer_name,
                receipt_data.date,
                receipt_data.time,
                receipt_data.order_id,
                receipt_data.check_number,
                receipt_data.table_number,
                receipt_data.pickup_time,
                receipt_data.total_amount,
                receipt_data.restaurant_name,
                receipt_data.confidence_score,
                raw_text,
                receipt_data.model_dump_json(),
                processing_time,
                self.model_name
            ))
            
            receipt_id = cursor.lastrowid
            
            # Insert order items
            for item in receipt_data.order_items:
                cursor.execute('''
                    INSERT INTO order_items (receipt_id, item_name, quantity, modifiers, price)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    receipt_id,
                    item.item_name,
                    item.quantity,
                    json.dumps(item.modifiers),
                    item.price
                ))
            
            conn.commit()
            return receipt_id
            
        except Exception as e:
            conn.rollback()
            print(f"💾 Database error: {e}")
            return -1
        finally:
            conn.close()
    
    def process_receipt_file(self, file_path: str) -> Dict[str, Any]:
        """Process a receipt file and return results"""
        start_time = time.time()
        
        try:
            # Read receipt text
            with open(file_path, 'r', encoding='utf-8') as f:
                receipt_text = f.read()
            
            print(f"\n📄 Processing: {Path(file_path).name}")
            
            # Parse the receipt
            receipt_data = self.parse_receipt(receipt_text)
            processing_time = time.time() - start_time
            
            # Save to database
            db_id = self.save_to_database(receipt_data, receipt_text, processing_time)
            
            return {
                "success": True,
                "database_id": db_id,
                "confidence_score": receipt_data.confidence_score,
                "processing_time": processing_time,
                "extracted_data": receipt_data.dict(),
                "raw_text_length": len(receipt_text)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "database_id": -1,
                "confidence_score": 0.0,
                "processing_time": processing_time
            }
    
    def batch_process(self, receipt_folder: str) -> List[Dict[str, Any]]:
        """Process multiple receipt files in a folder"""
        results = []
        folder_path = Path(receipt_folder)
        txt_files = list(folder_path.glob("*.txt"))
        
        print(f"\n🔄 Processing {len(txt_files)} receipt files with local {self.model_name}...")
        
        for i, file_path in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] Processing: {file_path.name}")
            
            result = self.process_receipt_file(str(file_path))
            result["filename"] = file_path.name
            results.append(result)
            
            # Show progress
            if result["success"]:
                print(f"✅ Success (confidence: {result['confidence_score']:.2f}, time: {result['processing_time']:.1f}s)")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        avg_confidence = sum(r["confidence_score"] for r in results if r["success"]) / max(successful, 1)
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        
        print(f"\n📊 BATCH SUMMARY:")
        print(f"✅ Successful: {successful}/{len(results)}")
        print(f"🎯 Average confidence: {avg_confidence:.2f}")
        print(f"⏱️ Average processing time: {avg_time:.1f}s")
        
        return results
    
    def get_receipt_data(self, receipt_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve receipt data from database by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM receipts WHERE id = ?
            ''', (receipt_id,))
            
            receipt = cursor.fetchone()
            if not receipt:
                return None
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            receipt_dict = dict(zip(columns, receipt))
            
            # Get order items
            cursor.execute('''
                SELECT item_name, quantity, modifiers, price 
                FROM order_items WHERE receipt_id = ?
            ''', (receipt_id,))
            
            items = cursor.fetchall()
            receipt_dict["order_items"] = [
                {
                    "item_name": item[0],
                    "quantity": item[1],
                    "modifiers": json.loads(item[2]) if item[2] else [],
                    "price": item[3]
                }
                for item in items
            ]
            
            return receipt_dict
            
        except Exception as e:
            print(f"💾 Database retrieval error: {e}")
            return None
        finally:
            conn.close()


# Example usage and testing
def main():
    """Example usage of the LocalReceiptParser with FLAN-T5"""
    
    # Initialize parser with local FLAN-T5 model
    parser = LocalReceiptParser(
        model_name="google/flan-t5-base",
        use_pipeline=True,  # Set to False for direct model loading
        device="auto"  # Will auto-detect best device
    )
    
    # Example receipt text
    sample_receipt = """
Date 06/03/25       Time 11:17am
!!Chelsi!!
Check#:424327
!!Table: !!

!!Pick up Time11:45am!!


����������������������������������������
r!1�� ToGo 1 ����������!
r!1  1 Gyro Sandwich   !
r!1     Lamb           !
rr!1     Skinny Fry     !
r!1  1 Gyro Sandwich   !
r!1     Lamb           !
rr!1     Skinny Fry     !
rr!1    !CN KEITH       !
r����������������������������������������
"""
    
    # Parse the receipt
    print("🚀 Starting receipt parsing with local FLAN-T5...")
    result = parser.parse_receipt(sample_receipt)
    
    # Display results
    print("\n📋 PARSED RECEIPT DATA:")
    print(f"👤 Customer: {result.customer_name}")
    print(f"📅 Date: {result.date}")
    print(f"🕐 Time: {result.time}")
    print(f"🧾 Check#: {result.check_number}")
    print(f"⏰ Pickup Time: {result.pickup_time}")
    print(f"🎯 Confidence: {result.confidence_score}")
    print("\n🍽️ ORDER ITEMS:")
    for item in result.order_items:
        print(f"  • {item.quantity}x {item.item_name}")
        if item.modifiers:
            print(f"    └─ {', '.join(item.modifiers)}")
    
    # Save to database
    db_id = parser.save_to_database(result, sample_receipt)
    print(f"\n💾 Saved to database with ID: {db_id}")
    
    print(f"\n🎉 Successfully processed receipt with local {parser.model_name}!")


if __name__ == "__main__":
    main()