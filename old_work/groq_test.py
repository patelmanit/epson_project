import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import requests
import sqlite3
from pathlib import Path
import time
from dotenv import load_dotenv
import os

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

class GroqReceiptParser:
    def __init__(self, 
                 api_key: str,
                 model: str = "llama-3.1-8b-instruct",
                 groq_base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize the receipt parser with Groq API
        
        Args:
            api_key: Your Groq API key (required)
            model: Groq model to use (llama-3.1-8b-instant, llama-3.1-70b-versatile, etc.)
            groq_base_url: Base URL for Groq API
        """
        if not api_key:
            raise ValueError("Groq API key is required")
            
        self.api_key = api_key
        self.groq_base_url = groq_base_url.rstrip('/')
        self.model = model
        self.db_path = "receipts.db"
        self._init_database()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Groq service"""
        try:
            response = self._make_groq_request("Hello", max_tokens=10)
            print(f"✅ Connected to Groq API")
            print(f"✅ Using model: {self.model}")
        except Exception as e:
            print(f"❌ Failed to connect to Groq: {e}")
            print("Make sure your API key is valid and you have credits")
    
    def _make_groq_request(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Make request to Groq API using OpenAI-compatible format
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Prepare request payload (OpenAI chat format)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for consistent outputs
            "top_p": 0.9,
            "stop": ["</json>", "```"]  # Stop tokens to prevent over-generation
        }
        
        # Set headers with API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Make the request to chat/completions endpoint
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60  # 60 second timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request timed out - model may be slow or unavailable")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to {self.groq_base_url}")
        except requests.exceptions.HTTPError as e:
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
        """Create a structured prompt optimized for Llama models"""
        
        # Llama models respond better to clear, direct instructions
        prompt = f"""<receipt_parsing_task>
You are an expert receipt parser. Extract structured data from this kitchen receipt and return ONLY valid JSON.

RECEIPT TEXT:
{receipt_text}

EXTRACT THESE FIELDS:
- customer_name: Customer's name (look for names between !! marks or similar)
- date: Date in MM/DD/YY format
- time: Time with am/pm
- check_number: Number after "Check#:" 
- table_number: Table number if mentioned
- pickup_time: Pickup time if specified
- order_items: Array of food items with quantities and modifiers

PARSING RULES:
1. Clean item names - remove dots, special characters, formatting symbols
2. Group modifiers (like "Lamb", "Skinny Fry") with their main items
3. Extract quantities (default to 1 if not specified)
4. Ignore system codes and formatting artifacts
5. Be confident - assign confidence_score 0.9+ for clear extractions

REQUIRED JSON FORMAT (return this exact structure):
{{
    "customer_name": "extracted_name_or_null",
    "date": "MM/DD/YY_or_null", 
    "time": "HH:MM_am/pm_or_null",
    "order_id": "order_id_or_null",
    "check_number": "check_number_or_null",
    "table_number": "table_or_null",
    "pickup_time": "pickup_time_or_null",
    "order_items": [
        {{
            "item_name": "clean_item_name",
            "quantity": 1,
            "modifiers": ["modifier1", "modifier2"],
            "price": null
        }}
    ],
    "total_amount": null,
    "restaurant_name": null,
    "confidence_score": 0.95
}}

Return ONLY the JSON, no explanations or markdown:
</receipt_parsing_task>"""
        
        return prompt
    
    def parse_receipt(self, receipt_text: str) -> ReceiptData:
        """
        Parse a receipt using Groq API
        
        Args:
            receipt_text: Raw receipt text to parse
            
        Returns:
            ReceiptData object with extracted information
        """
        start_time = time.time()
        
        try:
            # Create the prompt
            prompt = self._create_extraction_prompt(receipt_text)
            
            # Call Groq API
            print(f"🤖 Processing with {self.model}...")
            response_text = self._make_groq_request(prompt)
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing took {processing_time:.2f} seconds")
            
            # Clean the response
            response_text = response_text.strip()
            
            # Remove any markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Find JSON in the response (in case model adds extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # Parse JSON response
            parsed_json = json.loads(json_text)
            
            # Validate and create ReceiptData object
            receipt_data = ReceiptData(**parsed_json)
            
            print(f"✅ Parsed successfully (confidence: {receipt_data.confidence_score:.2f})")
            return receipt_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {response_text[:200]}...")
            return ReceiptData(confidence_score=0.0)
            
        except ValidationError as e:
            print(f"❌ Data validation error: {e}")
            return ReceiptData(confidence_score=0.0)
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Parsing error after {processing_time:.2f}s: {e}")
            return ReceiptData(confidence_score=0.0)
    
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
                receipt_data.json(),
                processing_time,
                self.model
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
        
        print(f"\n🔄 Processing {len(txt_files)} receipt files...")
        
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
    
    def switch_model(self, new_model: str):
        """
        Switch to a different Groq model
        
        Args:
            new_model: New model name (e.g., "llama-3.1-70b-versatile", "mixtral-8x7b-32768")
        """
        self.model = new_model
        print(f"🔄 Switched to model: {self.model}")
        self._test_connection()
    
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
    """Example usage of the GroqReceiptParser"""
    
    # Initialize parser with Groq API
    # You need to set your Groq API key here
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("❌ Please set your Groq API key in the GROQ_API_KEY variable")
        return
    
    parser = GroqReceiptParser(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"  # Fast and cost-effective
        # Alternative models: "llama-3.1-70b-versatile", "mixtral-8x7b-32768"
    )
    
    # Example receipt text (from your samples)
    sample_receipt = """
Date 06/03/25       Time 11:17am
!!Chelsi!
Check#:424327
!!Table: !

!!Pick up Time11:45am!


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
    result = parser.parse_receipt(sample_receipt)
    
    # Display results
    print("\n📋 PARSED RECEIPT DATA:")
    print(f"👤 Customer: {result.customer_name}")
    print(f"📅 Date: {result.date}")
    print(f"🕐 Time: {result.time}")
    print(f"🧾 Check#: {result.check_number}")
    print(f"🎯 Confidence: {result.confidence_score}")
    print("\n🍽️ ORDER ITEMS:")
    for item in result.order_items:
        print(f"  • {item.quantity}x {item.item_name}")
        if item.modifiers:
            print(f"    └─ {', '.join(item.modifiers)}")
    
    # Save to database
    db_id = parser.save_to_database(result, sample_receipt)
    print(f"\n💾 Saved to database with ID: {db_id}")
    
    # Example: Switch to a more powerful model for better accuracy
    # parser.switch_model("llama-3.1-70b-versatile")


if __name__ == "__main__":
    main()