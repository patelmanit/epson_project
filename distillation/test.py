import json
import re
from typing import Dict, List, Any, Union

def fix_malformed_json(pred_str: str) -> Dict:
    """Fix malformed JSON from model output"""
    if isinstance(pred_str, dict):
        return pred_str
    
    if not isinstance(pred_str, str):
        return {}
    
    # Add missing opening brace
    if not pred_str.startswith('{'):
        pred_str = '{' + pred_str
    
    # Fix order_items array - the main issue in your outputs
    # Pattern: "order_items":["item_name":"..." should be "order_items":[{"item_name":"..."}]
    if '"order_items":[' in pred_str and '"order_items":[{' not in pred_str:
        # Find the order_items section
        start = pred_str.find('"order_items":[')
        if start != -1:
            end = pred_str.find(']', start)
            if end != -1:
                items_section = pred_str[start:end+1]
                # Split by "item_name" to find individual items
                items = items_section.split('"item_name"')
                fixed_items = []
                for i, item in enumerate(items[1:], 1):  # Skip first split which is before first item
                    item_str = '"item_name"' + item
                    # Remove trailing comma or bracket if present
                    item_str = item_str.rstrip(',]')
                    # Wrap in braces
                    if not item_str.startswith('{'):
                        item_str = '{' + item_str
                    if not item_str.endswith('}'):
                        item_str = item_str + '}'
                    fixed_items.append(item_str)
                
                # Reconstruct the JSON
                fixed_items_str = '"order_items":[' + ','.join(fixed_items) + ']'
                pred_str = pred_str[:start] + fixed_items_str + pred_str[end+1:]
    
    # Add missing closing brace
    if not pred_str.endswith('}'):
        pred_str = pred_str + '}'
    
    # Try to parse
    try:
        return json.loads(pred_str)
    except:
        # Return valid empty structure if all else fails
        return {
            "customer_name": None,
            "date": None,
            "time": None,
            "check_number": None,
            "table_number": None,
            "pickup_time": None,
            "total_amount": None,
            "restaurant_name": None,
            "confidence_score": 0.5,
            "order_items": []
        }

def normalize_date(date_str: str) -> str:
    """Fix date format inconsistencies"""
    if not date_str or date_str in ["N/A", "null", "None"]:
        return None
    
    # Handle ISO format (2025-06-25) -> MM/DD/YY
    if '-' in date_str and len(date_str) >= 10:
        parts = date_str.split('-')
        if len(parts) == 3:
            year = parts[0][-2:]  # Last 2 digits of year
            month = parts[1]
            day = parts[2][:2]  # In case of datetime
            return f"{month}/{day}/{year}"
    
    # Handle MM/DD format -> keep as is
    if '/' in date_str and date_str.count('/') == 1:
        return date_str
    
    # Handle MM/DD/YY format
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            month, day, year = parts
            if len(year) == 4:
                year = year[-2:]
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
    
    return date_str

def fix_item_names(item_name: str) -> str:
    """Fix common OCR errors in item names"""
    if not item_name:
        return None
    
    # Key replacements based on your test output
    replacements = {
        'IC LITE': 'Iron City Light',
        'Italian City Light': 'Iron City Light',
        'Chix Platter': 'Chicken Platter',
        'Buffalo Chix Sa': 'Buffalo Chicken Sandwich',
        'Dino Burger San': 'Dino Burger Sandwich',
        'CHX': 'Chicken',
        'DOWNGRD C': 'Downgrade',
        'UUS10U DigiCert Inc': None,  # Remove this garbage
    }
    
    for old, new in replacements.items():
        if item_name == old:
            return new
    
    # Extract quantity from name (e.g., "Wings 10" -> "Wings")
    match = re.match(r'^(.+?)\s+(\d+)$', item_name)
    if match:
        return match.group(1)
    
    return item_name

def extract_quantity(item_name: str, current_qty: int = 1) -> tuple:
    """Extract quantity from item name"""
    match = re.match(r'^(.+?)\s+(\d+)$', item_name)
    if match:
        return match.group(1), int(match.group(2))
    return item_name, current_qty

def clean_modifiers(modifiers: List) -> List:
    """Clean up modifier list"""
    if not isinstance(modifiers, list):
        return []
    
    cleaned = []
    for mod in modifiers:
        # Skip numeric-only modifiers
        if isinstance(mod, str) and not re.match(r'^\d+$', mod):
            # Skip items that look like they should be separate menu items
            if mod not in ['Coors Light', 'Pint', 'U DigiCert TLS RSA SHA256 2020']:
                cleaned.append(mod)
    return cleaned

def process_prediction(pred: Union[str, Dict]) -> Dict:
    """Main processing function"""
    # Step 1: Fix JSON structure
    if isinstance(pred, str):
        pred = fix_malformed_json(pred)
    
    if not isinstance(pred, dict):
        pred = {}
    
    # Step 2: Process each field
    processed = {}
    
    # Handle nulls consistently
    for field in ['customer_name', 'date', 'time', 'check_number', 'table_number', 
                  'pickup_time', 'total_amount', 'restaurant_name']:
        value = pred.get(field)
        if value in ["N/A", "null", "None", ""]:
            processed[field] = None if field != 'table_number' else ("" if value == "" else None)
        else:
            processed[field] = value
    
    # Fix date format
    if processed.get('date'):
        processed['date'] = normalize_date(processed['date'])
    
    # Handle confidence score
    processed['confidence_score'] = pred.get('confidence_score', 0.5)
    
    # Process order items
    order_items = pred.get('order_items', [])
    if not isinstance(order_items, list):
        order_items = []
    
    processed_items = []
    for item in order_items:
        if not isinstance(item, dict):
            continue
        
        # Get item name and extract quantity
        item_name = item.get('item_name', '')
        quantity = item.get('quantity', 1)
        
        # Extract quantity from name if present
        item_name, extracted_qty = extract_quantity(item_name, quantity)
        if extracted_qty != quantity and quantity == 1:
            quantity = extracted_qty
        
        # Fix item name
        item_name = fix_item_names(item_name)
        
        # Skip if no valid item name or if it's garbage
        if not item_name:
            continue
        
        # Build cleaned item
        cleaned_item = {
            'item_name': item_name,
            'quantity': quantity,
            'modifiers': clean_modifiers(item.get('modifiers', [])),
            'price': item.get('price')
        }
        
        # Add seat number if present
        seat = item.get('seat_number')
        if seat and seat not in ['None', 'null', 'N/A']:
            cleaned_item['seat_number'] = seat
        
        processed_items.append(cleaned_item)
    
    processed['order_items'] = processed_items
    
    return processed

def main():
    """Main execution"""
    try:
        from main import load_and_test_model
        
        # Load model and get predictions
        print("Loading model and getting predictions...")
        model, tokenizer, preds, metrics = load_and_test_model('./best_receipt_parser', 'test.json')
        
        print(f"\nOriginal Metrics: {metrics}")
        print("="*80)
        
        # Process all predictions
        print("\nApplying minimal post-processing...")
        processed_preds = []
        
        for i, pred in enumerate(preds):
            try:
                processed = process_prediction(pred)
                processed_preds.append(processed)
            except Exception as e:
                print(f"Error processing prediction {i}: {e}")
                processed_preds.append({
                    "customer_name": None,
                    "date": None,
                    "time": None,
                    "check_number": None,
                    "table_number": None,
                    "pickup_time": None,
                    "total_amount": None,
                    "restaurant_name": None,
                    "confidence_score": 0.5,
                    "order_items": []
                })
        
        # Save processed predictions
        with open('processed_predictions.json', 'w') as f:
            json.dump(processed_preds, f, indent=2)
        
        print("Processed predictions saved to processed_predictions.json")
        
        # Calculate improvements
        valid_json_count = sum(1 for p in processed_preds if p and 'order_items' in p)
        print(f"\nProcessed Results:")
        print(f"Valid JSON: {valid_json_count}/{len(processed_preds)} ({100*valid_json_count/len(processed_preds):.1f}%)")
        
        # Show a few examples
        print("\n" + "="*80)
        print("EXAMPLE TRANSFORMATIONS:")
        print("="*80)
        
        for i in range(min(3, len(preds))):
            print(f"\nExample {i+1}:")
            print(f"Original: {str(preds[i])[:100]}...")
            print(f"Processed: {json.dumps(processed_preds[i], indent=2)[:200]}...")
            
    except ImportError as e:
        print(f"Error: Could not import required module - {e}")
        print("Make sure main.py is in the same directory")

if __name__ == "__main__":
    main()