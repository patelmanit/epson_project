#!/usr/bin/env python3
"""
Simple Receipt Text Cleaner
Only removes obvious OCR noise while preserving ALL receipt data
"""

import re
import json
from typing import Dict, List

class SimpleReceiptCleaner:
    """Conservative cleaner that only removes clear OCR garbage"""
    
    def clean_text(self, text: str) -> str:
        """Clean OCR noise while preserving receipt structure"""
        
        # 1. Remove Microsoft certificate garbage (very specific patterns)
        text = re.sub(r'ih\?[^!]*?microsoft\.com[^!]*?(?=Date|Time|Check|Table|\n|$)', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'HTTP/1\.1[^!]*?Connection:[^!]*?(?=Date|Time|Check|Table|\n|$)', ' ', text)
        text = re.sub(r'DigiCert[^!]*?CA \d+', ' ', text)
        text = re.sub(r'RSA TLS[^!]*?(?=Date|Time|Check|Table|\n|$)', ' ', text)
        
        # 2. Remove very long hex/base64 strings (40+ chars) - never receipt data
        text = re.sub(r'[0-9A-F]{40,}', ' ', text)
        text = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', ' ', text)
        
        # 3. Remove specific garbage unicode characters (keep normal letters/names)
        text = re.sub(r'[ɖɥμάώĢƝƖ�]{2,}', ' ', text)
        
        # 4. Clean up modifier prefixes but preserve structure
        text = text.replace('rr!1', ' ')  # Remove indent markers
        text = text.replace('r!1', ' ')   # Remove indent markers
        
        # 5. Minimal whitespace cleanup
        text = re.sub(r'[ \t]{3,}', ' ', text)      # 3+ spaces to 1
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 3+ newlines to 2
        text = text.strip()
        
        return text
    
    def process_training_file(self, input_file: str, output_file: str):
        """Process your training JSON file"""
        
        print(f"🧹 Cleaning {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = []
        
        for i, item in enumerate(data):
            print(f"  Processing {i+1}/{len(data)}")
            
            # Extract receipt text from your format
            original_input = item['input']
            
            if "Receipt Text:" in original_input:
                # Find the receipt section
                start = original_input.find("Receipt Text:") + len("Receipt Text:")
                if "\n\nInstructions:" in original_input:
                    end = original_input.find("\n\nInstructions:")
                    receipt_text = original_input[start:end].strip()
                else:
                    receipt_text = original_input[start:].strip()
                
                # Clean the receipt text
                cleaned_receipt = self.clean_text(receipt_text)
                
                # Replace in the full input
                new_input = original_input.replace(receipt_text, cleaned_receipt)
            else:
                # Simple case - clean the whole input
                new_input = self.clean_text(original_input)
            
            # Create cleaned item
            cleaned_item = {
                "file_id": item.get('file_id', f"cleaned_{i}"),
                "input": new_input,
                "target": item['target'],
                "input_length": len(new_input),
                "target_length": len(item['target']),
                "original_length": len(original_input),
                "reduction": len(original_input) - len(new_input)
            }
            
            cleaned_data.append(cleaned_item)
        
        # Save cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Stats
        total_reduction = sum(item['reduction'] for item in cleaned_data)
        avg_reduction = total_reduction / len(cleaned_data)
        
        print(f"✅ Saved to {output_file}")
        print(f"📊 Average noise reduction: {avg_reduction:.0f} chars per sample")
        print(f"📊 Total noise removed: {total_reduction} characters")

def test_cleaner():
    """Test with your actual noisy data"""
    
    test_sample = """Date 06/03/25       Time 11:45am
!!Aaliyah!
Check#:424338
!!Table: W-5       !
!!Pick up TimeN/A!
!1[Seat 1]�����������������q��S�Ɩw�W�7
�; ��[P���l���C��B�����Xբ�õur��DwP�έ0ݕeg���K�	��\���{<��y���
��2�ܥdc!
!1  1 Boneless Wings  !
r!1     Garlic + Parme !
rr!1     Ranch          !
r!1  1 Diet Pepsi      !
!1[Seat 2]����������!
!1  1 Wings 10        !
r!1     Cajun          !
rr!1     Ranch          !
rr!1     Triple Time    !
r!1  1 Diet Pepsi      !
itM���w���!�=��֑jA��:u����W����ٖL#$��"�'�O�@7z��p�"""
    
    cleaner = SimpleReceiptCleaner()
    cleaned = cleaner.clean_text(test_sample)
    
    print("🧪 TESTING CLEANER")
    print("=" * 40)
    print(f"Original length: {len(test_sample)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Reduction: {len(test_sample) - len(cleaned)} chars")
    print("\nCLEANED TEXT:")
    print(cleaned)
    
    # Verify important data preserved
    important_data = ['Aaliyah', 'W-5', '424338', 'Boneless Wings', 'Diet Pepsi']
    preserved = [item for item in important_data if item in cleaned]
    print(f"\n✅ Preserved data: {preserved}")
    
    if len(preserved) == len(important_data):
        print("✅ All important data preserved!")
    else:
        print(f"⚠️  Some data may be lost: {set(important_data) - set(preserved)}")

def main():
    """Main function"""
    print("🧹 Simple Receipt Text Cleaner")
    print("=" * 40)
    
    # Test first
    test_cleaner()
    
    print("\n" + "=" * 40)
    
    # Process files
    cleaner = SimpleReceiptCleaner()
    
    if input("\nProcess train.json? (y/n): ").lower() == 'y':
        try:
            cleaner.process_training_file("train.json", "train_cleaned.json")
        except FileNotFoundError:
            print("❌ train.json not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if input("Process val.json? (y/n): ").lower() == 'y':
        try:
            cleaner.process_training_file("val.json", "val_cleaned.json") 
        except FileNotFoundError:
            print("❌ val.json not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Done! Use the *_cleaned.json files for training your T5 parser.")

if __name__ == "__main__":
    main()