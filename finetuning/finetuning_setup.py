import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re
from dataclasses import dataclass
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class DatasetStats:
    """Statistics about the dataset"""
    total_examples: int
    avg_input_length: int
    avg_output_length: int
    max_input_length: int
    max_output_length: int
    field_coverage: Dict[str, float]
    item_stats: Dict[str, Any]


class ReceiptDataPreprocessor:
    """
    Preprocessor for receipt data to prepare it for FLAN-T5 fine-tuning
    """
    
    def __init__(self, 
                 data_folder: str = "receipt_training_data",
                 model_name: str = "google/flan-t5-base",
                 max_input_length: int = 512,
                 max_output_length: int = 512):
        """
        Initialize the data preprocessor
        
        Args:
            data_folder: Path to folder containing 'inputs' and 'ground_truths' subfolders
            model_name: Model name for tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.data_folder = Path(data_folder)
        self.inputs_folder = self.data_folder / "inputs"
        self.ground_truths_folder = self.data_folder / "ground_truths"
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load tokenizer for length analysis
        print(f"🤖 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Data storage
        self.raw_data = []
        self.processed_data = []
        self.dataset_stats = None
        
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load all input.txt and ground_truth.json pairs
        
        Returns:
            List of dictionaries containing raw input text and ground truth data
        """
        print(f"📂 Loading data from {self.data_folder}")
        
        # Find all input files
        input_files = list(self.inputs_folder.glob("input_*.txt"))
        print(f"📄 Found {len(input_files)} input files")
        
        raw_data = []
        
        for input_file in sorted(input_files):
            # Extract file number (e.g., input_1.txt -> 1)
            file_num = input_file.stem.split('_')[1]
            
            # Corresponding ground truth file
            ground_truth_file = self.ground_truths_folder / f"ground_truth_{file_num}.json"
            
            if not ground_truth_file.exists():
                print(f"⚠️ Missing ground truth for {input_file.name}")
                continue
            
            try:
                # Load input text
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_text = f.read().strip()
                
                # Load ground truth JSON
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                
                raw_data.append({
                    'file_id': file_num,
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'input_file': str(input_file),
                    'ground_truth_file': str(ground_truth_file)
                })
                
            except Exception as e:
                print(f"❌ Error loading {input_file.name}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(raw_data)} examples")
        self.raw_data = raw_data
        return raw_data
    
    def clean_input_text(self, text: str) -> str:
        """
        Clean and normalize input text
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive special characters while preserving structure
        text = re.sub(r'[^\w\s!@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive exclamation marks but keep some structure
        text = re.sub(r'!{3,}', '!!', text)
        
        # Truncate if too long (rough character-based estimate)
        # 1 token ≈ 4 characters for English text
        max_chars = self.max_input_length * 3  # Conservative estimate
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            print(f"⚠️ Truncated long input text to {max_chars} characters")
        
        return text.strip()
    
    def normalize_ground_truth(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize ground truth data for consistent formatting
        
        Args:
            ground_truth: Raw ground truth dictionary
            
        Returns:
            Normalized ground truth dictionary
        """
        normalized = {}
        
        # Standard fields
        normalized['customer_name'] = ground_truth.get('customer_name')
        normalized['date'] = ground_truth.get('date')
        normalized['time'] = ground_truth.get('time')
        normalized['check_number'] = ground_truth.get('check_number')
        normalized['table_number'] = ground_truth.get('table_number')
        normalized['pickup_time'] = ground_truth.get('pickup_time')
        normalized['total_amount'] = ground_truth.get('total_amount')
        normalized['restaurant_name'] = ground_truth.get('restaurant_name')
        normalized['confidence_score'] = ground_truth.get('confidence_score', 0.8)
        
        # Normalize order items with better error handling
        order_items = []
        raw_items = ground_truth.get('order_items', [])
        
        if isinstance(raw_items, list):
            for i, item in enumerate(raw_items):
                if isinstance(item, dict):
                    # Standard dictionary item
                    normalized_item = {
                        'item_name': item.get('item_name', ''),
                        'quantity': item.get('quantity', 1),
                        'modifiers': item.get('modifiers', []) if isinstance(item.get('modifiers'), list) else [],
                        'price': item.get('price')
                    }
                    
                    # Add seat_number if present (specific to your dataset)
                    if 'seat_number' in item:
                        normalized_item['seat_number'] = str(item['seat_number'])
                    
                    order_items.append(normalized_item)
                    
                elif isinstance(item, str):
                    # Handle string items (create basic structure)
                    print(f"⚠️ Converting string item to dict: {item[:50]}...")
                    normalized_item = {
                        'item_name': item,
                        'quantity': 1,
                        'modifiers': [],
                        'price': None
                    }
                    order_items.append(normalized_item)
                    
                else:
                    print(f"⚠️ Skipping unknown item type {type(item)} at index {i}")
                    continue
        else:
            print(f"⚠️ order_items is not a list: {type(raw_items)}")
        
        normalized['order_items'] = order_items
        
        return normalized
    
    def create_training_prompt(self, input_text: str) -> str:
        """
        Create a training prompt optimized for FLAN-T5
        
        Args:
            input_text: Receipt text
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Extract structured information from this restaurant receipt and return valid JSON.

Receipt Text:
{input_text}

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
    
    def create_target_output(self, ground_truth: Dict[str, Any]) -> str:
        """
        Create target output JSON string for training
        
        Args:
            ground_truth: Normalized ground truth data
            
        Returns:
            JSON string for target output
        """
        # Create a clean JSON output
        return json.dumps(ground_truth, indent=None, separators=(',', ':'))
    
    def analyze_dataset_statistics(self) -> DatasetStats:
        """
        Analyze dataset statistics for insights
        
        Returns:
            DatasetStats object with analysis results
        """
        if not self.raw_data:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        print("📊 Analyzing dataset statistics...")
        
        input_lengths = []
        output_lengths = []
        field_counts = Counter()
        total_items = 0
        modifier_counts = []
        seat_numbers = []
        long_sequences = []
        
        for i, example in enumerate(self.raw_data):
            try:
                # Input length analysis (with error handling)
                try:
                    input_tokens = self.tokenizer.encode(
                        example['input_text'], 
                        max_length=self.max_input_length * 2,  # Allow longer for analysis
                        truncation=True,
                        add_special_tokens=False
                    )
                    input_lengths.append(len(input_tokens))
                    
                    if len(input_tokens) > self.max_input_length:
                        long_sequences.append(f"Input {i+1}: {len(input_tokens)} tokens")
                        
                except Exception as e:
                    print(f"⚠️ Error tokenizing input {i+1}: {e}")
                    input_lengths.append(0)
                
                # Output length analysis (with error handling)
                try:
                    output_json = json.dumps(example['ground_truth'])
                    output_tokens = self.tokenizer.encode(
                        output_json,
                        max_length=self.max_output_length * 2,  # Allow longer for analysis
                        truncation=True,
                        add_special_tokens=False
                    )
                    output_lengths.append(len(output_tokens))
                    
                    if len(output_tokens) > self.max_output_length:
                        long_sequences.append(f"Output {i+1}: {len(output_tokens)} tokens")
                        
                except Exception as e:
                    print(f"⚠️ Error tokenizing output {i+1}: {e}")
                    output_lengths.append(0)
                
                # Field coverage analysis
                gt = example['ground_truth']
                for field in ['customer_name', 'date', 'time', 'check_number', 'table_number', 'pickup_time']:
                    if gt.get(field) is not None and gt.get(field) != 'N/A':
                        field_counts[field] += 1
                
                # Item analysis (with better error handling)
                items = gt.get('order_items', [])
                if isinstance(items, list):
                    total_items += len(items)
                    
                    for item in items:
                        # Handle both dict and string items
                        if isinstance(item, dict):
                            modifiers = item.get('modifiers', [])
                            if isinstance(modifiers, list):
                                modifier_counts.append(len(modifiers))
                            else:
                                modifier_counts.append(0)
                                
                            if 'seat_number' in item:
                                seat_numbers.append(str(item['seat_number']))
                        elif isinstance(item, str):
                            # If item is a string, we can't extract modifiers
                            modifier_counts.append(0)
                            print(f"⚠️ String item found in example {i+1}: {item[:50]}...")
                        else:
                            print(f"⚠️ Unknown item type in example {i+1}: {type(item)}")
                            modifier_counts.append(0)
                else:
                    print(f"⚠️ order_items is not a list in example {i+1}: {type(items)}")
                    
            except Exception as e:
                print(f"⚠️ Error analyzing example {i+1}: {e}")
                continue
        
        # Report long sequences
        if long_sequences:
            print(f"⚠️ Found {len(long_sequences)} sequences exceeding limits:")
            for seq in long_sequences[:5]:  # Show first 5
                print(f"   {seq}")
            if len(long_sequences) > 5:
                print(f"   ... and {len(long_sequences) - 5} more")
        
        # Calculate statistics
        total_examples = len(self.raw_data)
        
        field_coverage = {
            field: count / total_examples 
            for field, count in field_counts.items()
        }
        
        item_stats = {
            'total_items': total_items,
            'avg_items_per_receipt': total_items / total_examples,
            'avg_modifiers_per_item': sum(modifier_counts) / len(modifier_counts) if modifier_counts else 0,
            'seat_numbers_used': len(seat_numbers) > 0,
            'unique_seat_numbers': len(set(seat_numbers)) if seat_numbers else 0
        }
        
        stats = DatasetStats(
            total_examples=total_examples,
            avg_input_length=int(sum(input_lengths) / len(input_lengths)),
            avg_output_length=int(sum(output_lengths) / len(output_lengths)),
            max_input_length=max(input_lengths),
            max_output_length=max(output_lengths),
            field_coverage=field_coverage,
            item_stats=item_stats
        )
        
        self.dataset_stats = stats
        return stats
    
    def print_dataset_analysis(self):
        """Print detailed dataset analysis"""
        if not self.dataset_stats:
            self.analyze_dataset_statistics()
        
        stats = self.dataset_stats
        
        print("\n" + "="*50)
        print("📊 DATASET ANALYSIS REPORT")
        print("="*50)
        
        print(f"\n📈 BASIC STATISTICS:")
        print(f"  Total Examples: {stats.total_examples}")
        print(f"  Average Input Length: {stats.avg_input_length} tokens")
        print(f"  Average Output Length: {stats.avg_output_length} tokens")
        print(f"  Maximum Input Length: {stats.max_input_length} tokens")
        print(f"  Maximum Output Length: {stats.max_output_length} tokens")
        
        print(f"\n🎯 FIELD COVERAGE:")
        for field, coverage in stats.field_coverage.items():
            print(f"  {field}: {coverage:.1%}")
        
        print(f"\n🍽️ ORDER ITEM STATISTICS:")
        print(f"  Total Items: {stats.item_stats['total_items']}")
        print(f"  Average Items per Receipt: {stats.item_stats['avg_items_per_receipt']:.1f}")
        print(f"  Average Modifiers per Item: {stats.item_stats['avg_modifiers_per_item']:.1f}")
        print(f"  Uses Seat Numbers: {stats.item_stats['seat_numbers_used']}")
        print(f"  Unique Seat Numbers: {stats.item_stats['unique_seat_numbers']}")
        
        # Length warnings
        if stats.max_input_length > self.max_input_length:
            print(f"\n⚠️ WARNING: {stats.max_input_length} max input length exceeds limit of {self.max_input_length}")
        
        if stats.max_output_length > self.max_output_length:
            print(f"⚠️ WARNING: {stats.max_output_length} max output length exceeds limit of {self.max_output_length}")
    
    def process_for_training(self) -> List[Dict[str, str]]:
        """
        Process all data into training format
        
        Returns:
            List of training examples with 'input' and 'target' keys
        """
        if not self.raw_data:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        print("🔄 Processing data for training...")
        
        processed_examples = []
        skipped_count = 0
        
        for i, example in enumerate(self.raw_data):
            try:
                # Clean input text
                cleaned_input = self.clean_input_text(example['input_text'])
                
                # Normalize ground truth
                normalized_gt = self.normalize_ground_truth(example['ground_truth'])
                
                # Create prompt and target
                prompt = self.create_training_prompt(cleaned_input)
                target = self.create_target_output(normalized_gt)
                
                # Check token lengths with proper error handling
                try:
                    prompt_tokens = len(self.tokenizer.encode(
                        prompt, 
                        max_length=self.max_input_length * 2,
                        truncation=True,
                        add_special_tokens=False
                    ))
                except Exception as e:
                    print(f"⚠️ Error tokenizing prompt for example {i+1}: {e}")
                    skipped_count += 1
                    continue
                
                try:
                    target_tokens = len(self.tokenizer.encode(
                        target,
                        max_length=self.max_output_length * 2,
                        truncation=True,
                        add_special_tokens=False
                    ))
                except Exception as e:
                    print(f"⚠️ Error tokenizing target for example {i+1}: {e}")
                    skipped_count += 1
                    continue
                
                # Apply length limits with truncation option
                if prompt_tokens > self.max_input_length:
                    print(f"⚠️ Example {i+1}: input too long ({prompt_tokens} tokens), truncating...")
                    # Truncate the prompt by reducing the input text
                    truncated_input = self.clean_input_text(example['input_text'][:len(example['input_text'])//2])
                    prompt = self.create_training_prompt(truncated_input)
                    
                    # Recheck length
                    prompt_tokens = len(self.tokenizer.encode(
                        prompt,
                        max_length=self.max_input_length,
                        truncation=True,
                        add_special_tokens=False
                    ))
                    
                    if prompt_tokens > self.max_input_length:
                        print(f"⚠️ Still too long after truncation, skipping example {i+1}")
                        skipped_count += 1
                        continue
                
                if target_tokens > self.max_output_length:
                    print(f"⚠️ Example {i+1}: output too long ({target_tokens} tokens)")
                    # For now, skip overly long targets
                    # Could implement target truncation if needed
                    skipped_count += 1
                    continue
                
                processed_examples.append({
                    'file_id': example['file_id'],
                    'input': prompt,
                    'target': target,
                    'input_length': prompt_tokens,
                    'target_length': target_tokens
                })
                
            except Exception as e:
                print(f"❌ Error processing example {i+1}: {e}")
                skipped_count += 1
                continue
        
        print(f"✅ Processed {len(processed_examples)} examples")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} examples due to length or errors")
        
        self.processed_data = processed_examples
        return processed_examples
    
    def create_train_val_test_splits(self, 
                                   train_ratio: float = 0.7, 
                                   val_ratio: float = 0.15, 
                                   test_ratio: float = 0.15,
                                   random_state: int = 42) -> Tuple[List, List, List]:
        """
        Split processed data into train/validation/test sets
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if not self.processed_data:
            raise ValueError("No processed data available. Call process_for_training() first.")
        
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        print(f"📊 Creating data splits: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            self.processed_data, 
            test_size=test_ratio, 
            random_state=random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        print(f"✅ Created splits:")
        print(f"  Training: {len(train_data)} examples")
        print(f"  Validation: {len(val_data)} examples") 
        print(f"  Test: {len(test_data)} examples")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, 
                           train_data: List, 
                           val_data: List, 
                           test_data: List,
                           output_dir: str = "processed_receipt_data"):
        """
        Save processed data to files for training
        
        Args:
            train_data: Training examples
            val_data: Validation examples
            test_data: Test examples
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 Saving processed data to {output_path}")
        
        # Save each split
        for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            
            # Save as JSON Lines format (common for training)
            jsonl_path = output_path / f"{split_name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
            
            # Save as JSON for easy inspection
            json_path = output_path / f"{split_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Saved {split_name}: {len(data)} examples")
        
        # Save dataset statistics
        if self.dataset_stats:
            stats_path = output_path / "dataset_stats.json"
            stats_dict = {
                'total_examples': self.dataset_stats.total_examples,
                'avg_input_length': self.dataset_stats.avg_input_length,
                'avg_output_length': self.dataset_stats.avg_output_length,
                'max_input_length': self.dataset_stats.max_input_length,
                'max_output_length': self.dataset_stats.max_output_length,
                'field_coverage': self.dataset_stats.field_coverage,
                'item_stats': self.dataset_stats.item_stats
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            print(f"  ✅ Saved dataset statistics")
        
        # Save preprocessing config
        config = {
            'model_name': self.model_name,
            'max_input_length': self.max_input_length,
            'max_output_length': self.max_output_length,
            'data_folder': str(self.data_folder),
            'total_raw_examples': len(self.raw_data),
            'total_processed_examples': len(self.processed_data)
        }
        
        config_path = output_path / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✅ Saved preprocessing configuration")
        print(f"\n🎉 All processed data saved to {output_path}")
    
    def inspect_examples(self, num_examples: int = 3):
        """
        Print a few examples to inspect the data quality
        
        Args:
            num_examples: Number of examples to display
        """
        if not self.processed_data:
            raise ValueError("No processed data available. Call process_for_training() first.")
        
        print(f"\n🔍 INSPECTING {num_examples} EXAMPLES:")
        print("="*70)
        
        for i in range(min(num_examples, len(self.processed_data))):
            example = self.processed_data[i]
            
            print(f"\n📄 EXAMPLE {i+1} (File ID: {example['file_id']}):")
            print(f"Input Length: {example['input_length']} tokens")
            print(f"Target Length: {example['target_length']} tokens")
            
            print(f"\n🔸 INPUT:")
            input_preview = example['input'][:300] + "..." if len(example['input']) > 300 else example['input']
            print(input_preview)
            
            print(f"\n🔸 TARGET:")
            target_preview = example['target'][:300] + "..." if len(example['target']) > 300 else example['target']
            print(target_preview)
            
            print("-" * 70)


def main():
    """
    Example usage of the ReceiptDataPreprocessor
    """
    # Initialize preprocessor
    preprocessor = ReceiptDataPreprocessor(
        data_folder="receipt_training_data",
        model_name="google/flan-t5-base",
        max_input_length=512,
        max_output_length=512
    )
    
    # Load and process data
    print("🚀 Starting data preprocessing pipeline...")
    
    # Step 1: Load raw data
    raw_data = preprocessor.load_raw_data()
    
    # Step 2: Analyze dataset
    preprocessor.analyze_dataset_statistics()
    preprocessor.print_dataset_analysis()
    
    # Step 3: Process for training
    processed_data = preprocessor.process_for_training()
    
    # Step 4: Create splits
    train_data, val_data, test_data = preprocessor.create_train_val_test_splits()
    
    # Step 5: Inspect examples
    preprocessor.inspect_examples(num_examples=2)
    
    # Step 6: Save processed data
    preprocessor.save_processed_data(train_data, val_data, test_data)
    
    print("\n🎉 Data preprocessing complete! Ready for fine-tuning.")


if __name__ == "__main__":
    main()