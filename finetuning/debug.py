import json
import os
# Compare raw ground truth vs processed target
raw_gt_file = 'receipt_training_data/ground_truths/ground_truth_1.json'
if os.path.exists(raw_gt_file):
    print("P")
    with open(raw_gt_file, 'r') as f:
        original = json.load(f)
    print("Original ground truth:")
    print(json.dumps(original, indent=2)[:300])
    
    # Compare with processed
    with open('processed_receipt_data/train.jsonl', 'r') as f:
        processed = json.loads(f.readline())
    print("\nProcessed target:")
    print(processed['target'][:300])