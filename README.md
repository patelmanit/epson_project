# Overview

The main pipeline involves first using the preprocessing folder to convert the Wireshark data text files into chunks of receipts to be processed. Next we move into the finetuning 
folder for the two main finetuning approaches. Approach 1 here did not seem very promising, which is why a data-distillation based method to generate ground truth from a larger
model and use it to train the smaller model was employed in Approach 2. We then have a evaluation and partially implemented post-processing pipeline implemented in the distillation folder. 
Refer to the individual directory readmes for a detailed outlook on the approaches and the files in them.

## Approach 1 (decode_data.py → separate_data.py → tts_gen.py → main.py) is a streamlined pipeline designed for network packet captures: 

It decodes Wireshark hex data into text, separates individual receipts, uses Groq API to generate ground truth labels with automatic 80/10/10 train/val/test splits, 
then directly fine-tunes T5-FLAN with basic JSON optimization. 

## Approach 2 (ground_truth_gen.py → finetuning_setup.py → ft.py → testing.py) is a more comprehensive pipeline for pre-separated receipt files: 

It generates ground truths from multi-receipt files, performs extensive preprocessing with dataset statistics and analysis, fine-tunes with advanced JSON-focused techniques (schema validation, custom metrics, multi-strategy generation), and includes thorough testing with quality scoring. Both are fine-tuning workflows (not distillation), but Approach 1 is faster and simpler while Approach 2 offers better data analysis, training control, and model validation.
