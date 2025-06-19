import spacy
import re
from datetime import datetime
import dateparser
# Load a pre-trained spaCy model - using en_core_web_sm which has good balance of accuracy and speed
nlp = spacy.load("en_core_web_trf")

def extract_info_from_receipt(text):
    cleaned_text = text
    clt = ""
    for word in cleaned_text.split(" "):
        print(word)
        clt+=word
        clt+="\n"
    cleaned_text = clt
    # print(cleaned_text)
    
    date_match = re.search(r'Date\s+(\d{2}/\d{2}/\d{2})', text)
    time_match = re.search(r'Time\s+(\d{2}:\d{2}(?:\s*am|pm))', text)
    
    date = date_match.group(1) if date_match else None
    receipt_time = time_match.group(1) if time_match else None
    
    doc = nlp(cleaned_text)

    customer_name = None
    print(doc.ents)
    possible_names = []
    namefound = False
    for ent in doc.ents:
        print("checking labels:", ent.text, ent.label_, "\n")
        if not namefound and ent.label_ == "PERSON":
            print("ent", ent.text)
            context = text[max(0, text.find(ent.text)-30):min(len(text), text.find(ent.text)+30)]
            possible_names.append([ent.text])
            if "Check#" in context or "Table" in context or ent.text.upper() in text:
                customer_name = ent.text
                namefound = True
    print(possible_names)
    return {
        "date": date,
        "time": receipt_time,
        "customer_name": customer_name
    }

def parse_file(file_path):
        """Parse a receipt file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        return text
sample_receipt = parse_file("sample1.txt")

result = extract_info_from_receipt(sample_receipt)
print(f"Date: {result['date']}")
print(f"Time: {result['time']}")
print(f"Customer Name: {result['customer_name']}")