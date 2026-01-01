#!/usr/bin/env python3
"""
Extract and decode receipt data from Wireshark packet dissection output.
Finds hex data strings and converts them to readable text.
"""

import re
import binascii

def extract_hex_data_from_wireshark(input_file, output_file):
    """
    Extract hex data from Wireshark dissection and convert to readable text.
    
    Args:
        input_file (str): Path to Wireshark dissection text file
        output_file (str): Path to output file for decoded receipt data
    """
    try:
        # Read the file
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        print(f"Successfully read file ({len(content)} characters)")
        
        # Find all hex data lines using regex
        # Looking for lines like: "Data: 446174652030372f31312f32352020202020202054696d652031323a3036706d"
        hex_pattern = r'Data:\s*([0-9a-fA-F]+)'
        hex_matches = re.findall(hex_pattern, content)
        
        print(f"Found {len(hex_matches)} hex data entries")
        
        decoded_sections = []
        
        for i, hex_string in enumerate(hex_matches):
            try:
                # Convert hex string to bytes
                hex_bytes = binascii.unhexlify(hex_string)
                
                # Try to decode as text (try multiple encodings)
                decoded_text = None
                encodings = ['utf-8', 'ascii', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        decoded_text = hex_bytes.decode(encoding, errors='replace')
                        break
                    except:
                        continue
                
                if decoded_text:
                    # Clean up the text (remove non-printable characters except newlines/tabs)
                    clean_text = ''.join(char if char.isprintable() or char in '\n\r\t' else ' ' 
                                       for char in decoded_text)
                    
                    # Only keep sections that have meaningful content (not just spaces/control chars)
                    if clean_text.strip() and len(clean_text.strip()) > 3:
                        decoded_sections.append(clean_text)
                        
            except Exception as e:
                print(f"Error decoding hex string {i+1}: {e}")
                continue
        
        if not decoded_sections:
            print("No decodable receipt data found in hex strings.")
            return
        
        # Write decoded data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(decoded_sections))
        
        print(f'Successfully decoded {len(decoded_sections)} data sections')
        print(f'Output written to "{output_file}"')
        
        # Show preview
        if decoded_sections:
            print('\nPreview of decoded data:')
            preview = decoded_sections[0][:300] + '...' if len(decoded_sections[0]) > 300 else decoded_sections[0]
            print(preview)
        
    except FileNotFoundError:
        print(f'Error: File "{input_file}" not found.')
    except Exception as e:
        print(f'Error processing file: {e}')

def extract_all_packet_data(input_file, output_file):
    """
    More comprehensive extraction that looks for any hex data patterns in the Wireshark output.
    """
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        print(f"Successfully read file ({len(content)} characters)")
        
        # Multiple patterns to catch different hex data formats
        patterns = [
            r'Data:\s*([0-9a-fA-F]+)',  # Standard "Data:" lines
            r'^\s*[0-9a-fA-F]{4}\s+([0-9a-fA-F\s]+)\s+(.+)$',  # Hex dump format with ASCII
            r'([0-9a-fA-F]{32,})',  # Long hex strings (32+ chars)
        ]
        
        all_decoded = []
        
        for pattern_name, pattern in enumerate(patterns):
            matches = re.findall(pattern, content, re.MULTILINE)
            print(f"Pattern {pattern_name + 1}: Found {len(matches)} matches")
            
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multi-group matches
                    for group in match:
                        if group and len(group.replace(' ', '')) > 10:  # Skip short or empty groups
                            try:
                                # Remove spaces and try to decode
                                hex_clean = group.replace(' ', '')
                                if re.match(r'^[0-9a-fA-F]+$', hex_clean) and len(hex_clean) % 2 == 0:
                                    hex_bytes = binascii.unhexlify(hex_clean)
                                    decoded = hex_bytes.decode('utf-8', errors='replace')
                                    clean_decoded = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in decoded)
                                    if clean_decoded.strip():
                                        all_decoded.append(clean_decoded)
                            except:
                                continue
                else:
                    # Handle single group matches
                    try:
                        hex_clean = match.replace(' ', '')
                        if re.match(r'^[0-9a-fA-F]+$', hex_clean) and len(hex_clean) % 2 == 0:
                            hex_bytes = binascii.unhexlify(hex_clean)
                            decoded = hex_bytes.decode('utf-8', errors='replace')
                            clean_decoded = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in decoded)
                            if clean_decoded.strip():
                                all_decoded.append(clean_decoded)
                    except:
                        continue
        
        # Remove duplicates
        unique_decoded = list(dict.fromkeys(all_decoded))
        
        if not unique_decoded:
            print("No decodable data found.")
            return
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_decoded))
        
        print(f'Successfully decoded {len(unique_decoded)} unique data sections')
        print(f'Output written to "{output_file}"')
        
        # Show preview
        if unique_decoded:
            print('\nPreview of first decoded section:')
            preview = unique_decoded[0][:200] + '...' if len(unique_decoded[0]) > 200 else unique_decoded[0]
            print(preview)
        
    except Exception as e:
        print(f'Error processing file: {e}')

def main():
    # File paths
    input_file = 'full_dissection.txt'  # Your Wireshark dissection file
    output_file = 'decoded_receipt_data.txt'
    
    print("Receipt Data Extractor for Wireshark Dissection")
    print("=" * 50)
    print("1. Extract only 'Data:' hex lines")
    print("2. Extract all hex data patterns (comprehensive)")
    
    choice = input("Choose extraction method (1 or 2, default=2): ").strip()
    
    if choice == "1":
        print("\nExtracting 'Data:' hex lines only...")
        extract_hex_data_from_wireshark(input_file, output_file)
    else:
        print("\nExtracting all hex data patterns...")
        extract_all_packet_data(input_file, output_file)

if __name__ == '__main__':
    main()