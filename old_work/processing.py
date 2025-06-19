#!/usr/bin/env python3
"""
Text extraction script for finding "Date 0" occurrences
and preserving surrounding text (4 chars before + "Date 0" + 1000 chars after)
"""

def extract_text_sections(input_file, output_file, search_string="Date 0", prev_chars=4, next_chars=1000):
    """
    Extract sections around occurrences of a search string from a text file.
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        search_string (str): String to search for
        prev_chars (int): Number of characters to include before the search string
        next_chars (int): Number of characters to include after the search string
    """
    try:
        # Try different encodings to handle unreadable characters
        encodings_to_try = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
        file_content = None
        
        for encoding in encodings_to_try:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    file_content = f.read()
                print(f"Successfully read file using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            # Last resort: read as binary and decode with errors='replace'
            with open(input_file, 'rb') as f:
                file_content = f.read().decode('utf-8', errors='replace')
            print("Used binary mode with error replacement for unreadable characters")
        
        extracted_sections = []
        start_index = 0
        
        # Find all occurrences of the search string
        while True:
            found_index = file_content.find(search_string, start_index)
            
            if found_index == -1:
                break  # No more occurrences found
            
            # Calculate extraction boundaries
            extract_start = max(0, found_index - prev_chars)
            extract_end = min(len(file_content), found_index + len(search_string) + next_chars)
            
            # Extract the section
            extracted_section = file_content[extract_start:extract_end]
            extracted_sections.append(extracted_section)
            
            # Move to next potential occurrence
            start_index = found_index + 1
        
        if not extracted_sections:
            print(f'No occurrences of "{search_string}" found in the file.')
            return
        
        # Combine all extracted sections with separators
        combined_content = '\n\n--- NEXT OCCURRENCE ---\n\n'.join(extracted_sections)
        
        # Write to new file using the same encoding approach
        with open(output_file, 'w', encoding='latin-1', errors='replace') as f:
            f.write(combined_content)
        
        print(f'Successfully extracted {len(extracted_sections)} occurrences of "{search_string}"')
        print(f'New file "{output_file}" created with {len(combined_content)} characters')
        
        # Show preview of first extraction
        if extracted_sections:
            print('\nPreview of first extracted section:')
            preview = extracted_sections[0][:200] + '...' if len(extracted_sections[0]) > 200 else extracted_sections[0]
            print(preview)
        
    except FileNotFoundError:
        print(f'Error: File "{input_file}" not found.')
    except Exception as e:
        print(f'Error processing file: {e}')

def main():
    # File paths - both input and output in 'data' folder
    input_file = 'data/tcp_data.txt'
    output_file = 'data/tcp_data_mod1.txt'
    
    # Run the extraction
    extract_text_sections(input_file, output_file)

if __name__ == '__main__':
    main()