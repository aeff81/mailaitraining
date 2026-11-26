#!/usr/bin/env python3
"""
Remove BASE64-encoded attachment parts from MIME email files while preserving structure.

These are mbox files with multipart/mixed content. The script removes parts
with Content-Transfer-Encoding: base64 (the actual attachment data), keeping
only the headers and the boundaries.
"""

import re
import sys
import os

MAIL_DIR = "/home/opencode/training/sentfolder/cur"
BOUNDARY_PATTERN = re.compile(r'^--------------.*$')
BASE64_HEADER_PATTERN = re.compile(r'^Content-Transfer-Encoding: base64$')

def process_file(filepath):
    """Process a single email file and remove base64-encoded parts."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    output_lines = []
    in_base64 = False
    
    for i, line in enumerate(lines):
        # Check for boundary line
        if BOUNDARY_PATTERN.match(line):
            output_lines.append(line)
            # Check if next line is base64 Content-Transfer-Encoding
            if i + 1 < len(lines) and BASE64_HEADER_PATTERN.match(lines[i + 1]):
                in_base64 = True
                continue
        
        # Check for base64 Content-Transfer-Encoding line
        elif BASE64_HEADER_PATTERN.match(line):
            output_lines.append(line)
            in_base64 = True
            continue
        
        # If we're in base64 section and hit another boundary, stop base64 mode
        elif in_base64 and BOUNDARY_PATTERN.match(line):
            output_lines.append(line)
            in_base64 = False
            continue
        
        # If we're in base64 and don't hit a boundary, skip it
        elif in_base64:
            # This is base64 content - skip
            continue
        
        # Otherwise, print the line
        else:
            output_lines.append(line)
    
    # Write output
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"  Processed: {filepath}")

def main():
    # Find all files in MAIL_DIR
    for filename in os.listdir(MAIL_DIR):
        filepath = os.path.join(MAIL_DIR, filename)
        if os.path.isfile(filepath):
            process_file(filepath)
    
    print("Done!")

if __name__ == "__main__":
    main()
