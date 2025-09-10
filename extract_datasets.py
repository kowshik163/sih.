#!/usr/bin/env python3
"""
Extract dataset information from FRA DATASETS docx files
"""

import os
import sys
import json
import re
from pathlib import Path

def extract_from_docx(file_path):
    """Extract text from docx file using python-docx"""
    try:
        # Try importing python-docx
        from docx import Document
        doc = Document(file_path)
        
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())
        
        return '\n'.join(text_content)
        
    except ImportError:
        print("python-docx not available, trying alternative method...")
        return extract_with_zipfile(file_path)
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def extract_with_zipfile(file_path):
    """Extract text using zipfile (docx is essentially a zip)"""
    import zipfile
    import xml.etree.ElementTree as ET
    
    try:
        with zipfile.ZipFile(file_path, 'r') as doc:
            # Read the main document
            xml_content = doc.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Extract text from XML
            text_content = []
            for elem in root.iter():
                if elem.text:
                    text_content.append(elem.text)
            
            return ' '.join(text_content)
            
    except Exception as e:
        print(f"Error with zipfile method for {file_path}: {e}")
        return None

def parse_dataset_info(text, filename):
    """Parse dataset information from extracted text"""
    datasets = []
    
    if not text:
        return datasets
    
    # Look for common dataset patterns
    url_pattern = r'https?://[^\s<>"\]{}\|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    # Look for dataset names and descriptions
    lines = text.split('\n')
    current_dataset = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for dataset indicators
        if any(keyword in line.lower() for keyword in ['dataset', 'data', 'corpus', 'collection']):
            if current_dataset and current_dataset.get('name'):
                datasets.append(current_dataset)
                current_dataset = {}
            
            # Extract potential dataset name
            if ':' in line:
                name = line.split(':')[0].strip()
                description = ':'.join(line.split(':')[1:]).strip()
                current_dataset = {'name': name, 'description': description}
            else:
                current_dataset = {'name': line, 'description': ''}
        
        elif current_dataset and not current_dataset.get('description'):
            current_dataset['description'] = line
        
        # Look for URLs in the line
        line_urls = re.findall(url_pattern, line)
        if line_urls and current_dataset:
            current_dataset['urls'] = line_urls
    
    # Add the last dataset
    if current_dataset and current_dataset.get('name'):
        datasets.append(current_dataset)
    
    # Also add standalone URLs found
    for url in urls:
        if not any(url in str(d.get('urls', [])) for d in datasets):
            datasets.append({
                'name': f'Dataset from {filename}',
                'description': f'Dataset URL found in {filename}',
                'urls': [url]
            })
    
    return datasets

def main():
    datasets_folder = Path("/Users/gkowshikreddy/Downloads/sih_-main/FRA DATASETS")
    all_datasets = {}
    
    print("Extracting dataset information from FRA DATASETS folder...")
    
    for docx_file in datasets_folder.glob("*.docx"):
        print(f"\nProcessing: {docx_file.name}")
        
        # Extract text content
        text_content = extract_from_docx(docx_file)
        
        if text_content:
            # Parse dataset information
            datasets = parse_dataset_info(text_content, docx_file.name)
            
            if datasets:
                category = docx_file.stem.replace(' ', '_').replace('&', 'and')
                all_datasets[category] = datasets
                print(f"Found {len(datasets)} datasets in {docx_file.name}")
                
                # Print first few entries
                for i, dataset in enumerate(datasets[:3]):
                    print(f"  - {dataset.get('name', 'Unnamed')}")
                    if dataset.get('urls'):
                        print(f"    URL: {dataset['urls'][0]}")
            else:
                print(f"No datasets found in {docx_file.name}")
        else:
            print(f"Could not extract content from {docx_file.name}")
    
    # Save extracted datasets
    output_file = datasets_folder.parent / "extracted_datasets.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_datasets, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted dataset information saved to: {output_file}")
    print(f"Total categories: {len(all_datasets)}")
    total_datasets = sum(len(datasets) for datasets in all_datasets.values())
    print(f"Total datasets found: {total_datasets}")
    
    return all_datasets

if __name__ == "__main__":
    extracted_datasets = main()
