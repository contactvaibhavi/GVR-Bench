#!/usr/bin/env python3
"""
Master script to generate all input and answer images for the benchmark.
This script imports and runs all individual question scripts (q1.py through q125.py).
"""

import os
import sys
from pathlib import Path
import importlib.util

# Ensure the images directory exists
os.makedirs('images', exist_ok=True)

def generate_all_images():
    """Generate all images by calling each question script."""
    print("Starting image generation for all 125 tasks...")
    print("=" * 60)
    
    total_tasks = 125
    failed_tasks = []
    
    for i in range(1, total_tasks + 1):
        module_name = f'q{i}'
        module_path = f'image_generation_scripts/{module_name}.py'
        print(f"\nGenerating images for Task {i}/{total_tasks}...", end=" ")
        
        try:
            # Import the module dynamically from the image_generation_scripts folder
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call the input and answer functions
            module.__dict__[f'q{i}_input']()
            module.__dict__[f'q{i}_answer']()
            
            print("✓ Done")
            
        except Exception as e:
            print(f"✗ Failed")
            print(f"  Error: {str(e)}")
            failed_tasks.append(i)
    
    print("\n" + "=" * 60)
    print(f"\nImage generation complete!")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {total_tasks - len(failed_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed tasks: {', '.join(map(str, failed_tasks))}")
        return 1
    else:
        print("\nAll images generated successfully!")
        return 0

if __name__ == '__main__':
    exit_code = generate_all_images()
    sys.exit(exit_code)

