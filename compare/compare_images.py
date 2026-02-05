import json
import numpy as np
from PIL import Image
from pathlib import Path

# Setup directories
images_dir = Path("images")
output_dir = Path("output")

def compare_images(img1_path, img2_path):
    """Compare two images and return percentage of matching pixels"""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Ensure same size
        if img1.size != img2.size:
            print(f"    Warning: Images have different sizes: {img1.size} vs {img2.size}")
            img2 = img2.resize(img1.size)
        
        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Compare pixels
        matching_pixels = np.all(arr1 == arr2, axis=-1).sum()
        total_pixels = arr1.shape[0] * arr1.shape[1]
        match_percentage = (matching_pixels / total_pixels) * 100
        
        return match_percentage, img1.size
    except Exception as e:
        print(f"    Error comparing images: {e}")
        return None, None

# Find all output images
output_images = sorted(output_dir.glob("q*_output.png"))

print(f"Found {len(output_images)} output images to compare")
print("="*70)

results = []

for output_path in output_images:
    # Extract task ID from filename (e.g., q1_output.png -> 1)
    task_id = int(output_path.stem.split('_')[0][1:])
    
    # Find corresponding answer image
    answer_path = images_dir / f"q{task_id}_answer.png"
    
    if not answer_path.exists():
        print(f"Task {task_id}: No answer image found")
        results.append({
            'task_id': task_id,
            'match_percentage': None,
            'status': 'no_answer'
        })
        continue
    
    print(f"Task {task_id}: Comparing output vs answer...")
    match_pct, size = compare_images(output_path, answer_path)
    
    if match_pct is not None:
        print(f"  Match: {match_pct:.2f}% (size: {size})")
        results.append({
            'task_id': task_id,
            'match_percentage': match_pct,
            'image_size': size,
            'status': 'compared'
        })
    else:
        results.append({
            'task_id': task_id,
            'match_percentage': None,
            'status': 'error'
        })

# Save results
results_file = output_dir / "comparison_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

valid_matches = [r['match_percentage'] for r in results if r['match_percentage'] is not None]

if valid_matches:
    avg_match = sum(valid_matches) / len(valid_matches)
    min_match = min(valid_matches)
    max_match = max(valid_matches)
    
    print(f"Total comparisons: {len(valid_matches)}")
    print(f"\nPixel Match Statistics:")
    print(f"  Average: {avg_match:.2f}%")
    print(f"  Min: {min_match:.2f}%")
    print(f"  Max: {max_match:.2f}%")
    
    # Distribution
    perfect = sum(1 for m in valid_matches if m == 100.0)
    high = sum(1 for m in valid_matches if 90 <= m < 100)
    medium = sum(1 for m in valid_matches if 50 <= m < 90)
    low = sum(1 for m in valid_matches if m < 50)
    
    print(f"\nMatch Distribution:")
    print(f"  Perfect (100%): {perfect}")
    print(f"  High (90-99%): {high}")
    print(f"  Medium (50-89%): {medium}")
    print(f"  Low (<50%): {low}")
    
    # Show best and worst matches
    sorted_results = sorted([r for r in results if r['match_percentage'] is not None], 
                          key=lambda x: x['match_percentage'], reverse=True)
    
    print(f"\nTop 5 Best Matches:")
    for r in sorted_results[:5]:
        print(f"  Task {r['task_id']}: {r['match_percentage']:.2f}%")
    
    print(f"\nTop 5 Worst Matches:")
    for r in sorted_results[-5:]:
        print(f"  Task {r['task_id']}: {r['match_percentage']:.2f}%")
else:
    print("No valid comparisons found")

print(f"\nDetailed results saved to: {results_file}")
