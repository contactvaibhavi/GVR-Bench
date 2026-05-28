import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare pixel matching vs perceptual similarity')
parser.add_argument('--output-dir', type=str, default='output', 
                    help='Directory containing results files (default: output)')
parser.add_argument('--tasks-file', type=str, default='task_descriptions.jsonl', 
                    help='Path to task descriptions file (default: task_descriptions.jsonl)')
parser.add_argument('--pixel-results', type=str, default=None,
                    help='Path to pixel comparison results (default: <output-dir>/comparison_results.json)')
parser.add_argument('--perceptual-results', type=str, default=None,
                    help='Path to perceptual similarity results (default: <output-dir>/perceptual_similarity_results.json)')
parser.add_argument('--output-file', type=str, default=None,
                    help='Output comparison file (default: <output-dir>/pixel_vs_perceptual_comparison.json)')

args = parser.parse_args()

# Setup paths
output_dir = Path(args.output_dir)

if args.pixel_results:
    pixel_file = Path(args.pixel_results)
else:
    pixel_file = output_dir / "comparison_results.json"

if args.perceptual_results:
    perceptual_file = Path(args.perceptual_results)
else:
    perceptual_file = output_dir / "perceptual_similarity_results.json"

if args.output_file:
    output_file = Path(args.output_file)
else:
    output_file = output_dir / "pixel_vs_perceptual_comparison.json"

print(f"Output directory: {output_dir}")
print(f"Tasks file: {args.tasks_file}")
print(f"Pixel results: {pixel_file}")
print(f"Perceptual results: {perceptual_file}")
print(f"Output file: {output_file}")
print("="*80)

# Load task descriptions
tasks = {}
with open(args.tasks_file, "r") as f:
    for line in f:
        task = json.loads(line.strip())
        tasks[task['task_id']] = task

# Load pixel matching results
with open(pixel_file, 'r') as f:
    pixel_results = json.load(f)

# Load perceptual similarity results
with open(perceptual_file, 'r') as f:
    perceptual_results = json.load(f)

# Create combined dataset
combined_data = []
for p_result in perceptual_results:
    if p_result['status'] != 'success':
        continue
    
    task_id = p_result['task_id']
    
    # Find corresponding pixel result
    pixel_match = None
    for px_result in pixel_results:
        if px_result['task_id'] == task_id:
            pixel_match = px_result.get('match_percentage', 0)
            break
    
    if pixel_match is None:
        continue
    
    combined_data.append({
        'task_id': task_id,
        'pixel_match': pixel_match,
        'ssim': p_result['ssim'] * 100,
        'histogram_similarity': p_result['histogram_similarity'] * 100,
        'mse_similarity': p_result['mse_similarity'] * 100,
        'instruction': tasks.get(task_id, {}).get('instruction', 'Unknown')
    })

# Categorize tasks
def categorize_task(instruction):
    """Categorize task based on keywords"""
    instruction_lower = instruction.lower()
    categories = []
    
    if any(word in instruction_lower for word in ['color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'cyan', 'magenta']):
        categories.append('color_change')
    if any(word in instruction_lower for word in ['circle', 'square', 'rectangle', 'triangle', 'shape', 'oval', 'ellipse']):
        categories.append('shape')
    if any(word in instruction_lower for word in ['rotate', 'flip', 'mirror', 'turn', 'invert']):
        categories.append('rotation')
    if any(word in instruction_lower for word in ['size', 'larger', 'smaller', 'scale', 'resize', 'grow', 'shrink']):
        categories.append('size_change')
    if any(word in instruction_lower for word in ['move', 'position', 'shift', 'place', 'center', 'left', 'right', 'top', 'bottom']):
        categories.append('position')
    if any(word in instruction_lower for word in ['add', 'remove', 'delete', 'draw', 'create', 'erase']):
        categories.append('add_remove')
    if any(word in instruction_lower for word in ['pattern', 'repeat', 'grid', 'array', 'stripe', 'checkerboard']):
        categories.append('pattern')
    if any(word in instruction_lower for word in ['background']):
        categories.append('background')
    if any(word in instruction_lower for word in ['text', 'letter', 'word', 'number', 'digit']):
        categories.append('text')
    
    if not categories:
        categories.append('other')
    
    return categories

# Add categories to combined data
for item in combined_data:
    item['categories'] = categorize_task(item['instruction'])

# Print overall comparison
print("="*80)
print("PIXEL MATCHING VS PERCEPTUAL SIMILARITY COMPARISON")
print("="*80)

print(f"\nTotal tasks analyzed: {len(combined_data)}")

print("\n" + "-"*80)
print("OVERALL STATISTICS")
print("-"*80)

pixel_scores = [d['pixel_match'] for d in combined_data]
ssim_scores = [d['ssim'] for d in combined_data]
hist_scores = [d['histogram_similarity'] for d in combined_data]
mse_scores = [d['mse_similarity'] for d in combined_data]

print(f"\nPixel Matching:")
print(f"  Average: {np.mean(pixel_scores):.2f}%")
print(f"  Median: {np.median(pixel_scores):.2f}%")
print(f"  Min: {np.min(pixel_scores):.2f}%")
print(f"  Max: {np.max(pixel_scores):.2f}%")

print(f"\nSSIM (Structural Similarity):")
print(f"  Average: {np.mean(ssim_scores):.2f}%")
print(f"  Median: {np.median(ssim_scores):.2f}%")
print(f"  Min: {np.min(ssim_scores):.2f}%")
print(f"  Max: {np.max(ssim_scores):.2f}%")

print(f"\nHistogram Similarity:")
print(f"  Average: {np.mean(hist_scores):.2f}%")
print(f"  Median: {np.median(hist_scores):.2f}%")

print(f"\nMSE Similarity:")
print(f"  Average: {np.mean(mse_scores):.2f}%")
print(f"  Median: {np.median(mse_scores):.2f}%")

# Performance classification
print("\n" + "-"*80)
print("PERFORMANCE CLASSIFICATION")
print("-"*80)

def classify_performance(pixel, ssim):
    """Classify performance based on both metrics"""
    if ssim >= 90:
        if pixel >= 80:
            return "Excellent (pixel-perfect)"
        elif pixel >= 50:
            return "Good (near pixel-perfect)"
        else:
            return "Perceptually correct (not pixel-perfect)"
    elif ssim >= 70:
        return "Acceptable (some differences)"
    else:
        return "Poor (failed edit)"

classifications = defaultdict(list)
for item in combined_data:
    classification = classify_performance(item['pixel_match'], item['ssim'])
    classifications[classification].append(item)

for classification in ["Excellent (pixel-perfect)", "Good (near pixel-perfect)", 
                       "Perceptually correct (not pixel-perfect)", 
                       "Acceptable (some differences)", "Poor (failed edit)"]:
    items = classifications[classification]
    if items:
        print(f"\n{classification}: {len(items)} tasks ({len(items)/len(combined_data)*100:.1f}%)")
        avg_pixel = np.mean([i['pixel_match'] for i in items])
        avg_ssim = np.mean([i['ssim'] for i in items])
        print(f"  Average Pixel Match: {avg_pixel:.2f}%")
        print(f"  Average SSIM: {avg_ssim:.2f}%")
        
        # Show examples
        if len(items) <= 5:
            print(f"  Tasks: {', '.join(str(i['task_id']) for i in items)}")

# Compare by category
print("\n" + "="*80)
print("COMPARISON BY TASK CATEGORY")
print("="*80)

category_comparison = defaultdict(lambda: {
    'pixel_scores': [],
    'ssim_scores': [],
    'count': 0
})

for item in combined_data:
    for category in item['categories']:
        category_comparison[category]['pixel_scores'].append(item['pixel_match'])
        category_comparison[category]['ssim_scores'].append(item['ssim'])
        category_comparison[category]['count'] += 1

# Sort by SSIM performance
category_list = []
for category, data in category_comparison.items():
    category_list.append({
        'category': category,
        'count': data['count'],
        'avg_pixel': np.mean(data['pixel_scores']),
        'avg_ssim': np.mean(data['ssim_scores']),
        'improvement': np.mean(data['ssim_scores']) - np.mean(data['pixel_scores'])
    })

category_list.sort(key=lambda x: x['avg_ssim'])

print("\nCategories ranked by SSIM performance:")
print(f"{'Category':<20} {'Count':<8} {'Pixel %':<10} {'SSIM %':<10} {'Improvement':<12}")
print("-"*80)

for cat in category_list:
    print(f"{cat['category']:<20} {cat['count']:<8} {cat['avg_pixel']:<10.2f} {cat['avg_ssim']:<10.2f} {cat['improvement']:+<12.2f}")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

improvement = np.mean(ssim_scores) - np.mean(pixel_scores)
print(f"\n1. Overall improvement: SSIM is {improvement:.2f}% higher than pixel matching")

perceptually_correct = len(classifications["Perceptually correct (not pixel-perfect)"])
print(f"\n2. {perceptually_correct} tasks ({perceptually_correct/len(combined_data)*100:.1f}%) are perceptually correct")
print(f"   but not pixel-perfect (high SSIM, low pixel match)")

poor_tasks = len(classifications["Poor (failed edit)"])
print(f"\n3. Only {poor_tasks} tasks ({poor_tasks/len(combined_data)*100:.1f}%) are genuine failures")
print(f"   (low SSIM indicates incorrect edits)")

# Categories with biggest gaps
category_list.sort(key=lambda x: x['improvement'], reverse=True)
print(f"\n4. Categories with biggest pixel vs SSIM gaps:")
for cat in category_list[:3]:
    print(f"   {cat['category']}: {cat['improvement']:+.2f}% improvement")
    print(f"     (Pixel: {cat['avg_pixel']:.1f}%, SSIM: {cat['avg_ssim']:.1f}%)")

# Save results
with open(output_file, 'w') as f:
    json.dump({
        'overall_stats': {
            'pixel_avg': float(np.mean(pixel_scores)),
            'ssim_avg': float(np.mean(ssim_scores)),
            'improvement': float(improvement)
        },
        'classifications': {k: len(v) for k, v in classifications.items()},
        'category_comparison': category_list,
        'combined_data': combined_data
    }, f, indent=2)

print(f"\n{'='*80}")
print(f"Detailed results saved to: {output_file}")
