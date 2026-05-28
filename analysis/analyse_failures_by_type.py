import json
from pathlib import Path
from collections import defaultdict

# Load task descriptions
tasks = []
with open("task_descriptions.jsonl", "r") as f:
    for line in f:
        task = json.loads(line.strip())
        tasks.append(task)

# Load perceptual similarity results
perceptual_file = Path("output/perceptual_similarity_results.json")
with open(perceptual_file, 'r') as f:
    perceptual_results = json.load(f)

# Create mapping of task_id to metrics
task_metrics = {
    r['task_id']: {
        'ssim': r.get('ssim', 0) * 100,  # Convert to percentage
        'histogram_similarity': r.get('histogram_similarity', 0) * 100,
        'mse_similarity': r.get('mse_similarity', 0) * 100
    }
    for r in perceptual_results if r['status'] == 'success'
}

# Categorize tasks
def categorize_task(instruction):
    """Categorize task based on keywords in instruction"""
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
    
    if any(word in instruction_lower for word in ['gradient', 'blend', 'fade']):
        categories.append('gradient')
    
    if not categories:
        categories.append('other')
    
    return categories

# Analyze by category
category_stats = defaultdict(lambda: {
    'tasks': [],
    'ssim_scores': [],
    'hist_scores': [],
    'mse_scores': []
})

for task in tasks:
    task_id = task['task_id']
    instruction = task['instruction']
    
    if task_id not in task_metrics:
        continue
    
    metrics = task_metrics[task_id]
    categories = categorize_task(instruction)
    
    for category in categories:
        category_stats[category]['tasks'].append({
            'id': task_id,
            'instruction': instruction,
            'ssim': metrics['ssim']
        })
        category_stats[category]['ssim_scores'].append(metrics['ssim'])
        category_stats[category]['hist_scores'].append(metrics['histogram_similarity'])
        category_stats[category]['mse_scores'].append(metrics['mse_similarity'])

# Calculate statistics
import numpy as np

category_results = []
for category, stats in category_stats.items():
    ssim_scores = stats['ssim_scores']
    
    category_results.append({
        'category': category,
        'count': len(ssim_scores),
        'avg_ssim': np.mean(ssim_scores),
        'min_ssim': np.min(ssim_scores),
        'max_ssim': np.max(ssim_scores),
        'median_ssim': np.median(ssim_scores),
        'std_ssim': np.std(ssim_scores),
        'tasks': stats['tasks']
    })

# Sort by average SSIM (worst first)
category_results.sort(key=lambda x: x['avg_ssim'])

# Print results
print("="*80)
print("FAILURE ANALYSIS BY TASK TYPE")
print("="*80)
print("\nTask categories ranked by performance (SSIM, worst to best):")
print("-"*80)

for cat in category_results:
    print(f"\n{cat['category'].upper().replace('_', ' ')}")
    print(f"  Count: {cat['count']} tasks")
    print(f"  Average SSIM: {cat['avg_ssim']:.2f}%")
    print(f"  Range: {cat['min_ssim']:.2f}% - {cat['max_ssim']:.2f}%")
    print(f"  Median: {cat['median_ssim']:.2f}%")
    print(f"  Std Dev: {cat['std_ssim']:.2f}%")
    
    # Identify failure rate (SSIM < 70%)
    failures = [t for t in cat['tasks'] if t['ssim'] < 70]
    if failures:
        print(f"  Failure rate: {len(failures)}/{cat['count']} ({len(failures)/cat['count']*100:.1f}%)")
        print(f"  Failed tasks: {', '.join(str(t['id']) for t in failures[:5])}")
        
        # Show worst instruction examples
        worst = sorted(failures, key=lambda x: x['ssim'])[:3]
        print(f"  Worst examples:")
        for t in worst:
            print(f"    Task {t['id']} (SSIM: {t['ssim']:.1f}%): {t['instruction'][:70]}...")
    else:
        print(f"  No major failures (all SSIM > 70%)")

# Identify patterns in failures
print("\n" + "="*80)
print("FAILURE PATTERN ANALYSIS")
print("="*80)

all_tasks_with_metrics = []
for task in tasks:
    if task['task_id'] in task_metrics:
        all_tasks_with_metrics.append({
            'id': task['task_id'],
            'instruction': task['instruction'],
            'ssim': task_metrics[task['task_id']]['ssim'],
            'categories': categorize_task(task['instruction'])
        })

# Find tasks with multiple categories (complex tasks)
complex_tasks = [t for t in all_tasks_with_metrics if len(t['categories']) > 1]
simple_tasks = [t for t in all_tasks_with_metrics if len(t['categories']) == 1]

print(f"\nComplex tasks (multiple categories): {len(complex_tasks)}")
print(f"  Average SSIM: {np.mean([t['ssim'] for t in complex_tasks]):.2f}%")
print(f"  Failures (SSIM < 70%): {len([t for t in complex_tasks if t['ssim'] < 70])}")

print(f"\nSimple tasks (single category): {len(simple_tasks)}")
print(f"  Average SSIM: {np.mean([t['ssim'] for t in simple_tasks]):.2f}%")
print(f"  Failures (SSIM < 70%): {len([t for t in simple_tasks if t['ssim'] < 70])}")

# Show worst overall tasks
print("\n" + "="*80)
print("TOP 15 WORST PERFORMING TASKS")
print("="*80)

worst_tasks = sorted(all_tasks_with_metrics, key=lambda x: x['ssim'])[:15]
for i, task in enumerate(worst_tasks, 1):
    print(f"\n{i}. Task {task['id']} - SSIM: {task['ssim']:.2f}%")
    print(f"   Categories: {', '.join(task['categories'])}")
    print(f"   Instruction: {task['instruction']}")

# Save detailed results
output_file = Path("output/failure_analysis_by_type.json")
with open(output_file, 'w') as f:
    json.dump({
        'category_results': category_results,
        'complex_vs_simple': {
            'complex': {
                'count': len(complex_tasks),
                'avg_ssim': float(np.mean([t['ssim'] for t in complex_tasks])),
                'failures': len([t for t in complex_tasks if t['ssim'] < 70])
            },
            'simple': {
                'count': len(simple_tasks),
                'avg_ssim': float(np.mean([t['ssim'] for t in simple_tasks])),
                'failures': len([t for t in simple_tasks if t['ssim'] < 70])
            }
        },
        'worst_tasks': worst_tasks
    }, f, indent=2)

print(f"\n{'='*80}")
print(f"Detailed results saved to: {output_file}")
