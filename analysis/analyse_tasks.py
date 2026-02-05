import json
from pathlib import Path
from collections import defaultdict
import re

# Load task descriptions
tasks = []
with open("task_descriptions.jsonl", "r") as f:
    for line in f:
        task = json.loads(line.strip())
        tasks.append(task)

# Load comparison results
results_file = Path("output/comparison_results.json")
with open(results_file, 'r') as f:
    results = json.load(f)

# Create a mapping of task_id to match_percentage
task_matches = {r['task_id']: r['match_percentage'] for r in results if r['match_percentage'] is not None}

# Categorize tasks by instruction keywords
def categorize_task(instruction):
    """Categorize task based on keywords in instruction"""
    instruction_lower = instruction.lower()
    categories = []
    
    # Color operations
    if any(word in instruction_lower for word in ['color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange']):
        categories.append('color_change')
    
    # Shape operations
    if any(word in instruction_lower for word in ['circle', 'square', 'rectangle', 'triangle', 'shape']):
        categories.append('shape')
    
    # Geometric transformations
    if any(word in instruction_lower for word in ['rotate', 'flip', 'mirror', 'turn']):
        categories.append('rotation')
    
    # Size operations
    if any(word in instruction_lower for word in ['size', 'larger', 'smaller', 'scale', 'resize']):
        categories.append('size_change')
    
    # Position operations
    if any(word in instruction_lower for word in ['move', 'position', 'shift', 'place', 'center', 'left', 'right', 'top', 'bottom']):
        categories.append('position')
    
    # Addition/removal
    if any(word in instruction_lower for word in ['add', 'remove', 'delete', 'draw', 'create']):
        categories.append('add_remove')
    
    # Pattern operations
    if any(word in instruction_lower for word in ['pattern', 'repeat', 'grid', 'array']):
        categories.append('pattern')
    
    # Background operations
    if any(word in instruction_lower for word in ['background']):
        categories.append('background')
    
    # If no category matched, it's 'other'
    if not categories:
        categories.append('other')
    
    return categories

# Analyze tasks by category
category_stats = defaultdict(lambda: {'tasks': [], 'matches': [], 'count': 0})

for task in tasks:
    task_id = task['task_id']
    instruction = task['instruction']
    
    if task_id not in task_matches:
        continue
    
    match_pct = task_matches[task_id]
    categories = categorize_task(instruction)
    
    for category in categories:
        category_stats[category]['tasks'].append(task_id)
        category_stats[category]['matches'].append(match_pct)
        category_stats[category]['count'] += 1

# Calculate statistics for each category
category_results = []
for category, stats in category_stats.items():
    matches = stats['matches']
    avg_match = sum(matches) / len(matches) if matches else 0
    min_match = min(matches) if matches else 0
    max_match = max(matches) if matches else 0
    
    category_results.append({
        'category': category,
        'count': stats['count'],
        'avg_match': avg_match,
        'min_match': min_match,
        'max_match': max_match,
        'task_ids': stats['tasks']
    })

# Sort by average match (worst first)
category_results.sort(key=lambda x: x['avg_match'])

# Print results
print("="*70)
print("TASK TYPE ANALYSIS")
print("="*70)
print(f"\nTotal tasks analyzed: {len(task_matches)}")
print("\nPerformance by Task Type (sorted by average match):")
print("-"*70)

for cat in category_results:
    print(f"\n{cat['category'].upper().replace('_', ' ')}")
    print(f"  Tasks: {cat['count']}")
    print(f"  Average match: {cat['avg_match']:.2f}%")
    print(f"  Range: {cat['min_match']:.2f}% - {cat['max_match']:.2f}%")
    
    # Show examples of worst and best in this category
    task_matches_in_cat = [(tid, task_matches[tid]) for tid in cat['task_ids']]
    task_matches_in_cat.sort(key=lambda x: x[1])
    
    print(f"  Worst 3: ", end="")
    for tid, match in task_matches_in_cat[:3]:
        print(f"Task {tid} ({match:.1f}%)", end="  ")
    print()
    
    print(f"  Best 3: ", end="")
    for tid, match in task_matches_in_cat[-3:]:
        print(f"Task {tid} ({match:.1f}%)", end="  ")
    print()

# Save detailed results
output_file = Path("output/task_type_analysis.json")
with open(output_file, 'w') as f:
    json.dump(category_results, f, indent=2)

print(f"\n{'='*70}")
print(f"Detailed analysis saved to: {output_file}")

# Find tasks with multiple categories
print(f"\n{'='*70}")
print("SAMPLE INSTRUCTIONS BY CATEGORY")
print("="*70)

for cat in category_results[:5]:  # Top 5 worst categories
    print(f"\n{cat['category'].upper().replace('_', ' ')}:")
    sample_tasks = [t for t in tasks if t['task_id'] in cat['task_ids'][:3]]
    for task in sample_tasks:
        print(f"  Task {task['task_id']}: {task['instruction'][:80]}...")
