import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL file"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def categorize_task(instruction):
    """Categorize task based on keywords in instruction"""
    instruction_lower = instruction.lower()
    categories = []
    
    if any(word in instruction_lower for word in ['color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'cyan', 'magenta']):
        categories.append('Color Change')
    
    if any(word in instruction_lower for word in ['circle', 'square', 'rectangle', 'triangle', 'shape', 'oval', 'ellipse']):
        categories.append('Geometric')
    
    if any(word in instruction_lower for word in ['rotate', 'flip', 'mirror', 'turn', 'invert']):
        categories.append('Rotation/Flip')
    
    if any(word in instruction_lower for word in ['size', 'larger', 'smaller', 'scale', 'resize', 'grow', 'shrink']):
        categories.append('Size Change')
    
    if any(word in instruction_lower for word in ['move', 'position', 'shift', 'place', 'center', 'left', 'right', 'top', 'bottom']):
        categories.append('Position')
    
    if any(word in instruction_lower for word in ['add', 'remove', 'delete', 'draw', 'create', 'erase']):
        categories.append('Add/Remove')
    
    if any(word in instruction_lower for word in ['pattern', 'repeat', 'grid', 'array', 'stripe', 'checkerboard']):
        categories.append('Pattern')
    
    if any(word in instruction_lower for word in ['background']):
        categories.append('Background')
    
    if any(word in instruction_lower for word in ['text', 'letter', 'word', 'number', 'digit']):
        categories.append('Text')
    
    if any(word in instruction_lower for word in ['gradient', 'blend', 'fade']):
        categories.append('Gradient')
    
    if not categories:
        categories.append('Other')
    
    return categories

def plot_failure_by_task_type(stats, tasks, output_path):
    """Bar chart of failure rates by task type"""
    # Build task lookup
    task_lookup = {t['task_id']: t for t in tasks}
    
    # Categorize each task
    category_stats = defaultdict(lambda: {'total': 0, 'failed': 0, 'tasks': []})
    
    for stat in stats:
        task_id = stat.get('task_id', 0)
        passed = stat.get('passed_lenient', False)
        
        if task_id not in task_lookup:
            continue
        
        instruction = task_lookup[task_id]['instruction']
        categories = categorize_task(instruction)
        
        for category in categories:
            category_stats[category]['total'] += 1
            if not passed:
                category_stats[category]['failed'] += 1
                category_stats[category]['tasks'].append(task_id)
    
    # Calculate failure rates
    categories = []
    failure_rates = []
    totals = []
    failed_counts = []
    
    for cat in sorted(category_stats.keys()):
        stats_cat = category_stats[cat]
        total = stats_cat['total']
        failed = stats_cat['failed']
        
        if total > 0:
            categories.append(cat)
            failure_rates.append(failed / total * 100)
            totals.append(total)
            failed_counts.append(failed)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.RdYlGn_r(np.array(failure_rates) / 100)
    bars = ax.bar(categories, failure_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add labels
    for i, (bar, rate, total, failed) in enumerate(zip(bars, failure_rates, totals, failed_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{failed}/{total}\n({rate:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Failure Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
    ax.set_title('Failure Rate by Task Type', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, max(failure_rates) * 1.2 if failure_rates else 100)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_accuracy_by_task_type(stats, tasks, output_path):
    """Box plot of accuracies by task type"""
    task_lookup = {t['task_id']: t for t in tasks}
    
    category_accuracies = defaultdict(lambda: {'changed': [], 'unchanged': []})
    
    for stat in stats:
        task_id = stat.get('task_id', 0)
        if task_id not in task_lookup:
            continue
        
        instruction = task_lookup[task_id]['instruction']
        categories = categorize_task(instruction)
        
        changed_acc = stat.get('changed_accuracy', 0)
        unchanged_acc = stat.get('unchanged_accuracy', 0)
        
        for category in categories:
            category_accuracies[category]['changed'].append(changed_acc)
            category_accuracies[category]['unchanged'].append(unchanged_acc)
    
    # Create subplots
    categories = sorted([c for c in category_accuracies.keys() if category_accuracies[c]['changed']])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Accuracy Distribution by Task Type', fontsize=18, fontweight='bold')
    
    # Changed pixels accuracy
    changed_data = [category_accuracies[cat]['changed'] for cat in categories]
    bp1 = ax1.boxplot(changed_data, labels=categories, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax1.set_ylabel('Changed Pixels Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Changed Pixels Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.legend()
    
    # Unchanged pixels accuracy
    unchanged_data = [category_accuracies[cat]['unchanged'] for cat in categories]
    bp2 = ax2.boxplot(unchanged_data, labels=categories, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    
    ax2.set_ylabel('Unchanged Pixels Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Unchanged Pixels Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_task_complexity(stats, tasks, output_path):
    """Compare simple vs complex tasks"""
    task_lookup = {t['task_id']: t for t in tasks}
    
    simple_tasks = {'changed': [], 'unchanged': [], 'passed': 0, 'total': 0}
    complex_tasks = {'changed': [], 'unchanged': [], 'passed': 0, 'total': 0}
    
    for stat in stats:
        task_id = stat.get('task_id', 0)
        if task_id not in task_lookup:
            continue
        
        instruction = task_lookup[task_id]['instruction']
        categories = categorize_task(instruction)
        
        changed_acc = stat.get('changed_accuracy', 0)
        unchanged_acc = stat.get('unchanged_accuracy', 0)
        passed = stat.get('passed_lenient', False)
        
        if len(categories) == 1:
            simple_tasks['changed'].append(changed_acc)
            simple_tasks['unchanged'].append(unchanged_acc)
            simple_tasks['total'] += 1
            if passed:
                simple_tasks['passed'] += 1
        else:
            complex_tasks['changed'].append(changed_acc)
            complex_tasks['unchanged'].append(unchanged_acc)
            complex_tasks['total'] += 1
            if passed:
                complex_tasks['passed'] += 1
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Simple vs Complex Tasks', fontsize=18, fontweight='bold')
    
    # Accuracy comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    
    changed_means = [np.mean(simple_tasks['changed']), np.mean(complex_tasks['changed'])]
    unchanged_means = [np.mean(simple_tasks['unchanged']), np.mean(complex_tasks['unchanged'])]
    
    ax.bar(x - width/2, changed_means, width, label='Changed Pixels', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, unchanged_means, width, label='Unchanged Pixels', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple\n(1 category)', 'Complex\n(2+ categories)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, (c, u) in enumerate(zip(changed_means, unchanged_means)):
        ax.text(i - width/2, c + 2, f'{c:.1f}%', ha='center', fontweight='bold', fontsize=10)
        ax.text(i + width/2, u + 2, f'{u:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    # Pass rate comparison
    ax = axes[1]
    simple_pass_rate = (simple_tasks['passed'] / simple_tasks['total'] * 100) if simple_tasks['total'] > 0 else 0
    complex_pass_rate = (complex_tasks['passed'] / complex_tasks['total'] * 100) if complex_tasks['total'] > 0 else 0
    
    bars = ax.bar(['Simple', 'Complex'], [simple_pass_rate, complex_pass_rate],
                  color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pass Rate (>=95% threshold)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add labels
    for i, (bar, rate, passed, total) in enumerate(zip(bars, 
                                                         [simple_pass_rate, complex_pass_rate],
                                                         [simple_tasks['passed'], complex_tasks['passed']],
                                                         [simple_tasks['total'], complex_tasks['total']])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{passed}/{total}\n({rate:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_worst_categories(stats, tasks, output_path):
    """Show worst performing categories with examples"""
    task_lookup = {t['task_id']: t for t in tasks}
    
    category_stats = defaultdict(lambda: {
        'total': 0, 
        'failed': 0, 
        'avg_changed': [],
        'avg_unchanged': [],
        'worst_tasks': []
    })
    
    for stat in stats:
        task_id = stat.get('task_id', 0)
        if task_id not in task_lookup:
            continue
        
        instruction = task_lookup[task_id]['instruction']
        categories = categorize_task(instruction)
        passed = stat.get('passed_lenient', False)
        changed_acc = stat.get('changed_accuracy', 0)
        unchanged_acc = stat.get('unchanged_accuracy', 0)
        
        for category in categories:
            category_stats[category]['total'] += 1
            category_stats[category]['avg_changed'].append(changed_acc)
            category_stats[category]['avg_unchanged'].append(unchanged_acc)
            
            if not passed:
                category_stats[category]['failed'] += 1
                category_stats[category]['worst_tasks'].append({
                    'id': task_id,
                    'changed': changed_acc,
                    'unchanged': unchanged_acc,
                    'instruction': instruction[:60] + '...'
                })
    
    # Calculate failure rates and sort
    category_results = []
    for cat, data in category_stats.items():
        if data['total'] > 0:
            failure_rate = data['failed'] / data['total'] * 100
            avg_changed = np.mean(data['avg_changed'])
            avg_unchanged = np.mean(data['avg_unchanged'])
            
            category_results.append({
                'category': cat,
                'failure_rate': failure_rate,
                'avg_changed': avg_changed,
                'avg_unchanged': avg_unchanged,
                'total': data['total'],
                'failed': data['failed'],
                'worst_tasks': sorted(data['worst_tasks'], key=lambda x: x['changed'])[:3]
            })
    
    # Sort by failure rate
    category_results.sort(key=lambda x: x['failure_rate'], reverse=True)
    top_10 = category_results[:10]
    
    # Create plot with more space
    fig, ax = plt.subplots(figsize=(18, 12))
    
    y_pos = np.arange(len(top_10))
    failure_rates = [c['failure_rate'] for c in top_10]
    colors = plt.cm.RdYlGn_r(np.array(failure_rates) / 100)
    
    bars = ax.barh(y_pos, failure_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Create detailed labels with descriptions
    category_descriptions = {
        'Pattern': 'Creating/modifying repeating patterns, grids, arrays',
        'Rotation/Flip': 'Rotating or flipping objects/images',
        'Position': 'Moving objects to specific locations',
        'Geometric': 'Working with shapes (circles, squares, triangles)',
        'Color Change': 'Changing colors of objects or regions',
        'Size Change': 'Resizing, scaling objects',
        'Add/Remove': 'Adding or removing elements',
        'Background': 'Modifying background elements',
        'Text': 'Working with text, letters, numbers',
        'Gradient': 'Creating or modifying gradients',
        'Other': 'Uncategorized tasks'
    }
    
    labels = []
    for c in top_10:
        cat_name = c['category']
        description = category_descriptions.get(cat_name, '')
        
        # Format label with category name and description
        label = f"{cat_name}\n"
        if description:
            label += f"{description}\n"
        label += f"Failed: {c['failed']}/{c['total']} tasks  |  "
        label += f"Avg Accuracy: {c['avg_changed']:.1f}% (changed), {c['avg_unchanged']:.1f}% (unchanged)"
        labels.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, linespacing=1.5)
    ax.set_xlabel('Failure Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Most Challenging Task Types - Detailed Analysis', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for i, (bar, rate, cat) in enumerate(zip(bars, failure_rates, top_10)):
        width = bar.get_width()
        # Show failure rate
        ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                f'{rate:.1f}%',
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    # Add legend explaining what the metrics mean
    legend_text = (
        "Metrics Explained:\n"
        "• Failure Rate: % of tasks that didn't pass ≥95% accuracy threshold\n"
        "• Changed Accuracy: % of pixels that should change and did change correctly\n"
        "• Unchanged Accuracy: % of pixels that should stay the same and did"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return category_results

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize failures by task type')
parser.add_argument('--analysis-dir', type=str, required=True,
                    help='Directory containing analysis results')
parser.add_argument('--tasks-file', type=str, default='task_descriptions.jsonl',
                    help='Path to task descriptions file')
parser.add_argument('--output-dir', type=str, default='graphs',
                    help='Directory to save graphs')

args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

# Load data
stats_file = Path(args.analysis_dir) / "per_question_stats.jsonl"
stats = load_jsonl(stats_file)
tasks = load_jsonl(args.tasks_file)

if not stats or not tasks:
    print(f"Error: Could not load data")
    exit(1)

print(f"Loaded {len(stats)} task results and {len(tasks)} task descriptions")

# Generate plots
print("\nGenerating task type analysis...")
plot_failure_by_task_type(stats, tasks, output_dir / "failure_by_task_type.png")
plot_accuracy_by_task_type(stats, tasks, output_dir / "accuracy_by_task_type.png")
plot_task_complexity(stats, tasks, output_dir / "simple_vs_complex.png")
category_results = plot_worst_categories(stats, tasks, output_dir / "worst_categories.png")

# Print summary
print("\n" + "="*80)
print("TASK TYPE FAILURE SUMMARY")
print("="*80)
for i, cat in enumerate(category_results[:5], 1):
    print(f"\n{i}. {cat['category']}")
    print(f"   Failure Rate: {cat['failure_rate']:.1f}% ({cat['failed']}/{cat['total']})")
    print(f"   Avg Accuracy: Changed={cat['avg_changed']:.1f}%, Unchanged={cat['avg_unchanged']:.1f}%")

print(f"\n✓ All graphs saved to {output_dir}/")
