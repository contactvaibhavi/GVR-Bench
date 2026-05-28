import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(file_path):
    """Load JSONL file"""
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return results

def categorize_failures(stats):
    """Categorize by actual failure patterns"""
    failure_types = {
        'over_editing': [],      # Changed everything (changed_pixel_ratio ~100%)
        'under_editing': [],     # Changed too few (low changed_correct)
        'wrong_pixels': [],      # Changed wrong pixels (high changed_incorrect)
        'hallucination': [],     # Changed unchanged areas (high unchanged_incorrect)
        'correct': []            # Passed lenient
    }
    
    for stat in stats:
        task_id = stat.get('task_id', 0)
        changed_ratio = stat.get('changed_pixel_ratio', 0)
        changed_correct = stat.get('changed_correct', 0)
        changed_incorrect = stat.get('changed_incorrect', 0)
        unchanged_incorrect = stat.get('unchanged_incorrect', 0)
        total_changed = stat.get('total_changed_pixels', 1)
        total_unchanged = stat.get('total_unchanged_pixels', 1)
        
        data = {
            'task_id': task_id,
            'changed_acc': stat.get('changed_accuracy', 0),
            'unchanged_acc': stat.get('unchanged_accuracy', 0),
            'changed_ratio': changed_ratio,
            'changed_correct': changed_correct,
            'changed_incorrect': changed_incorrect,
            'unchanged_incorrect': unchanged_incorrect
        }
        
        if stat.get('passed_lenient', False):
            failure_types['correct'].append(data)
        # Over-editing: changed nearly everything (>90% of image)
        elif changed_ratio > 90:
            failure_types['over_editing'].append(data)
        # Hallucination: barely changed what should change, but changed a lot of what shouldn't
        elif unchanged_incorrect > total_unchanged * 0.5:
            failure_types['hallucination'].append(data)
        # Under-editing: didn't change much of what should change
        elif changed_correct < total_changed * 0.3:
            failure_types['under_editing'].append(data)
        # Wrong pixels: changed some pixels but mostly wrong ones
        else:
            failure_types['wrong_pixels'].append(data)
    
    return failure_types

def plot_failure_types_summary(failure_types, output_path):
    """Bar chart of failure type distribution"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = [
        'Correct\n(Passed)',
        'Under-Editing\n(Missed targets)',
        'Over-Editing\n(Changed >90%)',
        'Wrong Pixels\n(Changed wrong areas)',
        'Hallucination\n(Added artifacts)'
    ]
    
    counts = [
        len(failure_types['correct']),
        len(failure_types['under_editing']),
        len(failure_types['over_editing']),
        len(failure_types['wrong_pixels']),
        len(failure_types['hallucination'])
    ]
    
    colors = ['#28a745', '#ffc107', '#ff6b6b', '#e67e22', '#dc3545']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add count labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count/total*100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Number of Tasks', fontsize=14, fontweight='bold')
    ax.set_title('Image Editing Failure Type Distribution', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_failure_characteristics(failure_types, output_path):
    """Show characteristics of each failure type"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Failure Type Characteristics', fontsize=20, fontweight='bold')
    
    configs = [
        ('correct', 'Correct', '#28a745'),
        ('under_editing', 'Under-Editing', '#ffc107'),
        ('over_editing', 'Over-Editing', '#ff6b6b'),
        ('wrong_pixels', 'Wrong Pixels', '#e67e22'),
        ('hallucination', 'Hallucination', '#dc3545')
    ]
    
    for idx, (ftype, title, color) in enumerate(configs):
        if idx >= 5:
            break
        ax = axes[idx // 3, idx % 3]
        data = failure_types[ftype]
        
        if not data:
            ax.text(0.5, 0.5, f'No cases', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(f'{title} (0)', fontweight='bold', fontsize=14)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            continue
        
        changed_accs = [d['changed_acc'] for d in data]
        unchanged_accs = [d['unchanged_acc'] for d in data]
        task_ids = [d['task_id'] for d in data]
        
        # Scatter plot
        scatter = ax.scatter(changed_accs, unchanged_accs, s=200, c=color, alpha=0.6,
                            edgecolors='black', linewidth=2)
        
        # Add task labels
        for i, tid in enumerate(task_ids):
            ax.annotate(f'q{tid}', (changed_accs[i], unchanged_accs[i]),
                       fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Add reference lines
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axvline(x=95, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
        
        ax.set_xlabel('Changed Pixels Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Unchanged Pixels Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} ({len(data)} cases)', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
    
    # Hide last subplot if not needed
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_failure_details(failure_types, output_path):
    """Detailed view of what's wrong in each failure type"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Failure Analysis', fontsize=18, fontweight='bold')
    
    # 1. Changed pixel ratio by failure type
    ax = axes[0, 0]
    failure_names = ['Correct', 'Under-Edit', 'Over-Edit', 'Wrong Pixels', 'Hallucination']
    colors = ['#28a745', '#ffc107', '#ff6b6b', '#e67e22', '#dc3545']
    
    for idx, (ftype, fname, color) in enumerate(zip(
        ['correct', 'under_editing', 'over_editing', 'wrong_pixels', 'hallucination'],
        failure_names, colors
    )):
        data = failure_types[ftype]
        if data:
            ratios = [d['changed_ratio'] for d in data]
            ax.scatter([idx] * len(ratios), ratios, s=100, c=color, alpha=0.6,
                      edgecolors='black', linewidth=1.5, label=fname)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(failure_names, rotation=45, ha='right')
    ax.set_ylabel('% of Image Changed', fontsize=12, fontweight='bold')
    ax.set_title('Changed Pixel Ratio by Failure Type', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-5, 105)
    
    # 2. Correct vs Incorrect pixels
    ax = axes[0, 1]
    for idx, (ftype, fname, color) in enumerate(zip(
        ['correct', 'under_editing', 'over_editing', 'wrong_pixels', 'hallucination'],
        failure_names, colors
    )):
        data = failure_types[ftype]
        if data:
            for d in data:
                total_changed = d['changed_correct'] + d['changed_incorrect']
                if total_changed > 0:
                    correct_pct = d['changed_correct'] / total_changed * 100
                    ax.bar(idx, correct_pct, color=color, alpha=0.6, width=0.7)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(failure_names, rotation=45, ha='right')
    ax.set_ylabel('% Correct Among Changed Pixels', fontsize=12, fontweight='bold')
    ax.set_title('Quality of Changes', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 3. Task distribution
    ax = axes[1, 0]
    all_tasks = []
    all_colors = []
    all_labels = []
    
    for ftype, fname, color in zip(
        ['correct', 'under_editing', 'over_editing', 'wrong_pixels', 'hallucination'],
        failure_names, colors
    ):
        data = failure_types[ftype]
        for d in data:
            all_tasks.append(d['task_id'])
            all_colors.append(color)
            all_labels.append(fname)
    
    if all_tasks:
        sorted_indices = np.argsort(all_tasks)
        sorted_tasks = [all_tasks[i] for i in sorted_indices]
        sorted_colors = [all_colors[i] for i in sorted_indices]
        
        ax.bar(range(len(sorted_tasks)), [1]*len(sorted_tasks), color=sorted_colors,
              edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Task ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Failure Type', fontsize=12, fontweight='bold')
        ax.set_title('Tasks by Failure Type', fontweight='bold')
        ax.set_xticks(range(0, len(sorted_tasks), max(1, len(sorted_tasks)//20)))
        ax.set_xticklabels([f'q{sorted_tasks[i]}' for i in range(0, len(sorted_tasks), max(1, len(sorted_tasks)//20))], 
                          rotation=45, ha='right')
    
    # 4. Legend and summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Failure Type Definitions:\n\n"
    summary_text += "Correct: Passed >=95% threshold\n\n"
    summary_text += "Under-Editing:\n  Missed most target changes\n\n"
    summary_text += "Over-Editing:\n  Changed >90% of image\n\n"
    summary_text += "Wrong Pixels:\n  Changed wrong areas\n\n"
    summary_text += "Hallucination:\n  Added unwanted artifacts\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=13, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize analysis results by failure type')
parser.add_argument('--analysis-dir', type=str, required=True,
                    help='Directory containing analysis results')
parser.add_argument('--output-dir', type=str, default='graphs',
                    help='Directory to save graphs (default: graphs)')

args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

# Load data
stats_file = Path(args.analysis_dir) / "per_question_stats.jsonl"
stats = load_jsonl(stats_file)

if not stats:
    print(f"Error: No data found in {stats_file}")
    exit(1)

print(f"Loaded {len(stats)} task results")

# Categorize failures
failure_types = categorize_failures(stats)

print(f"\nFailure type breakdown:")
for ftype, data in failure_types.items():
    print(f"  {ftype.replace('_', ' ').title()}: {len(data)} tasks")

# Generate plots
print("\nGenerating visualizations...")
plot_failure_types_summary(failure_types, output_dir / "failure_types.png")
plot_failure_characteristics(failure_types, output_dir / "failure_characteristics.png")
plot_failure_details(failure_types, output_dir / "failure_details.png")

print(f"\n✓ All graphs saved to {output_dir}/")
