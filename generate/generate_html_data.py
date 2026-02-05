import json
import argparse
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    except:
        pass
    return results

def load_template(template_path):
    """Load HTML template"""
    with open(template_path, 'r') as f:
        return f.read()

def render_template(template, **kwargs):
    """Simple template rendering"""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{{ {key} }}}}", str(value))
    return result

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate HTML visualization from results')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory containing results files')
parser.add_argument('--images-dir', type=str, default='images',
                    help='Directory containing images')
parser.add_argument('--analysis-dir', type=str, default='analysis_results',
                    help='Directory containing analysis results')
parser.add_argument('--html-output', type=str, default='results.html',
                    help='Output HTML file')
parser.add_argument('--title', type=str, default='Image Results',
                    help='Title for the HTML page')
parser.add_argument('--template', type=str, default='template.html',
                    help='Path to HTML template file')

args = parser.parse_args()

# Load template
template = load_template(args.template)

# Load global stats
global_stats = {}
global_stats_path = Path(args.analysis_dir) / "global_stats.json"
if global_stats_path.exists():
    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)

# Load per-question stats
per_question_stats = load_jsonl(Path(args.analysis_dir) / "per_question_stats.jsonl")

# Generate question cards HTML
question_cards = ""
for stat in per_question_stats:
    task_id = stat.get('task_id')
    if task_id is None:
        print(f"Warning: Could not find task_id in stat: {stat.keys()}")
        continue
    
    # Format question ID (e.g., task_id 1 -> "q1")
    q_id = f"q{task_id}"
    
    # Image paths
    input_img = f"{args.images_dir}/{q_id}_input.png"
    answer_img = f"{args.images_dir}/{q_id}_answer.png"
    output_img = f"{args.output_dir}/{q_id}_output_resized.png"
    changed_img = f"{args.analysis_dir}/{q_id}_changed_pixels.png"
    correctness_img = f"{args.analysis_dir}/{q_id}_correctness_regions.png"
    
    # Stats (note the different key names in your data)
    # The values are already in percentage form, so don't multiply by 100
    changed_acc = stat.get('changed_accuracy', 0)
    unchanged_acc = stat.get('unchanged_accuracy', 0)
    is_perfect = stat.get('passed_perfect', False)
    is_strict = stat.get('passed_strict', False)
    is_lenient = stat.get('passed_lenient', False)
    
    # Determine badge
    if is_perfect:
        badge = '<span class="badge perfect">Perfect &#10003;</span>'
    elif is_strict:
        badge = '<span class="badge strict">Strict &#10003;</span>'
    elif is_lenient:
        badge = '<span class="badge lenient">Lenient &#10003;</span>'
    else:
        badge = '<span class="badge failed">Failed &#10007;</span>'
    
    question_cards += f"""
    <div class="question-card">
        <div class="question-header">
            <h2>{q_id}</h2>
            {badge}
        </div>
        
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Changed Pixels Accuracy</div>
                <div class="stat-value">{changed_acc:.2f}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Unchanged Pixels Accuracy</div>
                <div class="stat-value">{unchanged_acc:.2f}%</div>
            </div>
        </div>
        
        <div class="images-grid">
            <div class="image-container">
                <div class="image-label">Input</div>
                <img src="{input_img}" alt="Input">
            </div>
            <div class="image-container">
                <div class="image-label">Expected Answer</div>
                <img src="{answer_img}" alt="Answer">
            </div>
            <div class="image-container">
                <div class="image-label">Model Output</div>
                <img src="{output_img}" alt="Output">
            </div>
            <div class="image-container">
                <div class="image-label">Changed Pixels</div>
                <img src="{changed_img}" alt="Changed Pixels">
            </div>
            <div class="image-container">
                <div class="image-label">Correctness Map</div>
                <img src="{correctness_img}" alt="Correctness">
            </div>
        </div>
    </div>
    """

# Global stats summary
total = global_stats.get('total_questions', 0)
perfect = global_stats.get('perfect_matches', 0)
strict = global_stats.get('strict_correct', 0)
lenient = global_stats.get('lenient_correct', 0)

perfect_pct = f"{(perfect / total * 100):.1f}" if total > 0 else "0.0"
strict_pct = f"{(strict / total * 100):.1f}" if total > 0 else "0.0"
lenient_pct = f"{(lenient / total * 100):.1f}" if total > 0 else "0.0"

# Render template
html_content = render_template(
    template,
    title=args.title,
    total=total,
    perfect=perfect,
    perfect_pct=perfect_pct,
    strict=strict,
    strict_pct=strict_pct,
    lenient=lenient,
    lenient_pct=lenient_pct,
    question_cards=question_cards
)

# Write HTML with UTF-8 encoding
with open(args.html_output, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML generated: {args.html_output}")
print(f"Total questions processed: {total}")
print(f"Perfect matches: {perfect} ({perfect_pct}%)")
print(f"Strict correct: {strict} ({strict_pct}%)")
print(f"Lenient correct: {lenient} ({lenient_pct}%)")
