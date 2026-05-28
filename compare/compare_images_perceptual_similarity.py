import json
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate perceptual similarity metrics')
parser.add_argument('--output-dir', type=str, default='output', 
                    help='Directory containing output images (default: output)')
parser.add_argument('--images-dir', type=str, default='images', 
                    help='Directory containing answer images (default: images)')
parser.add_argument('--results-file', type=str, default=None,
                    help='Output JSON file (default: <output-dir>/perceptual_similarity_results.json)')

args = parser.parse_args()

# Setup directories
images_dir = Path(args.images_dir)
output_dir = Path(args.output_dir)

# Set default results file if not specified
if args.results_file:
    results_file = Path(args.results_file)
else:
    results_file = output_dir / "perceptual_similarity_results.json"

print(f"Output directory: {output_dir}")
print(f"Images directory: {images_dir}")
print(f"Results file: {results_file}")
print("="*70)

def calculate_ssim(img1_path, img2_path):
    """Calculate Structural Similarity Index (SSIM)"""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate SSIM for each channel
        ssim_scores = []
        for i in range(3):  # RGB channels
            score = ssim(arr1[:,:,i], arr2[:,:,i])
            ssim_scores.append(score)
        
        # Average SSIM across channels
        avg_ssim = np.mean(ssim_scores)
        
        return avg_ssim
    except Exception as e:
        print(f"    Error calculating SSIM: {e}")
        return None

def calculate_histogram_similarity(img1_path, img2_path):
    """Calculate histogram-based similarity (cosine similarity)"""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Calculate histograms
        hist1 = np.array(img1.histogram()).astype(float)
        hist2 = np.array(img2.histogram()).astype(float)
        
        # Normalize
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(hist1, hist2)
        
        return similarity
    except Exception as e:
        print(f"    Error calculating histogram similarity: {e}")
        return None

def calculate_mse(img1_path, img2_path):
    """Calculate Mean Squared Error"""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Convert to numpy arrays
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        
        # Calculate MSE
        mse = np.mean((arr1 - arr2) ** 2)
        
        # Normalize to 0-1 scale (lower is better, so we invert it)
        # Max MSE would be if one image is all black and other all white
        max_mse = 255 ** 2
        normalized_similarity = 1 - (mse / max_mse)
        
        return normalized_similarity
    except Exception as e:
        print(f"    Error calculating MSE: {e}")
        return None

# Find all output images
output_images = sorted(output_dir.glob("q*_output.png"))

if not output_images:
    print(f"No output images found in {output_dir}/q*_output.png")
    exit(1)

print(f"Found {len(output_images)} output images")
print("Calculating perceptual similarity metrics...")
print("="*70)

results = []

for output_path in output_images:
    # Extract task ID
    task_id = int(output_path.stem.split('_')[0][1:])
    
    # Find corresponding answer image
    answer_path = images_dir / f"q{task_id}_answer.png"
    
    if not answer_path.exists():
        print(f"Task {task_id}: No answer image found")
        continue
    
    print(f"Task {task_id}: Calculating metrics...", end=" ")
    
    # Calculate all metrics
    ssim_score = calculate_ssim(output_path, answer_path)
    hist_sim = calculate_histogram_similarity(output_path, answer_path)
    mse_sim = calculate_mse(output_path, answer_path)
    
    if ssim_score is not None:
        print(f"SSIM: {ssim_score:.4f}, Hist: {hist_sim:.4f}, MSE: {mse_sim:.4f}")
        results.append({
            'task_id': task_id,
            'ssim': ssim_score,
            'histogram_similarity': hist_sim,
            'mse_similarity': mse_sim,
            'status': 'success'
        })
    else:
        print("Failed")
        results.append({
            'task_id': task_id,
            'status': 'error'
        })

# Save results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Load pixel matching results for comparison (if exists)
pixel_results_file = output_dir / "comparison_results.json"
pixel_matches = {}

if pixel_results_file.exists():
    try:
        with open(pixel_results_file, 'r') as f:
            pixel_results = json.load(f)
        pixel_matches = {r['task_id']: r['match_percentage'] for r in pixel_results if r.get('match_percentage') is not None}
    except Exception as e:
        print(f"Note: Could not load pixel matching results: {e}")

# Print summary
print("\n" + "="*70)
print("PERCEPTUAL SIMILARITY SUMMARY")
print("="*70)

valid_results = [r for r in results if r['status'] == 'success']

if valid_results:
    ssim_scores = [r['ssim'] for r in valid_results]
    hist_scores = [r['histogram_similarity'] for r in valid_results]
    mse_scores = [r['mse_similarity'] for r in valid_results]
    
    print(f"\nTotal comparisons: {len(valid_results)}")
    
    print(f"\nSSIM (Structural Similarity, 0-1, higher is better):")
    print(f"  Average: {np.mean(ssim_scores):.4f}")
    print(f"  Min: {np.min(ssim_scores):.4f}")
    print(f"  Max: {np.max(ssim_scores):.4f}")
    print(f"  Median: {np.median(ssim_scores):.4f}")
    
    print(f"\nHistogram Similarity (0-1, higher is better):")
    print(f"  Average: {np.mean(hist_scores):.4f}")
    print(f"  Min: {np.min(hist_scores):.4f}")
    print(f"  Max: {np.max(hist_scores):.4f}")
    print(f"  Median: {np.median(hist_scores):.4f}")
    
    print(f"\nMSE Similarity (0-1, higher is better):")
    print(f"  Average: {np.mean(mse_scores):.4f}")
    print(f"  Min: {np.min(mse_scores):.4f}")
    print(f"  Max: {np.max(mse_scores):.4f}")
    print(f"  Median: {np.median(mse_scores):.4f}")
    
    # Compare with pixel matching if available
    if pixel_matches:
        print(f"\n" + "="*70)
        print("COMPARISON: Perceptual Metrics vs Pixel Matching")
        print("="*70)
        
        # Find tasks where perceptual metrics are much better than pixel matching
        improvements = []
        for r in valid_results:
            task_id = r['task_id']
            if task_id in pixel_matches:
                pixel_match = pixel_matches[task_id] / 100  # Convert to 0-1 scale
                ssim_improvement = r['ssim'] - pixel_match
                improvements.append({
                    'task_id': task_id,
                    'pixel_match': pixel_match,
                    'ssim': r['ssim'],
                    'improvement': ssim_improvement
                })
        
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        
        print(f"\nTop 10 tasks where SSIM is much better than pixel matching:")
        print("(These tasks are perceptually similar but not pixel-perfect)")
        for imp in improvements[:10]:
            print(f"  Task {imp['task_id']}: Pixel={imp['pixel_match']*100:.1f}%, SSIM={imp['ssim']*100:.1f}% (diff: +{imp['improvement']*100:.1f}%)")
        
        print(f"\nTasks where SSIM is also low (genuinely poor results):")
        low_ssim = sorted(valid_results, key=lambda x: x['ssim'])[:10]
        for r in low_ssim:
            task_id = r['task_id']
            pixel = pixel_matches.get(task_id, 0)
            print(f"  Task {task_id}: Pixel={pixel:.1f}%, SSIM={r['ssim']*100:.1f}%")

print(f"\nResults saved to: {results_file}")
