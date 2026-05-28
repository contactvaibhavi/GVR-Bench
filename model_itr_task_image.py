import google.generativeai as genai
from PIL import Image
import io
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Setup directories
input_dir = Path("images")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Use Nano Banana (gemini-2.5-flash-image) for image generation
model_name = "gemini-2.5-flash-image"
model = genai.GenerativeModel(model_name)

# Read task descriptions
tasks = []
with open("task_descriptions.jsonl", "r") as f:
    for line in f:
        task = json.loads(line.strip())
        tasks.append(task)

print(f"Loaded {len(tasks)} tasks from task_descriptions.jsonl")
print("="*70)

# Process each task
success_count = 0
fail_count = 0
skip_count = 0

for task in tasks:
    task_id = task["task_id"]
    instruction = task["instruction"]
    input_description = task.get("input_description", "")
    
    # Find corresponding input image
    image_path = input_dir / f"q{task_id}_input.png"
    
    if not image_path.exists():
        print(f"\n⚠ Task {task_id}: No image found at {image_path}")
        skip_count += 1
        continue
    
    try:
        print(f"\n[Task {task_id}] Processing: {image_path.name}")
        print(f"  Instruction: {instruction[:100]}...")
        
        # Load input image
        image = Image.open(image_path)
        
        # Create full prompt with context
        full_prompt = f"""Input image description: {input_description}

Task: {instruction}

Generate the edited image following these instructions exactly."""
        
        # Generate content with both text and image input
        response = model.generate_content([full_prompt, image])
        
        print(f"  ✓ Generated response")
        
        # Extract and save generated images
        saved = False
        for i, part in enumerate(response.parts):
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                output_image = Image.open(io.BytesIO(image_data))
                
                # Create output filename matching the pattern
                output_filename = output_dir / f"q{task_id}_output.png"
                output_image.save(output_filename)
                print(f"  ✓ Saved: {output_filename}")
                saved = True
                success_count += 1
            elif hasattr(part, 'text') and part.text:
                print(f"  Text: {part.text[:80]}...")
        
        if not saved:
            print(f"  ⚠ No image generated")
            fail_count += 1
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:150]}")
        fail_count += 1

print("\n" + "="*70)
print(f"Processing complete!")
print(f"  Success: {success_count}")
print(f"  Failed: {fail_count}")
print(f"  Skipped: {skip_count}")
print(f"Check {output_dir}/ for results (q<ID>_output.png)")
