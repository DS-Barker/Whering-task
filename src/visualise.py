import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from colour import resize_to_max_dimension

def visualise_results(num_examples=10, output_all=False):
    """
    Create annotated images showing detections, classifications, and colours
    
    Args:
        num_examples: Number of images to process (default 10)
        output_all: If True, process all images regardless of num_examples
    """
    # Load results
    with open("data/output.json") as f:
        results = json.load(f)
    
    # Group by filename
    by_file = {}
    for r in results:
        fname = r['filename']
        by_file.setdefault(fname, []).append(r)
    
    # Create output directory
    output_dir = Path("output/annotated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine how many to process
    items_to_process = list(by_file.items())
    if not output_all:
        items_to_process = items_to_process[:num_examples]
    
    # Visualise images
    for _, (fname, detections) in enumerate(items_to_process):
        img_path = Path("data/images") / fname

        img = resize_to_max_dimension(img_path, max_dim=400)
        
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)
        
        for det in detections:
            if det['bbox']:
                # Draw bounding box (coordinates now match downscaled image)
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label with category, colour, confidence
                label = f"{det['category']} ({det['colour']})\nConf: {det['confidence']:.2f}"
                
                ax.text(
                    x1, y1 - 10, label,
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', pad=3)
                )
            else:
                # No bbox - just show whole-image classification
                label = f"Whole image: {det['category']} ({det['colour']})\nConf: {det['confidence']:.2f}"
                
                ax.text(
                    10, 30, label,
                    color='white', fontsize=14, weight='bold',
                    bbox=dict(facecolor='blue', alpha=0.8, edgecolor='none', pad=5)
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        output_path = output_dir / fname
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    count = len(items_to_process)
    print(f"\nVisualisation complete! {count} images saved to {output_dir}")

if __name__ == "__main__":
    # Change output_all=True to process all images
    visualise_results(num_examples=10, output_all=True)