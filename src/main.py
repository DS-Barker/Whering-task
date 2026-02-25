from pathlib import Path
from detector import ClothingDetector
from classifier import ClothingClassifier
from colour import get_dominant_colour
import json

def process_image(image_path, detector, classifier):
    """Process single image through full pipeline"""
    
    # Try detection first
    detections = detector.detect(image_path)
    
    results = []
    
    if not detections:
        # Fallback: whole-image processing
        category, conf = classifier.classify(image_path)
        colour = get_dominant_colour(image_path)
        results.append({
            'filename': image_path.name,
            'bbox': None,
            'category': category,
            'confidence': conf,
            'colour': colour
        })
    else:
        # Process each detected region
        for det in detections:
            category, conf = classifier.classify(image_path, det['bbox'])
            colour = get_dominant_colour(image_path, det['bbox'])
            
            results.append({
                'filename': image_path.name,
                'bbox': det['bbox'],
                'category': category,
                'confidence': conf,
                'colour': colour,
                'detection_confidence': det['confidence']
            })
    
    return results

def main():
    detector = ClothingDetector()
    classifier = ClothingClassifier()
    
    all_results = []
    
    for img_path in sorted(Path("data/images").glob("*.jpg")):
        print(f"Processing {img_path.name}...")
        results = process_image(img_path, detector, classifier)
        all_results.extend(results)
    
    # Save to JSON
    with open("data/output.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessed {len(all_results)} items")

if __name__ == "__main__":
    main()