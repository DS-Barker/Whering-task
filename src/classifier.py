from transformers import pipeline
from PIL import Image

def resize_to_max_dimension(image_path, max_dim=400):
    """
    Resize image so longest dimension is max_dim, maintaining aspect ratio
    Args:
        image_path: path to image
        max_dim: maximum dimension in pixels
    Returns: PIL Image
    """

    # TODO: More testing to optimise min size of image to balance speed and accuracy of classifications

    img = Image.open(str(image_path))
    
    # Get current dimensions
    width, height = img.size
    
    # Calculate scale factor
    if width > height:
        scale = max_dim / width
    else:
        scale = max_dim / height
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img_resized

class ClothingClassifier:
    def __init__(self):
        
        # General purpose image CNN model, well known with 1000 categories - high efficiency due to bottleneck design
        # TODO: Test multiple classification models, specific models for clothing/fashion would be more ideal for production 
        # TODO: More learning on these models also a possibility
        self.classifier = pipeline(
            "image-classification",
            model="microsoft/resnet-50"  
        )

        # Clothing-related labels
        self.clothing_keywords = {
            'shirt': ['jersey', 'sweatshirt', 'cardigan', 'tee shirt', 'polo shirt'],
            'trousers': ['jean', 'trouser', 'denim'],
            'dress': ['gown', 'kimono', 'abaya', 'academic gown'],
            'jacket': ['jacket', 'windbreaker', 'suit', 'trench coat'],
            'shoes': ['shoe', 'boot', 'sneaker', 'sandal', 'loafer', 'running shoe'],
            'skirt': ['skirt', 'miniskirt'],
            'sweater': ['sweater', 'cardigan', 'pullover'],
            'coat': ['fur coat', 'lab coat', 'trench coat'],
            'accessory': ['tie', 'bow tie', 'scarf', 'sunglasses', 'sunglass']
        }

        # Flatten to a set of all allowed keywords
        self.all_clothing_keywords = set()
        for keywords in self.clothing_keywords.values():
            self.all_clothing_keywords.update(keywords)
    
    def classify(self, image_path, bbox=None):
        """Classify type of clothing"""

        # Resize first for speed
        img = resize_to_max_dimension(image_path, max_dim=400)
        
        # Crop if bbox provided
        if bbox:
            img = img.crop(bbox)
        
        # Pass PIL Image to pipeline, giving top 10 predictions - Increases the chances of finding an item of clothing
        results = self.classifier(img, top_k=10)
        
        # Find first clothing-related result, highest confidence
        for result in results:
            label_lower = result['label'].lower()
            
            # Check if this label contains any clothing keyword
            if any(keyword in label_lower for keyword in self.all_clothing_keywords):
                category = self._simplify_category(result['label'])
                return category, result['score']
        
        # If no clothing found in top 10, return generic "person" as this is most likely what will be found based on detector categories
        return 'person', results[0]['score']
    
    def _simplify_category(self, label):
        """Map labels to clothing categories"""
        
        label_lower = label.lower()
        
        for category, keywords in self.clothing_keywords.items():
            if any(kw in label_lower for kw in keywords):
                return category
        
        # Fallback
        return 'person'