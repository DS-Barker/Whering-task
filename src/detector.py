from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from classifier import resize_to_max_dimension

class ClothingDetector:
    def __init__(self):

        # Well known efficient model trained on COCO 2017 dataset
        # TODO: Test multiple classification models, specific models for clothing/fashion would be more ideal for production 
        # TODO: More learning on these models also a possibility
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
        # Limit detections to COCO person + a few clothing based categories - most likely finding a human from the images chosen
        self.allowed_labels = {'person', 'tie', 'handbag', 'backpack'}

    def detect(self, image_path):
        """Detect person regions (proxy for clothing)"""
        
        image = resize_to_max_dimension(image_path, max_dim=400)
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=0.5                           # TODO: Test this more to fine tune the thresholding at which something is considered
        )[0]
        
        detections = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            
            # Only keep allowed detections
            if label_name.lower() in self.allowed_labels:
                detections.append({
                    'bbox': box.tolist(),
                    'label': 'clothing_region',
                    'confidence': score.item()
                })
        
        return detections