import cv2
import numpy as np
from ultralytics import YOLO
import yaml

class Segmentation:
    def __init__(self, model_cfg, device='cuda'):
        self.device = device
        with open(model_cfg, 'r') as file:
            config = yaml.safe_load(file)
        self.model = YOLO(config['model_weights'])
        self.model.to(self.device)

    def infer(self, image_path):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        # Perform inference
        results = self.model(img)
        
        # Process results
        results.render()  # Render the results on the image
        segmented_image = results.imgs[0]  # Get the image with the results
        
        return segmented_image

# Example usage:
# segmentation = Segmentation(model_path='path_to_your_model.pt')
# segmented_image = segmentation.infer('path_to_your_image.jpg')
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()