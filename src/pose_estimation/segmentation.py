import cv2
import numpy as np
from ultralytics import YOLO
import yaml

class Segmentation:
    def __init__(self, model_cfg, device='cuda'):
        """
        A class to perform instance segmentation using YOLO series of models
        Args: model_cfg (str): Path to model configuration file
              device (str): Device to run inference on (default: 'cuda')
        """
        self.device = device
        with open(model_cfg, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model = YOLO(self.config['model_weights'])
        self.model.to(self.device)

    def infer(self, rgb_image):
        """
        Runs the inference on a single image
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              conf (float): confidence threshold
        Returns: results (list): List of results containing bounding box coordinates, class labels, and confidence
        """
        # Load image
        results = self.model.predict(rgb_image, conf=self.config['confidence'])
        
        return results
    
    def infer_video(self, input_video_path, output_video_path):
        """
        Runs the inference on a video and saves the annotated video
        Args: input_video_path (str): Path to input video image size: (848, 480, 3)
              output_video_path (str): Path to save annotated video, output image size: (640, 480, 3)
        """
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = frame[:, 104:744, :]
            
            # YOLO instance segmentation prediction
            results = self.infer(resized_frame)

            annotated_frame = np.array(resized_frame)
            
            if results[0].masks is not None:
                for i, mask in enumerate(results[0].masks.data):
                    # Convert mask tensor to numpy array
                    mask_array = mask.cpu().numpy()
                    
                    # Resize mask to match frame dimensions
                    mask_array = cv2.resize(mask_array, (640, 480))
                    
                    # Create binary mask
                    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
                    
                    # Convert to 3-channel for overlay
                    colored_mask = np.zeros_like(annotated_frame)
                    
                    # Random color for each mask
                    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                    colored_mask[binary_mask == 255] = color
                    
                    # Overlay mask with transparency
                    alpha = 0.5
                    annotated_frame = cv2.addWeighted(
                        annotated_frame, 1, colored_mask, alpha, 0
                    )
                    
                    # Get bounding box coordinates if available
                    if results[0].boxes is not None and i < len(results[0].boxes):
                        box = results[0].boxes[i].xyxy.cpu().numpy().squeeze().astype(np.int32)
                        x1, y1, x2, y2 = box
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.tolist(), 2)
                        
                        # Add class label and confidence if available
                        if hasattr(results[0].boxes, 'cls') and hasattr(results[0].boxes, 'conf'):
                            cls_id = int(results[0].boxes[i].cls.item())
                            conf = results[0].boxes[i].conf.item()
                            class_name = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)


            out.write(annotated_frame)

        cap.release()
        out.release()




if __name__ == "__main__":
    Seg = Segmentation('/home/kshitij/Documents/Bell Pepper/pose_estimation/src/pose_estimation/yolov8l.yaml')
    Seg.infer_video('/home/kshitij/Documents/Bell Pepper/pose_estimation/rgb_data_sample.mp4', 'inference_video.mp4')

