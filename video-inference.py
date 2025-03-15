import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


model = YOLO("/home/kshitij/Documents/Bell Pepper/pose_estimation/best.pt")


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = frame[:, 104:744, :]
        
        # YOLO instance segmentation prediction
        results = model.predict(resized_frame, conf=0.6)

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
                        class_name = model.names[cls_id] if cls_id in model.names else str(cls_id)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)


        out.write(annotated_frame)

    cap.release()
    out.release()

process_video('/home/kshitij/Documents/Bell Pepper/yolov8-obb/rgb_data_sample.mp4', 'inference_video_yolov8l.mp4')