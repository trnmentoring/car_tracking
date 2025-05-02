import os
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class TrackedObject:
    """TrackedObject class that maintains tracking information for a single object"""
    def __init__(self, track_id: int, class_id: int, 
             position: Tuple[float, float, float, float], 
             confidence: float, frame_id: int):
        """Initialize a new tracked object"""
        self.id = track_id
        self.class_id = class_id
        self.position = position  # Current position (x1, y1, x2, y2)
        self.confidence = confidence
        self.first_frame = frame_id
        self.last_frame = frame_id
        
        # Calculate center of bounding box
        x1, y1, x2, y2 = position
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Flag to track if this object has a current detection in this frame
        self.has_current_detection = True
        
        # Initialize position history with the first position
        # Format: [(frame_id, x1, y1, x2, y2), ...]
        self.position_history = [(frame_id, *position)]
    
    def update(self, position: Tuple[float, float, float, float], confidence: float, frame_id: int) -> None:
        """Update object position and history"""
        self.position = position
        self.confidence = confidence
        self.last_frame = frame_id
        
        # Calculate center of bounding box
        x1, y1, x2, y2 = position
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Add current position to history
        self.position_history.append((frame_id, *position))
        
        # Limit history length to avoid excessive memory usage
        max_history_length = 100
        if len(self.position_history) > max_history_length:
            self.position_history = self.position_history[-max_history_length:]
        
        # Mark as having a current detection
        self.has_current_detection = True
    
    def get_position(self) -> Tuple[float, float, float, float]:
        """Get current bounding box position"""
        return self.position
    
    def get_center(self) -> Tuple[float, float]:
        """Get current center position of object"""
        return self.center
    
    def mark_no_detection(self) -> None:
        """Mark this object as not having a detection in the current frame"""
        self.has_current_detection = False
    
    def __str__(self) -> str:
        """String representation of the object"""
        return f"Object(id={self.id}, pos={self.position}, frames={self.first_frame}-{self.last_frame})"


class DetectorTracker:
    """Combined detector and tracker class for YOLOv8 object tracking"""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the detector and tracker"""
        # Set default parameters
        self.params = params
        
        # Force CPU usage if specified
        if self.params["device"] == "cpu":
            torch.set_default_device('cpu')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model from {self.params['model_path']} on {self.params['device']}...")
        self.model = YOLO(self.params["model_path"])
        
        # Set model to evaluation mode on specified device
        self.model.to(self.params["device"])
        
        # Initialize tracking state
        self.reset()
        
        print("DetectorTracker initialized successfully!")
    
    def reset(self):
        """Reset tracker state for a new video"""
        self.objects = {}
        
        # Frame counter
        self.frame_count = 0
        
        # Store all class colors
        self.class_colors = {}
        
        print("Tracker reset for new video")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a new frame: detect, track, and visualize"""
        # Store original frame
        original_frame = frame.copy()
        
        # Increment frame counter
        self.frame_count += 1
        
        # Mark all existing objects as not having a current detection
        for obj in self.objects.values():
            obj.mark_no_detection()
        
        # Run detection with tracking
        results = self.model.track(
            source=frame,
            conf=self.params["conf_thres"],
            iou=self.params["iou_thres"],
            device=self.params["device"],
            tracker=self.params["tracker_config"],
            persist=True,
            verbose=False,
            agnostic_nms=self.params["agnostic_nms"],
            imgsz=self.params["imgsz"]
        )
        
        # Clear objects not seen for max_age frames
        if self.frame_count % 5 == 0:
            self._clean_old_tracks()

        # Process tracking results
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            
            # If we have tracking IDs available
            if hasattr(boxes, 'id') and boxes.id is not None:
                # Get all detection data
                track_ids = boxes.id.int().cpu().tolist()
                bboxes = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                
                # Get target classes from parameters
                target_classes = self.params.get("target_classes", [3, 4, 5])
                
                # Process each detection - filter by target classes
                for idx, track_id in enumerate(track_ids):
                    bbox = bboxes[idx]
                    conf = confs[idx]
                    cls_id = cls_ids[idx]
                    cls_name = results[0].names[cls_id]
                    
                    # Skip classes that are not in target classes
                    if cls_id not in target_classes:
                        continue
                    
                    # Calculate center
                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Treat all target classes as a single class
                    unified_class_id = 1  # Use a single class ID for all target classes
                    
                    # Update or create object
                    if track_id in self.objects:
                        self.objects[track_id].update(bbox, conf, self.frame_count)
                    else:
                        new_object = TrackedObject(
                            track_id=track_id,
                            class_id=unified_class_id,  # Use unified class ID
                            position=bbox,
                            confidence=conf,
                            frame_id=self.frame_count
                        )
                        
                        self.objects[track_id] = new_object
                        
                        # Assign color for this class if not already assigned
                        if cls_id not in self.class_colors:
                            # Generate a random color for this class
                            self.class_colors[cls_id] = (
                                np.random.randint(0, 255),
                                np.random.randint(0, 255),
                                np.random.randint(0, 255)
                            )
        
        # Draw tracking results on original frame
        annotated_frame = self._draw_tracking_results(original_frame)
        
        return annotated_frame
    
    def get_vehicles(self) -> List[Dict[str, Any]]:
        """Returns a list of tracked vehicles with their identifiers, positions, and tracking history."""
        
        vehicles = []
        
        for obj in self.objects.values():
            # Only include objects with current detections
            if obj.has_current_detection:
                # Create a dictionary with relevant vehicle information
                vehicle_info = {
                    'id': obj.id,
                    'position': obj.position,
                    'center': obj.center,
                    'first_frame': obj.first_frame,
                    'last_frame': obj.last_frame,
                    'confidence': obj.confidence,
                    'has_current_detection': obj.has_current_detection,
                    'position_history': obj.position_history
                }
                
                vehicles.append(vehicle_info)
        
        return vehicles
    
    def _clean_old_tracks(self) -> None:
        """Remove objects that haven't been seen for max_age frames"""
        objects_to_remove = []
        
        for track_id, obj in self.objects.items():
            age = self.frame_count - obj.last_frame
            if age > self.params["max_age"]:
                objects_to_remove.append(track_id)
        
        for track_id in objects_to_remove:
            del self.objects[track_id]
    
    def _draw_tracking_results(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        # Draw only objects with current detections
        for obj in self.objects.values():
            if not obj.has_current_detection:
                continue
                
            # Get current position
            x1, y1, x2, y2 = obj.get_position()
            
            # Get color for this class
            color = self.class_colors.get(obj.class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label with ID and class
            label = f"{obj.id}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame counter
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add object count (only those with current detections)
        current_objects = sum(1 for obj in self.objects.values() if obj.has_current_detection)
        cv2.putText(annotated_frame, f"Objects: {current_objects}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame