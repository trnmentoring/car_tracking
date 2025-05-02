#!/usr/bin/env python3
"""
Main entry point for the object tracking application
"""

import os
import cv2
import glob
import argparse
from utils import load_config, set_seed
from detector_tracker import DetectorTracker


def process_video(video_path, detector_tracker, output_path=None, output_dir="viz"):
    """
    Process a single video file with object tracking and visualization
    
    Args:
        video_path: Path to input video file
        detector_tracker: DetectorTracker instance to use
        output_path: Path to output video file (default: auto-generated based on input path)
        output_dir: Directory to save output videos
    """
    # Reset tracker state for the new video
    detector_tracker.reset()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path if not provided
    if output_path is None:
        # Get base filename without extension
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_basename}_tracked.mp4")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Total frames to process: {total_frames}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        
        # Process frame with detection and tracking
        annotated_frame = detector_tracker.process_frame(frame)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Print progress
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            objects = detector_tracker.get_vehicles()
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Tracking {len(objects)} objects")
    
    # Clean up
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_path}")


def process_directory(config):
    """
    Process all videos in a directory based on config
    
    Args:
        config: Configuration dictionary loaded from YAML
    """
    # Get general settings
    general = config.get("general", {})
    
    # Set random seed for reproducibility
    if "seed" in general:
        set_seed(general["seed"])
        print(f"Set random seed to {general['seed']} for reproducibility")
    
    # Get detector-tracker configuration
    detector_tracker_params = config.get("detector_tracker", {})
    
    # Get video and output directories
    video_dir = general.get("video_dir", "videos")
    output_dir = general.get("output_dir", "viz")
    
    # Initialize detector-tracker once for all videos
    print("Initializing detector-tracker...")
    detector_tracker = DetectorTracker(detector_tracker_params)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video with the same detector-tracker instance
    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        process_video(
            video_path,
            detector_tracker=detector_tracker,
            output_dir=output_dir
        )


def main():
    """Main entry point for object tracking application"""
    parser = argparse.ArgumentParser(description="Object tracking with YOLOv8")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", 
                        help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process directory of videos based on config
    process_directory(config)


if __name__ == "__main__":
    main()