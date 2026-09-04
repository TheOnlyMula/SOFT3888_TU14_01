from pathlib import Path
import numpy as np
import cv2
import argparse
import torch

from ultralytics import YOLO
from dsec_det.dataset import DSECDet
from dsec_det.visualize import render_object_detections_on_image 

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

def add_title_to_image(img, title):
    h, w = img.shape[:2]
    header_h = 45
    header = np.full((header_h, w, 3), 255, dtype=np.uint8)
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (header_h + text_size[1]) // 2
    cv2.putText(header, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AI vs GT Video for DSEC Det")
    parser.add_argument("--dsec_merged", type=Path, required=True, help="Path to DSEC dataset root")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--type", type=str, required=True, choices=["student", "teacher"], help="Model type: student or teacher")
    parser.add_argument("--sequence", type=int, required=True, help="Sequence index integer")
    parser.add_argument("--split", type=str, default="test", choices=['train', 'test'])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--debug", action="store_true", help="Apply gamma correction to brighten the visualization")
    
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    

    model_yaml = script_dir.parent / "configs" / f"yolo26n_{args.type}.yaml"
    if not model_yaml.is_file():
        model_yaml = script_dir / "configs" / f"yolo26n_{args.type}.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading YOLO26 architecture from {model_yaml} and weights from {args.weights}...")
    model = YOLO(str(model_yaml), task="detect")
    
    dsec_names = {
        0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus', 
        4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
    }
    model.model.names = dsec_names
    
    custom_checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if "model_state_dict" in custom_checkpoint:
        model.model.load_state_dict(custom_checkpoint["model_state_dict"])
    else:
        model.model.load_state_dict(custom_checkpoint)
        
    print("Model successfully loaded!")
    
    print("Initializing DSEC dataset...")
    dataset = DSECDet(args.dsec_merged, split=args.split, sync="back", debug=False)
    
    num_sequences = len(dataset.subsequence_directories)
    if args.sequence >= num_sequences:
        raise ValueError(f"Sequence {args.sequence} is out of bounds. There are only {num_sequences} sequences.")
        
    global_start_idx = 0
    for j in range(args.sequence):
        f_name = dataset.subsequence_directories[j].name
        global_start_idx += len(dataset.img_idx_track_idxs[f_name]) - 1
        
    seq_name = dataset.subsequence_directories[args.sequence].name
    num_frames_in_seq = len(dataset.img_idx_track_idxs[seq_name]) - 1
    
    output_video_path = script_dir.parent / "results" / f"{seq_name}_{args.type}.mp4"
    
    print(f"Sequence {args.sequence} mapped to directory '{seq_name}'.")
    print(f"Extracting {num_frames_in_seq} frames (Global index {global_start_idx} to {global_start_idx + num_frames_in_seq - 1}).")
    if args.debug:
        print("Debug mode ON: Video frames will be artificially brightened.")
    
    video_writer = None
    
    for i in range(num_frames_in_seq):
        current_global_idx = global_start_idx + i
        
        img_raw = dataset.get_image(current_global_idx).copy()
        tracks = dataset.get_tracks(current_global_idx)
        
        if len(img_raw.shape) == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            
        if args.debug:
            display_canvas = (255 * (img_raw.astype("float32") / 255) ** (1/2.2)).astype("uint8")
        else:
            display_canvas = img_raw.copy()
            
        img_gt = render_object_detections_on_image(display_canvas.copy(), tracks)
        if len(img_gt.shape) == 2:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)

        results = model.predict(img_raw, verbose=False)
        
        results[0].orig_img = display_canvas.copy()
        img_ai = results[0].plot() 
        
        h_ai, w_ai = img_ai.shape[:2]
        h_gt, w_gt = img_gt.shape[:2]
        if h_ai != h_gt:
            new_w_gt = int(w_gt * (h_ai / h_gt))
            img_gt = cv2.resize(img_gt, (new_w_gt, h_ai))
        
        img_ai = add_title_to_image(img_ai, f"YOLO26 {args.type.capitalize()}")
        img_gt = add_title_to_image(img_gt, "Ground Truth")
            
        divider = np.full((img_ai.shape[0], 10, 3), fill_value=255, dtype="uint8")
        panel = np.hstack((img_ai, divider, img_gt))
        
        if video_writer is None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = create_video_writer(output_video_path, args.fps, panel.shape[1], panel.shape[0])
            print(f"Writing video to {output_video_path}... (Press 'q' in window to stop early)")
            
        video_writer.write(panel)
        
        cv2.imshow(f"Sequence {args.sequence} ({seq_name}) Eval", panel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Processing complete!")