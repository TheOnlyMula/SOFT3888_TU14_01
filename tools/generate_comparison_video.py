from pathlib import Path
import numpy as np
import cv2
import argparse
import torch
import sys
import hdf5plugin  # Required to prevent Blosc decompression error

from ultralytics import YOLO
from dsec_det.dataset import DSECDet
from dsec_det.visualize import render_object_detections_on_image 

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import build_model
from dataset_loader import load_causal_events

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

def generate_event_histogram(events, height=480, width=640):
    hist = np.zeros((2, height, width), dtype=np.float32)
    if events is not None and len(events.get("x", [])) > 0:
        x = events["x"].astype(int)
        y = events["y"].astype(int)
        p = events["p"].astype(int)
        np.add.at(hist[0], (y[p == 0], x[p == 0]), 1)
        np.add.at(hist[1], (y[p == 1], x[p == 1]), 1)
    hist_tensor = torch.from_numpy(hist)
    hist_tensor = torch.log1p(hist_tensor)
    return hist_tensor

def load_yolo_model(weights_path, ch, device):
    model = YOLO("yolo26n.yaml", task="detect")
    model.model = build_model(ch)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.model.load_state_dict(state_dict)
    
    return model, ch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AI vs GT Video for DSEC Det")
    parser.add_argument("--dsec_merged", type=Path, required=True, help="Path to DSEC dataset root")
    parser.add_argument("--student-weights", type=Path, default=None, help="Path to student trained weights (.pt) [Optional]")
    parser.add_argument("--teacher-weights", type=Path, default=None, help="Path to teacher trained weights (.pt) [Optional]")
    parser.add_argument("--sequence", type=int, required=True, help="Sequence index integer")
    parser.add_argument("--split", type=str, default="test", choices=['train', 'test'])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--debug", action="store_true", help="Apply gamma correction to brighten the visualization")
    
    args = parser.parse_args()
    if args.student_weights is None and args.teacher_weights is None:
        parser.error("At least one of --student-weights or --teacher-weights must be provided.")
        
    script_dir = Path(__file__).resolve().parent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = None
    if args.student_weights is not None:
        print(f"Loading Student model from {args.student_weights}...")
        student_model, _ = load_yolo_model(args.student_weights, 3, device)
    
    teacher_model = None
    if args.teacher_weights is not None:
        print(f"Loading Teacher model from {args.teacher_weights}...")
        teacher_model, _ = load_yolo_model(args.teacher_weights, 5, device)
        
    print("Models successfully loaded!")
    
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
    
    output_video_path = script_dir.parent / "results" / f"{seq_name}_comparison.mp4"
    
    print(f"Sequence {args.sequence} mapped to directory '{seq_name}'.")
    print(f"Extracting {num_frames_in_seq} frames (Global index {global_start_idx} to {global_start_idx + num_frames_in_seq - 1}).")
    if args.debug:
        print("Debug mode ON: Video frames will be artificially brightened.")
    
    video_writer = None
    
    for i in range(num_frames_in_seq):
        current_global_idx = global_start_idx + i
        
        img_raw = dataset.get_image(current_global_idx).copy()
        tracks = dataset.get_tracks(current_global_idx)
        events = load_causal_events(dataset, current_global_idx) if teacher_model is not None else None
        
        if len(img_raw.shape) == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            
        if args.debug:
            display_canvas = (255 * (img_raw.astype("float32") / 255) ** (1/2.2)).astype("uint8")
        else:
            display_canvas = img_raw.copy()
            
        img_gt = render_object_detections_on_image(display_canvas.copy(), tracks)
        if len(img_gt.shape) == 2:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)

        # Student Prediction (RGB)
        img_student = None
        if student_model is not None:
            results_student = student_model.predict(img_raw, verbose=False)
            results_student[0].orig_img = display_canvas.copy()
            img_student = results_student[0].plot() 
            img_student = add_title_to_image(img_student, "Student (RGB)")
        
        # Teacher Prediction (Fusion)
        img_teacher = None
        if teacher_model is not None:
            # The teacher is trained on RGB; cv2 frames are BGR.
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            event_tensor = generate_event_histogram(events, height=img_raw.shape[0], width=img_raw.shape[1])
            fused_tensor = torch.cat((img_tensor, event_tensor), dim=0).unsqueeze(0).to(device)
            
            # Ultralytics predictor divides tensors by 255 if max > 1.0. Pre multiply to ensure correct magnitudes.
            fused_tensor *= 255.0
            
            teacher_model.model.warmup = lambda *args, **kwargs: None
            import ultralytics.nn.autobackend
            ultralytics.nn.autobackend.AutoBackend.warmup = lambda *args, **kwargs: None
            
            # Temporarily suppress the Ultralytics logger to hide the harmless max value warning
            from ultralytics.utils import LOGGER
            import logging
            old_level = LOGGER.level
            LOGGER.setLevel(logging.ERROR)
            
            results_teacher = teacher_model.predict(fused_tensor, verbose=False)
            
            LOGGER.setLevel(old_level)
            
            results_teacher[0].orig_img = display_canvas.copy()
            img_teacher = results_teacher[0].plot()
            img_teacher = add_title_to_image(img_teacher, "Teacher (Fusion)")
            
        img_gt = add_title_to_image(img_gt, "Ground Truth")
            
        # Stack images horizontally
        h_panel = img_gt.shape[0]
        divider = np.full((h_panel, 10, 3), fill_value=255, dtype="uint8")
        
        panels = []
        if img_student is not None:
            panels.append(img_student)
        if img_teacher is not None:
            panels.append(img_teacher)
        panels.append(img_gt)
        
        combined = panels[0]
        for p in panels[1:]:
            combined = np.hstack((combined, divider, p))
            
        panel = combined
        
        if video_writer is None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = create_video_writer(output_video_path, args.fps, panel.shape[1], panel.shape[0])
            print(f"Writing video to {output_video_path}... (Press 'q' in window to stop early)")
            
        video_writer.write(panel)
        
        cv2.imshow(f"Sequence {args.sequence} ({seq_name}) Comparison", panel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Processing complete!")