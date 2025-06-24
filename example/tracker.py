import numpy as np
import cv2
from tqdm import tqdm
from blob import detect_blobs, ConvMethod, BlobTracker, ObjectStatus

def process_video(input_path, output_path, processing_size=250, output_fps=None):
    tracker = BlobTracker(use_prediction=True, use_ttl=True, ttl=100)

    status_colors = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (128, 128, 128)
    }

    next_id = 1
    id_map = {}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = output_fps if output_fps is not None else input_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=frame_count, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb
            scale = processing_size / max(gray.shape)
            small_img = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            img_normalized = 1.0 - (small_img.astype(np.float32) / 255.0)

            blobs = detect_blobs(
                image=img_normalized,
                min_sigma=1.0,
                max_sigma=6.0,
                num_sigma=5,
                threshold_abs=0.35,
                overlap=0.1,
                method=ConvMethod.FFT_SLOW
            )

            tracked = tracker.track(blobs, 1000.0, 1.5)

            for obj in tracked:
                if obj['id'] not in id_map:
                    id_map[obj['id']] = next_id
                    next_id += 1

            for obj in tracked:
                status = obj['status']
                if status == 3:  # DIED
                    continue

                display_id = id_map[obj['id']]

                color = status_colors[status]

                center = (int(obj['x']/scale), int(obj['y']/scale))
                radius = int(obj['r']/scale)

                cv2.circle(frame, center, radius, color, 2)

                status_text = ["BORN", "ALIVE", "GHOST", "DIED"][status]
                cv2.putText(frame, f"{display_id}:{status_text}",
                           (center[0]-15, center[1]-radius-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python tracker.py input.mp4 output.mp4 [processing_size] [output_fps]")
        sys.exit(1)
    
    processing_size = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    output_fps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    process_video(sys.argv[1], sys.argv[2], processing_size, output_fps)
