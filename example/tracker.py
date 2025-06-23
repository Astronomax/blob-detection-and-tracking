import numpy as np
import cv2
import imageio
from tqdm import tqdm
from blob import detect_blobs, ConvMethod, BlobTracker, ObjectStatus

def process_gif(input_path, output_path, processing_size=250, output_fps=None):

    tracker = BlobTracker(use_prediction=True)

    colors = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120)
    ]

    with imageio.get_reader(input_path) as reader:
        input_fps = reader.get_meta_data().get('fps', 10)
        frame_delay = 1.0 / input_fps if output_fps is None else 1.0 / output_fps

        with imageio.get_writer(
            output_path, 
            mode='I', 
            duration=frame_delay * 1000
        ) as writer:
            for frame in tqdm(reader, desc="Processing GIF"):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
                scale = processing_size / max(gray.shape)
                small_img = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                img_normalized = 1.0 - (small_img.astype(np.float32) / 255.0)
                
                blobs = detect_blobs(
                    image=img_normalized,
                    min_sigma=1.0,
                    max_sigma=6.0,
                    num_sigma=5,
                    threshold_abs=0.25,
                    overlap=0.1,
                    method=ConvMethod.FFT_SLOW
                )
                
                tracked = tracker.track(blobs, 30.0, 1.5)

                for obj in tracked:
                    if obj['status'] == ObjectStatus.DIED:
                        continue
                    color = colors[obj['id'] % len(colors)]
                    center = (int(obj['x']/scale), int(obj['y']/scale))
                    radius = int(obj['r']/scale)
                    cv2.circle(frame, center, radius, color, 2)
                    cv2.putText(frame, str(obj['id']), 
                               (center[0]-5, center[1]-radius-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                writer.append_data(frame)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python tracker.py input.gif output.gif [processing_size] [output_fps]")
        sys.exit(1)
    
    processing_size = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    output_fps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    process_gif(sys.argv[1], sys.argv[2], processing_size, output_fps)
