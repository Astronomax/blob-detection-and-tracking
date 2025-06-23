import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blob import ConvMethod, detect_blobs

def load_and_prepare_images(image_path, target_size=(256, 192)):
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    h, w = gray_img.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    color_resized = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray_resized = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    processed_img = np.zeros((target_h, target_w), dtype=np.float32)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    processed_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 1.0 - (gray_resized.astype(np.float32) / 255.0)

    color_result = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    color_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = color_resized
    
    return color_result, processed_img

def test_blob_detection():
    image_path = "birds.jpg"
    color_img, processed_img = load_and_prepare_images(image_path)

    blobs = detect_blobs(
        method=ConvMethod.SIMPLE,
        image=processed_img,
        min_sigma=3.0,
        max_sigma=8.0,
        num_sigma=10,
        threshold_abs=0.2,
        overlap=0.05,
        use_prune_blobs=True
    )

    print(f"Detected {len(blobs)} blobs")

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(color_img)

    for i, blob in enumerate(blobs):
        circle = patches.Circle(
            (blob.x, blob.y),
            blob.r,
            color='red',
            fill=False,
            linewidth=2,
            label=f'Blob {i}' if i == 0 else ""
        )
        ax.add_patch(circle)
        print(f"Blob {i}: x={blob.x:.1f}, y={blob.y:.1f}, r={blob.r:.1f}")

    ax.set_title("Detected blobs (red circles)")
    ax.set_xlim(0, color_img.shape[1])
    ax.set_ylim(color_img.shape[0], 0)
    ax.legend(loc='upper right')
    
    plt.tight_layout()

    output_path = "birds_processed.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Result saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    test_blob_detection()
