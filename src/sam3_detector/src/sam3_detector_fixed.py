# segment the rgb images using sam3
from utils.paths import REAL_PLANTS, pkg_path, real_plant_dir
from ultralytics.models.sam import SAM3SemanticPredictor
import os
import cv2
import numpy as np

# Debug flag - set to False to disable visualization output
SAVE_DEBUG_VISUALIZATION = True

# Initialize predictor with configuration
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=os.environ.get(
        'SAM3_CHECKPOINT', os.path.join(pkg_path('sam3_detector'), 'materials', 'sam3.pt')
    ),
    half=True,  # Use FP16 for faster inference
    save=False,  # Enable saving like the original working version
)
predictor = SAM3SemanticPredictor(overrides=overrides)


# Process plants 31
for plant_num in REAL_PLANTS:
    plant_name = f"plant{plant_num}"
    dir_path = os.path.join(real_plant_dir(plant_num), 'raw', 'rgb')
    store_path = os.path.join(real_plant_dir(plant_num), 'raw', 'masks')

    # Skip if input directory doesn't exist
    if not os.path.exists(dir_path):
        print(f"\nSkipping {plant_name}: directory not found")
        continue

    # Create output directory if it doesn't exist
    os.makedirs(store_path, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing {plant_name}")
    print(f"Input: {dir_path}")
    print(f"Output: {store_path}")
    print(f"{'=' * 60}")

    # Process all JPEG images in the directory
    for file_name in sorted(os.listdir(dir_path)):
        if file_name.endswith('.png') or file_name.endswith('.jpeg'):
            image_path = os.path.join(dir_path, file_name)
            print(f"\nProcessing: {file_name}")

            try:
                # Set image and run prediction
                predictor.set_image(image_path)
                results = predictor(text=["tomato fruits"])

                # Get image dimensions
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]

                # Prepare output mask file
                base_name = file_name[:-5] if file_name.endswith('.jpeg') else file_name[:-4]
                mask_path = os.path.join(store_path, f"{base_name}.png")

                # Initialize combined binary mask
                combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)

                detections = []

                # Process each detected mask
                if results and len(results) > 0:
                    print(f"  Found {len(results)} result(s)")

                    for i, result in enumerate(results):
                        # Extract all masks from this result
                        if hasattr(result, 'masks') and result.masks is not None:
                            # Iterate through ALL masks in this result
                            num_masks = len(result.masks.data)
                            print(f"  Result {i} has {num_masks} mask(s)")

                            for mask_idx in range(num_masks):
                                mask = result.masks.data[mask_idx].cpu().numpy() > 0.5

                                # Calculate mask area
                                area = np.sum(mask)

                                # Get confidence
                                confidence = (
                                    result.conf[mask_idx].item()
                                    if hasattr(result, 'conf') and len(result.conf) > mask_idx
                                    else 0.0
                                )

                                # Store detection
                                detections.append(
                                    {
                                        'area': area,
                                        'mask': mask.astype(np.uint8),
                                        'confidence': confidence,
                                    }
                                )
                        elif isinstance(result, dict) and 'mask' in result:
                            mask = result['mask']
                            area = np.sum(mask)
                            confidence = result.get('conf', 0.0)

                            # Store detection
                            detections.append(
                                {
                                    'area': area,
                                    'mask': mask.astype(np.uint8),
                                    'confidence': confidence,
                                }
                            )
                        else:
                            print(f"  Warning: Could not extract mask from result {i}")
                            continue

                # Combine all masks into a single binary mask
                for i, det in enumerate(detections):
                    # Add this mask to the combined mask (any pixel in any mask becomes 1)
                    combined_mask = np.maximum(combined_mask, det['mask'])
                    print(
                        f"  Detection {i + 1}/{len(detections)}: area={det['area']:.0f} pixels, conf={det['confidence']:.4f}"
                    )

                # Save the binary mask as PNG (0 = background, 1 = tomato)
                cv2.imwrite(mask_path, combined_mask * 255)  # Scale to 0-255 for PNG

                if detections:
                    print(
                        f"  Saved binary mask with {len(detections)} detection(s) to: {os.path.basename(mask_path)}"
                    )
                else:
                    print(f"  No detections - empty mask created")

                # Debug visualization (optional)
                if SAVE_DEBUG_VISUALIZATION:
                    debug_path = os.path.join(store_path, f"{base_name}_debug.png")
                    # Convert to grayscale to save space
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Create visualization with mask overlay
                    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                    # Create colored overlay (red with 50% transparency)
                    overlay = debug_img.copy()
                    overlay[combined_mask > 0] = [0, 0, 255]  # Red color for detections
                    # Blend original and overlay
                    debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
                    # Draw contours for better visibility
                    contours, _ = cv2.findContours(
                        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)  # Green contours
                    cv2.imwrite(debug_path, debug_img)
                    print(f"  Saved debug visualization to: {os.path.basename(debug_path)}")

            except Exception as e:
                print(f"  Error processing {file_name}: {str(e)}")
                continue

print("\n" + "=" * 60)
print("All plants processed!")
print("=" * 60)
