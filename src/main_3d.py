import torch
import numpy as np
from models.custom_3d_fasterrcnn import create_3d_faster_rcnn, VolumetricDataLoader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create the model with 1 input channel (typical for medical scans)
    model = create_3d_faster_rcnn(
        num_classes=2,  # Background + lung nodule
        in_channels=1
    )
    model.to(device)
    model.eval()
    
    # Create a dummy 3D input (batch_size=1, channels=1, depth=64, height=512, width=512)
    # This simulates a volumetric scan like CT
    dummy_volume = torch.rand(1, 1, 64, 512, 512).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model([dummy_volume])
    
    # Print prediction structure
    for pred in predictions:
        print("Boxes:", pred['boxes'].shape)
        print("Labels:", pred['labels'].shape)
        print("Scores:", pred['scores'].shape)
    
    print("\nExample using a synthetic nodule:")
    # Create a synthetic volume with a simulated nodule
    synthetic_volume = np.zeros((1, 64, 512, 512), dtype=np.float32)
    
    # Add a synthetic nodule (bright spot)
    z_center, y_center, x_center = 32, 256, 256
    z_size, y_size, x_size = 8, 20, 20
    
    z_start, z_end = z_center - z_size//2, z_center + z_size//2
    y_start, y_end = y_center - y_size//2, y_center + y_size//2
    x_start, x_end = x_center - x_size//2, x_center + x_size//2
    
    synthetic_volume[0, z_start:z_end, y_start:y_end, x_start:x_end] = 1.0
    
    # Ground truth bbox (x1, y1, z1, x2, y2, z2)
    gt_bbox = [x_start, y_start, z_start, x_end, y_end, z_end]
    print(f"Ground truth 3D bbox: {gt_bbox}")
    
    # Convert to tensor
    synthetic_volume_tensor = torch.tensor(synthetic_volume, dtype=torch.float32)
    
    # Run inference
    with torch.no_grad():
        predictions = model([synthetic_volume_tensor])
        
    # Process and display results
    for pred in predictions:
        if len(pred['boxes']) > 0:
            highest_score_idx = pred['scores'].argmax()
            box = pred['boxes'][highest_score_idx].tolist()
            score = pred['scores'][highest_score_idx].item()
            label = pred['labels'][highest_score_idx].item()
            print(f"Predicted 2D bbox: {box}, Score: {score:.4f}, Label: {label}")


def demo_inference_on_real_data(volume_path, model=None):
    """
    Demo function to show how to use the model on real data
    
    Args:
        volume_path: Path to a volumetric scan (e.g., .npy file containing a CT scan)
        model: Optional pre-loaded model. If None, a new model is created
    """
    import numpy as np
    
    # Load volume (would be replaced with actual loading code)
    try:
        # This is just a placeholder. In practice, you would use a proper loader
        # for your specific data format (DICOM, NIfTI, etc.)
        volume = np.load(volume_path)
    except Exception as e:
        print(f"Error loading volume: {e}")
        print("Using a synthetic volume instead")
        volume = np.random.rand(64, 512, 512).astype(np.float32)
    
    # Preprocess volume
    preprocessed = VolumetricDataLoader.preprocess_volume(volume)
    
    # Create or use existing model
    if model is None:
        model = create_3d_faster_rcnn(num_classes=2)
        model.eval()
    
    # Run inference
    with torch.no_grad():
        predictions = model([preprocessed])
    
    # Process results
    for pred in predictions:
        if len(pred['boxes']) > 0:
            # Get the detection with highest confidence
            highest_score_idx = pred['scores'].argmax()
            box = pred['boxes'][highest_score_idx].tolist()
            score = pred['scores'][highest_score_idx].item()
            
            print(f"Detected nodule at {box} with confidence {score:.4f}")
            
            # In a real application, you would visualize this box on the volume
            # For example:
            # visualize_detection(volume, box)


if __name__ == "__main__":
    main()
    
    # Uncomment to run demo on a real data file
    # demo_inference_on_real_data("/path/to/your/volume.npy")
