import os
import pandas as pd
from tqdm import tqdm

def create_directory_structure(base_dir):
    """Creates the necessary directories for the project."""
    os.makedirs(os.path.join(base_dir, 'FaceForensics++'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'preprocessed_faces'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'extracted_features', 'spatial'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'extracted_features', 'frequency'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'extracted_features', 'semantic'), exist_ok=True)
    print("Directory structure created successfully.")

def generate_metadata(ffpp_dir, output_path, limit_per_category=None):
    """
    Generates a CSV file containing metadata about the videos, adapted for the 
    flat directory structure from the Kaggle dataset.
    
    Args:
        ffpp_dir (str): The root directory of the FaceForensics++ dataset 
                        (e.g., 'E:\\deepfake_detection_project\\FaceForensics++').
        output_path (str): Path to save the output metadata CSV file.
        limit_per_category (int, optional): The number of videos to sample from real and 
                                            each fake category. If None, all videos are used.
    """
    print("Generating metadata file for your specific directory structure...")
    if limit_per_category:
        print(f"--- DEMO MODE: Limiting to {limit_per_category} videos per category. ---")
    data = []
    
    # --- MODIFIED ---
    # Path for original videos is now simpler
    original_dir = os.path.join(ffpp_dir, 'original')

    # Add original videos
    if os.path.exists(original_dir):
        original_videos = [f for f in os.listdir(original_dir) if f.endswith('.mp4')]
        if limit_per_category:
            original_videos = original_videos[:limit_per_category]

        for video_file in original_videos:
            video_id = video_file.split('.')[0]
            data.append({
                'video_id': video_id,
                'label': 'real',
                'path': os.path.join(original_dir, video_file),
                'manipulation': 'Original'
            })
    else:
        print(f"Warning: 'original' directory not found at {original_dir}")

    # --- MODIFIED ---
    # Define the expected manipulation folders at the top level
    manipulation_methods = [
        'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 
        'FaceShifter', 'DeepFakeDetection' # Add all manipulation types you have
    ]
    
    for method in tqdm(manipulation_methods, desc="Processing manipulation methods"):
        # The method directory is now directly inside the ffpp_dir
        method_dir = os.path.join(ffpp_dir, method)
        
        if os.path.exists(method_dir):
            manipulated_videos = [f for f in os.listdir(method_dir) if f.endswith('.mp4')]
            if limit_per_category:
                manipulated_videos = manipulated_videos[:limit_per_category]
            
            for video_file in manipulated_videos:
                video_id = video_file.split('.')[0]
                data.append({
                    'video_id': f"{method}_{video_id}",
                    'label': 'fake',
                    'path': os.path.join(method_dir, video_file),
                    'manipulation': method
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}. Total videos found: {len(df)}")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Define the base directory for your project
    # This should be E:\deepfake_detection_project in your case
    # Let's use relative pathing to be safe.
    BASE_PROJECT_DIR = '.' 
    
    # This should point to E:\deepfake_detection_project\FaceForensics++
    FACEFORENSICS_DATA_DIR = os.path.join(BASE_PROJECT_DIR, 'FaceForensics++')

    print("--- Step 1: Creating Project Directories ---")
    create_directory_structure(BASE_PROJECT_DIR)

    print("\n--- Step 2: Generating Metadata ---")
    
    metadata_output_file = os.path.join(BASE_PROJECT_DIR, 'ffpp_metadata.csv')
    
    # --- For a quick test, use a small number ---
    # Set DEMO_LIMIT = None to run on the entire dataset
    DEMO_LIMIT = 50 
    
    generate_metadata(
        ffpp_dir=FACEFORENSICS_DATA_DIR, 
        output_path=metadata_output_file, 
        limit_per_category=DEMO_LIMIT
    )

    print("\n--- Setup Complete ---")
    print("Next steps:")
    print("1. Run 'preprocess.py' to detect and extract faces from videos.")
    print("2. Run 'feature_extraction.py' to generate embeddings for each domain.")
    print("3. Run 'train_evaluate.py' to train and evaluate the models.")

