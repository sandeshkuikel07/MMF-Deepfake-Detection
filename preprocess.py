import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd
import argparse

def extract_faces_from_videos(metadata_path, output_dir, frames_per_video=30, image_size=299):
    """
    Detects and extracts faces from videos listed in the metadata file.
    
    Args:
        metadata_path (str): Path to the CSV metadata file.
        output_dir (str): Directory to save the extracted face images.
        frames_per_video (int): Number of frames to sample from each video.
        image_size (int): The output size of the cropped face images (height and width).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    mtcnn = MTCNN(
        image_size=image_size,
        margin=40,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False,
        device=device
    )

    df = pd.read_csv(metadata_path)
    
    print(f"Starting face extraction for {len(df)} videos...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing videos"):
        video_path = row['path']
        video_id = row['video_id']
        label = row['label']
        
        video_output_dir = os.path.join(output_dir, label, str(video_id))
        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) >= frames_per_video:
            continue
            
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"Warning: Could not read frames from {video_path}")
                continue

            frame_indices = torch.linspace(0, total_frames - 1, frames_per_video).long()
            
            saved_count = 0
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Forward pass returns the aligned largest face when keep_all=False
                face = mtcnn(frame_rgb)

                if face is not None:
                    save_path = os.path.join(video_output_dir, f'frame_{saved_count}.png')
                    face_np = (face.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, face_bgr)
                    saved_count += 1

                if saved_count >= frames_per_video:
                    break
            
            cap.release()

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract faces from FaceForensics++ videos.")
    parser.add_argument('--base_dir', type=str, default='.', help='Base project directory.')
    parser.add_argument('--frames', type=int, default=20, help='Number of frames to extract per video.')
    parser.add_argument('--size', type=int, default=299, help='Size of the extracted face images (e.g., 299 for XceptionNet).')
    args = parser.parse_args()

    metadata_file = os.path.join(args.base_dir, 'ffpp_metadata.csv')
    preprocessed_dir = os.path.join(args.base_dir, 'preprocessed_faces')

    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}.")
        print("Please run 'download_and_prepare_ffpp.py' first.")
    else:
        extract_faces_from_videos(
            metadata_path=metadata_file,
            output_dir=preprocessed_dir,
            frames_per_video=args.frames,
            image_size=args.size
        )
    print("Face extraction complete.")
