import os
import subprocess
import zipfile
import requests
import shutil
import json
import argparse
from tqdm import tqdm

# Create dataset directory if it doesn't exist
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(DATASET_DIR, exist_ok=True)

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def download_kaggle_dataset(dataset_name, kaggle_username=None, kaggle_key=None):
    """Download dataset from Kaggle"""
    print(f"\nDownloading Kaggle dataset: {dataset_name}")
    
    # Check if Kaggle credentials are provided or in environment variables
    if not kaggle_username or not kaggle_key:
        try:
            # Try to get from environment variables
            kaggle_username = os.environ.get('KAGGLE_USERNAME')
            kaggle_key = os.environ.get('KAGGLE_KEY')
            
            # If still not available, try to get from kaggle.json
            if not kaggle_username or not kaggle_key:
                kaggle_config_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
                kaggle_config_path = os.path.join(kaggle_config_dir, 'kaggle.json')
                
                if os.path.exists(kaggle_config_path):
                    with open(kaggle_config_path, 'r') as f:
                        config = json.load(f)
                        kaggle_username = config.get('username')
                        kaggle_key = config.get('key')
        except Exception as e:
            print(f"Error getting Kaggle credentials: {e}")
    
    if not kaggle_username or not kaggle_key:
        print("Kaggle credentials not found. Please provide them as arguments or set them up:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to Account -> Create API Token to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ or provide credentials as arguments")
        return False
    
    # Set up Kaggle credentials
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    # Create dataset directory
    dataset_dir = os.path.join(DATASET_DIR, dataset_name.split('/')[-1])
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        # Install kaggle package if not already installed
        try:
            import kaggle
        except ImportError:
            print("Installing kaggle package...")
            subprocess.check_call(["pip", "install", "kaggle"])
            import kaggle
        
        # Download dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name,
            path=dataset_dir,
            unzip=True
        )
        
        print(f"Successfully downloaded and extracted Kaggle dataset to {dataset_dir}")
        return dataset_dir
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        return False

# HAGRID dataset download function removed as we're only using Kaggle datasets

def main():
    parser = argparse.ArgumentParser(description='Download Kaggle gesture recognition datasets')
    parser.add_argument('--kaggle-username', help='Kaggle username')
    parser.add_argument('--kaggle-key', help='Kaggle API key')
    parser.add_argument('--kaggle-datasets', nargs='+', default=[
        'kshitijdhyani/hand-gesture-recognition-dataset',
        'gti-upm/leapgestrecog'
    ], help='List of Kaggle datasets to download')
    
    args = parser.parse_args()
    
    print("=== Guardian Eye Kaggle Dataset Downloader ===")
    print(f"Datasets will be saved to: {DATASET_DIR}")
    
    # Download Kaggle datasets only
    kaggle_paths = []
    for dataset in args.kaggle_datasets:
        path = download_kaggle_dataset(dataset, args.kaggle_username, args.kaggle_key)
        if path:
            kaggle_paths.append(path)
    
    # Print summary
    print("\n=== Download Summary ===")
    if kaggle_paths:
        print(f"✓ Kaggle datasets: {len(kaggle_paths)}/{len(args.kaggle_datasets)} downloaded successfully")
        for path in kaggle_paths:
            print(f"  - {path}")
    else:
        print("✗ Kaggle datasets: Failed to download any datasets")
    
    print("\nTo use these datasets for training, run the training script:")
    print("python train_gesture_model.py --dataset-dir datasets")

if __name__ == "__main__":
    main()