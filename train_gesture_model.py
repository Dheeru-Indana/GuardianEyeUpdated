import os
import argparse
import numpy as np
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define gesture classes
GESTURE_CLASSES = {
    'help': 0,
    'fist': 1,
    'open_hand': 2,
    'pointing': 3,
    'other': 4
}

def extract_features(image):
    """Extract features from an image for gesture classification"""
    # Resize image to a standard size
    image = cv2.resize(image, (64, 64))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract HOG features
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(thresh)
    
    # Extract histogram features
    hist_features = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    
    # Combine features
    features = np.concatenate([hog_features.flatten(), hist_features])
    
    return features

# HAGRID dataset loading removed - using only Kaggle datasets

def load_kaggle_datasets(dataset_dirs, max_samples_per_class=1000):
    """Load and process Kaggle datasets"""
    print("Loading Kaggle datasets...")
    
    X = []
    y = []
    
    for dataset_dir in dataset_dirs:
        print(f"Processing dataset: {dataset_dir}")
        
        # Try to find image directories
        image_dirs = []
        for root, dirs, files in os.walk(dataset_dir):
            # Check if this directory contains images
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                image_dirs.append(root)
        
        if not image_dirs:
            print(f"No image directories found in {dataset_dir}")
            continue
        
        for image_dir in image_dirs:
            dir_name = os.path.basename(image_dir).lower()
            
            # Try to map directory name to a gesture class
            if 'help' in dir_name or 'emergency' in dir_name:
                class_label = GESTURE_CLASSES['help']
            elif 'fist' in dir_name or 'closed' in dir_name:
                class_label = GESTURE_CLASSES['fist']
            elif 'open' in dir_name or 'palm' in dir_name:
                class_label = GESTURE_CLASSES['open_hand']
            elif 'point' in dir_name or 'index' in dir_name:
                class_label = GESTURE_CLASSES['pointing']
            else:
                class_label = GESTURE_CLASSES['other']
            
            # Get list of image files
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples per class
            if len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
            
            print(f"Processing {len(image_files)} images from {image_dir} (class: {list(GESTURE_CLASSES.keys())[class_label]})")
            
            # Process each image
            for img_file in tqdm(image_files, desc=os.path.basename(image_dir)):
                img_path = os.path.join(image_dir, img_file)
                try:
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Extract features
                    features = extract_features(image)
                    
                    X.append(features)
                    y.append(class_label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train a gesture recognition model"""
    print("\nTraining gesture recognition model...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.4f}")
    
    # Print classification report
    # Filter out classes with zero samples to avoid ValueError
    class_counts = {}
    for class_name, class_id in GESTURE_CLASSES.items():
        count = np.sum(y == class_id)
        class_counts[class_name] = count
    
    # Only include classes that have samples
    active_classes = [class_name for class_name, count in class_counts.items() if count > 0]
    
    print("\nClassification Report:")
    try:
        print(classification_report(y_test, y_pred, target_names=active_classes))
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("This may be due to some classes having no samples in the test set.")
        print("Continuing with training process...")

    
    # Plot feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/feature_importance.png')
    
    return model, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument('--dataset-dir', default='datasets', help='Directory containing datasets')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples per class')
    parser.add_argument('--output-model', default='models/gesture_model.pkl', help='Output model file')
    
    args = parser.parse_args()
    
    print("=== Guardian Eye Gesture Recognition Model Training ===")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    # Find Kaggle datasets
    kaggle_dirs = []
    for dir_name in os.listdir(args.dataset_dir):
        kaggle_dirs.append(os.path.join(args.dataset_dir, dir_name))
    
    # Load datasets
    X, y = [], []
    if kaggle_dirs:
        X, y = load_kaggle_datasets(kaggle_dirs, args.max_samples)
    
    if len(X) == 0:
        print("Error: No valid datasets found")
        return
    
    print(f"\nTotal dataset size: {len(X)} samples")
    print("Class distribution:")
    for class_name, class_id in GESTURE_CLASSES.items():
        count = np.sum(y == class_id)
        print(f"  - {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Train model
    model, accuracy = train_model(X, y)
    
    # Save model
    print(f"\nSaving model to {args.output_model}")
    with open(args.output_model, 'wb') as f:
        pickle.dump(model, f)
    
    # Save class mapping
    class_mapping_file = os.path.join(os.path.dirname(args.output_model), 'gesture_classes.pkl')
    with open(class_mapping_file, 'wb') as f:
        pickle.dump(GESTURE_CLASSES, f)
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.output_model}")
    print(f"Class mapping saved to: {class_mapping_file}")
    print(f"Feature importance plot saved to: models/feature_importance.png")

if __name__ == "__main__":
    main()