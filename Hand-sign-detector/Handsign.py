import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

def load_dataset(dataset_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
                print(f"Loaded image: {file_path}, Label: {label}")
    return images, labels

dataset_path = r"B:\Training_images"  # Specify the path to your dataset folder
images, labels = load_dataset(dataset_path)

print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Split the dataset into training and testing sets
if len(images) > 0:  # Check if there are images in the dataset
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Feature extraction
    train_features = np.array([image.flatten() for image in images_train])
    test_features = np.array([image.flatten() for image in images_test])

    # Initialize the VGG16 model
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))  # Update input_shape to (64, 64, 3)

    # Extract features using VGG16
    train_features = preprocess_input(train_features)
    train_features = train_features.reshape(train_features.shape[0], 64, 64, 3)  # Reshape to match the input shape of VGG16
    train_features = vgg_model.predict(train_features)
    train_features = train_features.reshape(train_features.shape[0], -1)

    test_features = preprocess_input(test_features)
    test_features = test_features.reshape(test_features.shape[0], 64, 64, 3)  # Reshape to match the input shape of VGG16
    test_features = vgg_model.predict(test_features)
    test_features = test_features.reshape(test_features.shape[0], -1)

    # SVM model training
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(train_features, labels_train)

    # Model evaluation
    predictions = svm_model.predict(test_features)
    print(classification_report(labels_test, predictions))

    # Save the retrained model
    model_path = r"B:\Model\model.pkl"
    joblib.dump(svm_model, model_path)
else:
    print("No images found in the dataset.")
