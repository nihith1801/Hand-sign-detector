import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import joblib

# Load the trained model
model_path = r"B:\Model\model.pkl"  # Specify the path to the saved model
svm_model = joblib.load(model_path)

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

# Initialize the camera capture
camera = cv2.VideoCapture(0)  # Use the appropriate camera index, 0 for the default camera

batch_size = 10  # Adjust the batch size as needed
frame_buffer = []

while True:
    # Capture frame from the camera
    ret, frame = camera.read()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (64, 64))
    preprocessed_frame = preprocess_input(resized_frame)

    # Accumulate frames in the buffer
    frame_buffer.append(preprocessed_frame)

    # Check if the batch size is reached
    if len(frame_buffer) >= batch_size:
        # Convert the frame buffer to a numpy array
        input_frames = np.array(frame_buffer)

        # Extract features using VGG16
        features = vgg_model.predict(input_frames)
        flattened_features = features.reshape(features.shape[0], -1)

        # Perform the sign classification on the batch
        predicted_labels = svm_model.predict(flattened_features)

        # Process the predicted labels for the batch (e.g., perform majority voting)
        # ...

        # Display the frames with the predicted labels
        for i in range(len(frame_buffer)):
            frame = frame_buffer[i]
            predicted_label = predicted_labels[i]

            frame = np.array(frame)  # Convert frame to a numpy array
            frame = cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Sign Classification", frame)

        # Clear the frame buffer
        frame_buffer.clear()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
