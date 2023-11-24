import cv2
import numpy as np
from tensorflow import keras
import random
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = keras.models.load_model('emotion_model.hdf5')

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
funny_lines = {
    "Angry": "Its fine to lose sometimes....",
    "Disgust": "Any problem with the event",
    "Fear": "Don't worry you can do this",
    "Happy": "Excited for the event..?!",
    "Sad": "Don't worry better luck next time...",
    "Surprise": "Excited for the event??",
    "Neutral": "Tired trying to win??"
}

cap = cv2.VideoCapture(0)
save_directory = r'\Image'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save_frame = False

def save_frame_to_directory(event, x, y, flags, param):
    global save_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        save_frame = True

cv2.namedWindow('Emotion Analysis')
cv2.setMouseCallback('Emotion Analysis', save_frame_to_directory)

# Define a function to generate and display the collage
def generate_and_display_collage(image_directory, target_directory, header_image_path, images, collage_counter):
    header_image = cv2.imread(header_image_path)

    num_images = len(images)
    collage_rows = int(np.ceil(np.sqrt(num_images)))
    collage_cols = int(np.ceil(num_images / collage_rows))

    cell_height = images[0].shape[0]
    cell_width = images[0].shape[1]

    target_width = cell_width * collage_cols
    target_height = header_image.shape[0] * target_width // header_image.shape[1]

    header_image = cv2.resize(header_image, (target_width, target_height))

    collage = np.zeros((cell_height * collage_rows + target_height, cell_width * collage_cols, 3), dtype=np.uint8)
    collage[:target_height, :target_width] = header_image

    for i in range(num_images):
        row = i // collage_cols
        col = i % collage_cols
        collage[target_height + row * cell_height:target_height + (row + 1) * cell_height, col * cell_width:(col + 1) * cell_width] = images[i]

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    collage_filename = os.path.join(target_directory, f'collage_{collage_counter}.jpg')
    cv2.imwrite(collage_filename, collage)

# Keep track of the header image and images
header_image_path = r'it.jpg'  # Replace with your header image path
header_image = cv2.imread(header_image_path)
images = []

# Create a variable to keep track of the number of saved collages
collage_counter = 1

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.reshape(face_roi, (1, 64, 64, 1))
        face_roi = face_roi / 255.0
        emotion_predictions = model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(emotion_predictions)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if emotion_label in funny_lines:
            funny_line = funny_lines[emotion_label]
            cv2.putText(frame, funny_line, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.flip(frame, 1)
    cv2.imshow('Emotion Analysis', frame)

    if save_frame:
        frame_filename = os.path.join(save_directory, f'frame_{random.randint(1, 10000)}.jpg')
        cv2.imwrite(frame_filename, frame)
        save_frame = False

        # Append the new image to the images list
        new_image = cv2.imread(frame_filename)
        images.append(new_image)

        # Check if 32 images have been saved
        if len(images) == 36:
            # Generate and save the collage
            generate_and_display_collage(save_directory, r'\Collage',
                                          header_image_path, images, collage_counter)
            collage_counter += 1
            images = []  # Clear the list of images

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Generate and save the final collage for any remaining images
if len(images) > 0:
    generate_and_display_collage(save_directory, r'\Collage', header_image_path,
                                  images, collage_counter)

cap.release()
cv2.destroyAllWindows()
