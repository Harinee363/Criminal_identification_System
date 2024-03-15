import cv2
import numpy as np
import os

size = 2
lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create()

haar_cascade = cv2.CascadeClassifier('face_cascade.xml')

def train_model():
    model = cv2.face.LBPHFaceRecognizer_create()

    fn_dir = 'face_samples'

    print('Training...')

    images, labels, names = [], [], {}
    id = 0

    for subdir in os.listdir(fn_dir):
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)

        for filename in os.listdir(subjectpath):
            _, f_extension = os.path.splitext(filename)
            if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                print(f"Skipping {filename}, wrong file type")
                continue

            path = os.path.join(subjectpath, filename)
            label = id

            # Add to training data
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

    # Create Numpy arrays
    images, labels = np.array(images), np.array(labels)
    
    # OpenCV trains a model from the images
    model.train(images, labels)

    return model, names

def train_model_incremental(model):
    fn_dir = 'face_samples'
    print('Training...')

    label_id_map = {}

    for subdir in os.listdir(fn_dir):
        if subdir not in label_id_map:
            label_id_map[subdir] = len(label_id_map)

        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            _, f_extension = os.path.splitext(filename)
            if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                print(f"Skipping {filename}, wrong file type")
                continue

            path = os.path.join(subjectpath, filename)
            label = label_id_map[subdir]

            # Load and train on new image
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            model.update([image], [label])

    return model, label_id_map

def detect_faces(gray_frame):
    global size, haar_cascade

    # Resize to speed up detection (optional, change size above)
    mini_frame = cv2.resize(gray_frame, (gray_frame.shape[1] // size, gray_frame.shape[0] // size))

    # Detect faces and loop through each one
    faces = haar_cascade.detectMultiScale(mini_frame)
    return faces

def recognize_face(model, frame, gray_frame, face_coords, label_id_map):
    img_width, img_height = 112, 92
    recognized = []
    recog_names = []

    for face_i in face_coords:
        # Coordinates of face after scaling back by `size`
        x, y, w, h = [v * size for v in face_i]
        face = gray_frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))

        # Try to recognize the face
        prediction, confidence = model.predict(face_resize)

        if confidence < 95 and label_id_map[prediction] not in recog_names:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            recog_names.append(label_id_map[prediction])
            recognized.append((label_id_map[prediction].capitalize(), confidence))
        elif confidence >= 95:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, recognized

def main():
    global lbph_face_recognizer

    (model, label_id_map) = train_model_incremental(lbph_face_recognizer)

    # You can now use the model for recognition.

    # Example:
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     face_coords = detect_faces(gray_frame)
    #     frame, recognized = recognize_face(model, frame, gray_frame, face_coords, label_id_map)
    #     cv2.imshow('Recognition', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
