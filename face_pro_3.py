import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import tensorflow as tf
import sqlite3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# def get_face_info(name):
#     conn = sqlite3.connect('face_database.db')
#     c = conn.cursor()
#
#     c.execute('SELECT * FROM faces WHERE first_name = ?', (name,))
#     face_info = c.fetchone()
#
#     conn.close()
#     return face_info


def create_emotion_model():
    emotion_model = tf.keras.models.Sequential()
    emotion_model.add(tf.keras.layers.Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
    emotion_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    emotion_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    emotion_model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    emotion_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    emotion_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    emotion_model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    emotion_model.add(tf.keras.layers.Flatten())
    emotion_model.add(tf.keras.layers.Dense(1024, activation="relu"))
    emotion_model.add(tf.keras.layers.Dropout(0.2))
    emotion_model.add(tf.keras.layers.Dense(1024, activation="relu"))
    emotion_model.add(tf.keras.layers.Dropout(0.2))
    emotion_model.add(tf.keras.layers.Dense(7, activation="softmax"))
    return emotion_model


def load_face_encodings(image_files, names):
    known_face_encodings = []
    known_face_names = []
    for images, name in zip(image_files, names):
        for image_file in images:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(image)

            if boxes is not None:
                box = boxes[0].astype(int)
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = face / 255.0
                face = torch.tensor(face.transpose((2, 0, 1))).float().to(device).unsqueeze(0)
                embedding = resnet(face).detach().cpu().numpy().flatten()

                known_face_encodings.append(embedding)
                known_face_names.append(name)
    return known_face_encodings, known_face_names


# import sqlite3


# def get_face_info(name):
#     conn = sqlite3.connect('face_database.db')
#     c = conn.cursor()
#
#     c.execute('SELECT * FROM faces WHERE first_name = ?', (name,))
#     face_info = c.fetchone()
#
#     conn.close()
#     return face_info


def detect_and_display_faces(video_capture, known_face_encodings, known_face_names, emotion_model):
    max_line_length = 124  # choose a length that is suitable for your case

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                box = box.astype(int)
                face = frame[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = face / 255.0
                face = torch.tensor(face.transpose((2, 0, 1))).float().to(device).unsqueeze(0)
                embedding = resnet(face).detach().cpu().numpy().flatten()

                distances = np.linalg.norm(known_face_encodings - embedding, axis=1)
                argmin = distances.argmin()
                min_distance = distances[argmin]

                name = "Unknown"
                if min_distance < 1:
                    name = known_face_names[argmin]

                    face_info = "get_face_info(name)"

                    face_gray = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (48, 48))
                    face_gray = face_gray / 255.0
                    face_gray = np.reshape(face_gray, (1, 48, 48, 1))
                    emotion = emotion_labels[np.argmax(emotion_model.predict(face_gray, verbose=0))]

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.rectangle(frame, (box[0], box[3] - 35), (box[2], box[3]), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f'{name}: {emotion}', (box[0] + 6, box[3] - 6), font, 1.0, (255, 255, 255), 1)

                    line = "\rFace info: " + str(face_info) + ", Emotion: " + emotion
                    line = line.ljust(max_line_length)
                    print(line, end="", flush=True)
                else:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.rectangle(frame, (box[0], box[3] - 35), (box[2], box[3]), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (box[0] + 6, box[3] - 6), font, 1.0, (255, 255, 255), 1)

                    line = "\rUnknown face detected"
                    line = line.ljust(max_line_length)
                    print(line, end="", flush=True)
        else:
            line = "\rNo faces detected"
            line = line.ljust(max_line_length)
            print(line, end="", flush=True)

        cv2.imshow('Video', frame)
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


def main():
    emotion_model = create_emotion_model()
    emotion_model.load_weights('facial_expression_model_weights.h5')

    image_files = [
        ["sanjar01.jpg", "sanjar02.jpg", "sanjar03.jpg"],
        ["diyor01.jpg", "diyor02.jpg", "diyor03.jpg"],
        ["abdusamad01.jpg"],
        ["javohir01.jpg", "javohir02.jpg"]
    ]
    names = ["Sanjar", "Diyor", "Abdusamad", "Javohir"]

    known_face_encodings, known_face_names = load_face_encodings(image_files, names)

    video_capture = cv2.VideoCapture(0)
    detect_and_display_faces(video_capture, known_face_encodings, known_face_names, emotion_model)
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
