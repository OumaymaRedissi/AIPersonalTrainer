import cv2
import tensorflow as tf
import numpy as np


def frame_generator(video_path, IMG_SIZE):
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(frame_rate * 4)

    frame_indices = np.linspace(0, num_frames - 1, num=40, dtype=int)

    for i in range(num_frames):
        success, frame = cap.read()

        if not success:
            break

        if i in frame_indices:
            frame = frame[:, :, ::-1]
            img = tf.image.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(img)

    cap.release()

    return frames


def extract_features(frames, feature_extraction_model, SEQUENCE_LENGTH):
    selected_frames = frames[:SEQUENCE_LENGTH]
    all_features = []

    for i, frame in enumerate(selected_frames):
        frame = frame[:, :, ::-1]
        features = feature_extraction_model.predict(np.expand_dims(frame, axis=0))
        features = np.squeeze(features)
        all_features.append(features)

    all_features = np.array(all_features)
    return all_features
