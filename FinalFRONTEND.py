import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import cv2
from collections import Counter

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
model = tf.keras.models.load_model('D:/MAJOR/venv/myenv/app/trained_lung_cancer_model_final.h5')

# Define class labels
class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Non-Cancerous', 'Squamous Cell Carcinoma']


# Function to preprocess the uploaded image
def load_and_preprocess_image(img_path, target_size=(350, 350)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to preprocess each video frame
def preprocess_frame(frame, target_size=(350, 350)):
    frame = cv2.resize(frame, target_size)
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image to match training conditions
    return img_array


# Home section content
def home():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">Lung Cancer Detection</h2>
            <p style="color:#555;">
                Lung cancer is one of the most common and serious types of cancer. 
                Early detection is crucial for effective treatment and better outcomes.
            </p>
            <p style="color:#555; font-weight: bold;">
                Together, we can change the future of cancer care through innovation and early diagnosis.
            </p>
            <blockquote style="font-style: italic; color: #777; margin-top: 10px;">
                <strong>EVERY SCAN IS AN OPPORTUNITY TO CHANGE A LIFE.
            </blockquote>
        </div>
        """, unsafe_allow_html=True
    )

# Predict section content for images
def predict():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">Predict Lung Cancer Type from Image</h2>
            <p style="color:#555;">Upload a lung CT scan image to classify it into one of the following categories:</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.write(class_labels)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_path = 'uploaded_image.png'
        image.save(img_path)

        preprocessed_image = load_and_preprocess_image(img_path)
        st.write(f"Preprocessed image shape: {preprocessed_image.shape}")

        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        st.success(f"Prediction: **{predicted_label}**")
        st.write("Prediction probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i] * 100:.2f}%")

        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        st.pyplot(plt)


# New Predict section content for videos
def predict_video():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">Predict Lung Cancer Type from Video</h2>
            <p style="color:#555;">Upload a video to classify lung cancer types based on the frames.</p>
        </div>
        """, unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        video_path = 'uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps)

        frame_count = 0
        predictions_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                img_array = preprocess_frame(frame)
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predictions_list.append(predicted_class_index)

            frame_count += 1

        cap.release()

        predicted_counts = Counter(predictions_list)
        most_common_class_index = predicted_counts.most_common(1)[0][0]
        most_common_class_label = class_labels[most_common_class_index]

        st.success(f"The overall predicted class for the video is: **{most_common_class_label}**")


# About section content
def about():
    st.markdown(
        """
        <div style="width: 100%; padding: 20px;">
            <h2 style="color:#333;">About the Developer</h2>
            <p style="color:#555;">
                Developed by Rohit,Prathamesh and Team. 
                Passionate about AI and ML, I aim to leverage technology to improve healthcare outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True
    )


# Main app function for Streamlit
def main():
    st.markdown(
        """
        <style>
        body {
            background-image: 'Schedule a Consultation.png';
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            padding: 20px;
        }
        h2 {
            font-size: 2rem;
        }
        p {
            font-size: 1.2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Lung Cancer Detection App")
    st.markdown(
        """
        <h2 style="color:#FFA500;">CANCER IS A PART OF LIFE, IT'S NOT OUR WHOLE LIFE...
        TOGHETHER, WE CAN CONQUER CANCER.</h2>
        """, unsafe_allow_html=True
    )

    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    with st.sidebar:
        page = st.selectbox("Select a page", ["Home", "Predict From Image", "Predict From Video", "About"])
        st.session_state.page = page

    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Predict From Image":
        predict()
    elif st.session_state.page == "Predict From Video":
        predict_video()
    elif st.session_state.page == "About":
        about()

    st.markdown(
        """
        <style>
        footer {
            visibility: hidden;
        }
        .footer {
            visibility: visible;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #262730;
            color: white;
            text-align: center;
            padding: 5px;
        }
        </style>
        <div class="footer">
            Developed by Rohit,Prathamesh & TEAM
        </div>
        """, unsafe_allow_html=True
    )


# Run the app
if __name__ == '__main__':
    main()
