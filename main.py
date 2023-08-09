import os
import numpy as np
import streamlit as st
from backend.data_preprocessing import frame_generator, extract_features
from backend.feedback_generation import feedback_generator
from backend.model_inference import load_lstm_model, load_features_extractor, load_llm

# Constants
IMG_SIZE = 380
SEQUENCE_LENGTH = 40
CLASSES_LIST = ['good', 'bad_inner_thigh', 'bad_toe', 'bad_shallow', 'bad_back_round', 'bad_back_warp', 'bad_head']
LSTM_PATH = ('inference_models/LSTM_model___Date_Time_2023_07_14__11_23_24___Loss_0.15005652606487274___Accuracy_0'
             '.9453616142272949.h5')


def show_about_project_ui():
    st.markdown('# About AI-FitTrainer')
    st.write('Welcome to AI-FitTrainer, your personal AI assistant for analyzing squats!')
    st.write(
        'AI-FitTrainer is an innovative AI-powered solution designed to help you improve your squat technique and '
        'achieve your fitness goals. Whether you are a beginner or an experienced fitness enthusiast, AI-FitTrainer '
        'offers personalized guidance and feedback to enhance your training experience.')
    st.write('With AI-FitTrainer, you can:')
    st.write(
        '- Upload recorded videos: Upload videos of your squat performances, and AI-FitTrainer will analyze them to '
        'evaluate your squat technique.')
    st.write(
        '- Get comprehensive feedback: AI-FitTrainer utilizes state-of-the-art computer vision algorithms to assess '
        'your squat technique accurately. It identifies common mistakes and provides detailed feedback on how to '
        'correct them.')
    st.write(
        '- Access personalized instructions: AI-FitTrainer generates contextually relevant and informative text-based '
        'instructions to guide you in improving your squat technique.')
    st.write(
        '- Enjoy convenience and flexibility: AI-FitTrainer is accessible anytime, anywhere, allowing you to train at '
        'your convenience without the need for a personal trainer or gym membership.')
    st.write(
        'Whether you are looking to master the basics of squats or refine your form, AI-FitTrainer is here to support '
        'you throughout your fitness journey. Start using AI-FitTrainer today and unlock the full potential of your '
        'squat training!')


def show_realtime_assistance_ui():
    st.header("Real-time Video Analysis")
    st.header("Real-time Video Analysis")
    st.markdown('This mode enables real-time video analysis of your squat technique using your webcam.')
    st.markdown('Click on the "Start" button below to use your webcam and receive live feedback on your squat form.')
    st.write(
        'Once you start the webcam, perform squats in front of it, and the AI trainer will provide feedback in '
        'real-time.')
    st.write('Please make sure your full body is visible in the webcam feed for accurate analysis.')

    # Start the webcam and perform real-time video analysis


def show_upload_ui():
    lstm_model = load_lstm_model(LSTM_PATH)
    feature_extraction_model = load_features_extractor(IMG_SIZE)
    llm_model = load_llm()

    predicted_label = ""
    trainer_feedback = ""

    st.header("Upload Video")
    st.markdown('In this mode, you can upload a recorded video of your squat performance for analysis and feedback.')
    st.markdown('To get started, click on the "Choose a video..." button and select the video file from your device.')
    st.markdown(
        'Once the video is uploaded, you can click the "Classify The Video" button to analyze the squat technique and '
        'view the results.')
    col1, col2 = st.columns([0.5, 0.5])  # Increase the width of the first column

    with col1:
        # Upload video file
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])

        if uploaded_file is not None:
            # Store the uploaded video locally
            video_path = os.path.join("C:/Users/redis/Documents/tempDir", uploaded_file.name.split("/")[-1])
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Uploaded Successfully")


            progress_bar = st.progress(0)

            # Sample frames from the uploaded video
            frames = frame_generator(video_path, IMG_SIZE)

            if frames:
                # Update progress bar
                progress_bar.progress(25)

                # Extract features from the sampled frames
                features = extract_features(frames, feature_extraction_model, SEQUENCE_LENGTH)
                progress_bar.progress(50)
                if len(features) > 0:
                    # Update progress bar


                    # Classify the features using LSTM model
                    progress_bar.progress(75)

                    predicted_class = np.argmax(lstm_model.predict(np.expand_dims(features, axis=0)), axis=1)[0]
                    predicted_label = CLASSES_LIST[predicted_class]
                    trainer_feedback = feedback_generator(predicted_label,llm_model)
                    progress_bar.progress(100)
    with col2:

        # Provide feedback and suggestions based on the predicted class label
        st.markdown('## Feedback and Suggestions:')
        st.write("##### Predicted Squat Technique: ", predicted_label)
        st.write(trainer_feedback)


def main():
    st.set_page_config(
        page_title="AI-FitTrainer",
        page_icon="img/aifittrainer.jpg",
        layout="wide",
        initial_sidebar_state="auto"
    )

    st.sidebar.image("img/ai-fit-trainer.png", width=300)
    st.sidebar.title('AI-FitTrainer')
    st.sidebar.subheader('Deep Learning-based Personal Fitness Assistant')

    app_mode = st.sidebar.selectbox('Choose the app mode',
                                    ['About Project', 'Real-time assistance', 'Upload video'])

    if app_mode == 'About Project':
        show_about_project_ui()
    elif app_mode == "Real-time assistance":
        show_realtime_assistance_ui()
    elif app_mode == "Upload video":
        show_upload_ui()


if __name__ == "__main__":
    main()
