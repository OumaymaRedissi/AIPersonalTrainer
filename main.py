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
LSTM_PATH = ('inference_models\LSTM_model___Date_Time_2023_09_09__19_04___Loss_0.27015820145606995___Accuracy_0'
             '.9068396091461182.h5')


def show_about_project_ui():
    st.markdown('# About AI-FitTrainer')

    st.write('Welcome to AI-FitTrainer, your innovative AI-powered solution designed to help you improve your squat '
             'technique and achieve your fitness goals.')
    st.write('With AI-FitTrainer, you can:')
    _, col, _ = st.columns([0.15, 0.7, 0.15])
    with col:
        st.image("img/features.png", use_column_width=True)

    st.write('Whether you are looking to master the basics of squats or refine your form, AI-FitTrainer is here to '
             'support you throughout your fitness journey.')


def show_assist_squat_ui():
    lstm_model = load_lstm_model(LSTM_PATH)
    feature_extraction_model = load_features_extractor(IMG_SIZE)
    llm_model = load_llm()

    predicted_label = ""
    trainer_feedback = ""

    st.markdown(
        """
        <style>
            /* Update text color and font */
            .title-text {
                color: #4CAF50; /* Green color */
                font-family: "Helvetica Neue", sans-serif;
            }
            .subheader-text {
                color: #333; /* Dark gray color */
                font-family: "Arial", sans-serif;
            }
            .feedback-text {
                color: #E57373; /* Red color */
                font-family: "Verdana", sans-serif;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('# Assist My Squat ! ', )
    st.markdown(
        '<p class="subheader-text">Ready to improve your squat technique? Upload a video of your squat performance '
        'and get personalized feedback.</p>',
        unsafe_allow_html=True)

    # Upload video section
    st.subheader("Upload Your Video")
    st.write('Select a video file from your device to begin.')
    col1, col2 = st.columns([0.5, 0.5])
    activate_col2 = False

    with col1:
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])

        if uploaded_file is not None:
            video_path = os.path.join("C:/Users/redis/Documents/tempDir", uploaded_file.name.split("/")[-1])
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Uploaded Successfully")

        if uploaded_file is not None:
            # Processing and feedback generation
            st.subheader("Analyze Your Squat")
            st.write("Once uploaded, we'll analyze your squat technique and provide feedback.")

            progress_bar = st.progress(0)

            frames = frame_generator(video_path, IMG_SIZE)

            if frames:
                progress_bar.progress(25)

                features = extract_features(frames, feature_extraction_model, SEQUENCE_LENGTH)
                progress_bar.progress(50)
                if len(features) > 0:
                    progress_bar.progress(75)

                    predicted_class = np.argmax(lstm_model.predict(np.expand_dims(features, axis=0)), axis=1)[0]
                    predicted_label = CLASSES_LIST[predicted_class]
                    trainer_feedback = feedback_generator(predicted_label, llm_model)
                    progress_bar.progress(100)
                    activate_col2 = True

    with col2:
        if activate_col2:
            # Display feedback and suggestions
            st.subheader("Feedback and Suggestions")
            st.write("Here's the analysis of your squat technique:")
            st.markdown(
                '<p class="subheader-text">Predicted Squat Technique: <span class="feedback-text">{}</span></p>'.format(
                    predicted_label), unsafe_allow_html=True)
            st.markdown('<p class="subheader-text">Trainer Feedback:</p>', unsafe_allow_html=True)
            st.markdown('<p class="feedback-text">{}</p>'.format(trainer_feedback), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI-FitTrainer",
        page_icon="img/aifittrainer.png",
        layout="wide",
        initial_sidebar_state='auto'
    )

    st.sidebar.image("img/ai-fit-trainer.png", use_column_width=True)
    st.sidebar.title('AI-FitTrainer')
    st.sidebar.subheader('Deep Learning-based Personal Fitness Assistant')
    st.sidebar.write(
        ' AI-FitTrainer is an innovative AI-powered solution designed to help you improve your squat technique and '
        'achieve your fitness goals.')
    app_mode = st.sidebar.selectbox('Choose the app mode', ['About Project', 'Assist My Squat'])

    if app_mode == 'About Project':
        show_about_project_ui()

    elif app_mode == "Assist My Squat":
        show_assist_squat_ui()


if __name__ == "__main__":
    main()
