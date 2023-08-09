import os
import tensorflow as tf
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from langchain.llms import OpenAI
from dotenv import load_dotenv


def load_lstm_model(model_path):
    model_lstm = tf.keras.models.load_model(model_path,
                                            custom_objects={'MultiLabelConfusionMatrix': MultiLabelConfusionMatrix})
    return model_lstm


def load_features_extractor(img_size):
    efficientnet_b4 = tf.keras.applications.EfficientNetB4(input_shape=(img_size, img_size, 3),
                                                           include_top=False,
                                                           weights='imagenet')
    x = efficientnet_b4.output
    pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extraction_model = tf.keras.Model(efficientnet_b4.input, pooling_output)
    return feature_extraction_model


def load_llm():
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.2)
    return llm
