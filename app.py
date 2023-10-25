import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('results/transformer')

st.title('English to Spanish Translator')

text = st.text_input('Text to be translated')


vocab_size = 1000
max_length = 50

text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)
text_vec_layer_es = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)


def translate(sentence_en):
    translation = ""
    for word_idx in range(max_length):
        X = np.array([sentence_en])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


def onclick():
    predition = translate(text)
    st.success(predition)


trigger = st.button('Translate', on_click=onclick)
