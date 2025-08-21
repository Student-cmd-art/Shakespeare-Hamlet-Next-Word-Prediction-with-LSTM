import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_artifacts():
    model = load_model("lstmmodel.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

MAX_LEN = 14 

def predict_next_word_probs(model, sentence, top_k=5):
    seq = tokenizer.texts_to_sequences([sentence])[0]
    seq = pad_sequences([seq], maxlen=MAX_LEN, padding='pre')
    preds = model.predict(seq, verbose=0)[0]
    # Convert index -> word 
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    # collect all predicted probs
    all_word_probs = [(index_to_word.get(i, ''), preds[i]) for i in range(len(preds))]
    # sort descending
    all_word_probs.sort(key=lambda x: x[1], reverse=True)
    
    return all_word_probs[:top_k]

st.title("Shakespeare Next-Word Predictor")
st.write("Type a Shakespearean line below, and I’ll try to guess the *next word*!")

prompt = st.text_input("Enter your prompt:")

top_k = st.slider("Number of suggestions:", 1, 10, value=5)

if st.button("Predict Next Word") and prompt:
    try:
        results = predict_next_word_probs(model, prompt, top_k=top_k)
        st.subheader("Top Suggestions:")
        for word, prob in results:
            st.write(f"**{word}** — {prob:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

with st.expander("How this works"):
    st.write("""
    A LSTM language model was trained on Shakespearean text.
    The input is tokenized using the original training tokenizer, padded to a fixed length,
    and passed to the model. The network predicts the probability of each word in the vocabulary,
    and we return the top-k most likely next words.
    """)
