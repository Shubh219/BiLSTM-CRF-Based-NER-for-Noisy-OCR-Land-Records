import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOAD MODEL & DATA

model = load_model("bilstm_ner_model.keras")

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("idx2tag.pkl", "rb") as f:
    idx2tag = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

#STREAMLIT UI 

st.set_page_config(page_title="BiLSTM NER", layout="centered")

st.title("BiLSTM NER for Noisy OCR Land Records")

st.markdown(
    "This application extracts structured land record entities from **noisy OCR text** using a **BiLSTM-based NER model**."
)

text = st.text_area(
    "Paste Noisy OCR Text Here",
    "Ownar Nme Ramesh Ku mar Village Nag pur Survey No 124A Area 2.5",
    height=120
)

#PREDICTION 

if st.button("Extract Entities"):

    words = text.split()

    seq = pad_sequences(
        [[word2idx.get(w, word2idx["UNK"]) for w in words]],
        maxlen=max_len,
        padding="post"
    )

    preds = model.predict(seq, verbose=0)[0]
    tags = [idx2tag[p.argmax()] for p in preds][:len(words)]

    # -------------------- POST-PROCESSING --------------------

    entities = {}
    current_entity = ""
    current_label = ""

    for word, tag in zip(words, tags):

        if tag.startswith("B-"):
            current_label = tag[2:]
            current_entity = word
            entities.setdefault(current_label, []).append(current_entity)

        elif tag.startswith("I-") and current_label:
            entities[current_label][-1] += " " + word

        else:
            current_label = ""

    # -------------------- DISPLAY OUTPUT --------------------

    st.subheader("Extracted Entities")

    if entities:
        rows = []
        for label, values in entities.items():
            for value in values:
                rows.append({"Entity": label, "Value": value})

        df = pd.DataFrame(rows)
        st.table(df)

    else:
        st.warning("No entities detected.")
