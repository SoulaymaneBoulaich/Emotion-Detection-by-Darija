"""Streamlit UI — Shared (merge last once at least one model is trained)."""

import streamlit as st

st.set_page_config(page_title="Darija Emotion Detection", page_icon="🇲🇦")

st.title("Emotion Detection in Moroccan Darija")
st.write("Enter a Darija text to detect perceived emotions.")

text = st.text_area("Darija text", placeholder="اكتب النص هنا بالداريجة المغربية...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # TODO: load best model checkpoint, run inference, show per-emotion scores
        st.info("Model integration pending — connect to trained DarijaBERT checkpoint.")

st.markdown("---")
st.caption("Course project — Team: Soulaymane Boulaich, Fares Ilham, Tasnim Touil")