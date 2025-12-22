import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
from backend.ngram.ngram_loader import load_quadgrams
from backend.ngram.scorer import NGramScorer
from backend.cracking.caesar_cracker import crack_caesar


DATA_PATH = os.path.join(BASE_DIR, "data", "english-quadgrams.txt")

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Cipher Cracker", layout="centered")

st.title("🔐 Automated Classical Cipher Cracker")
st.caption("Using N-Gram Pattern Learning")

ciphertext = st.text_area("Enter Ciphertext")

if st.button("Crack Caesar Cipher"):
    if not ciphertext.strip():
        st.warning("Please enter some ciphertext.")
    else:
        with st.spinner("Loading N-gram model..."):
            quadgrams, floor = load_quadgrams(DATA_PATH)
            scorer = NGramScorer(quadgrams, floor)

        with st.spinner("Cracking in progress..."):
            plaintext, key, score = crack_caesar(ciphertext, scorer)

        st.success("Cracking Completed")

        st.subheader("Decrypted Plaintext")
        st.code(plaintext)

        st.write(f"🔑 **Key Found:** {key}")
        st.write(f"📊 **N-Gram Score:** {score}")