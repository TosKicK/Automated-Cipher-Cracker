# ---------- PATH SETUP ----------
import sys
import os
import math
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# ---------- IMPORTS ----------
import streamlit as st
import matplotlib.pyplot as plt

from backend.ngram.ngram_loader import load_quadgrams
from backend.ngram.scorer import NGramScorer
from backend.cracking.caesar_cracker import crack_caesar

# ---------- DATA PATH ----------
DATA_PATH = os.path.join(BASE_DIR, "data", "english-quadgrams.txt")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Automated Cipher Cracker",
    page_icon="🔐",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align:center;'>🔐 Automated Classical Cipher Cracker</h1>
    <p style='text-align:center; font-size:18px;'>
    Using Frequency Analysis & N-Gram Pattern Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Configuration")
cipher_type = st.sidebar.selectbox(
    "Cipher Type",
    ["Caesar Cipher"]
)

st.sidebar.markdown(
    """
    **Method Used**
    - Letter Frequency
    - Quadgram Scoring
    - Statistical Language Model
    """
)

# ---------- INPUT ----------
st.subheader("📥 Input Ciphertext")
ciphertext = st.text_area(
    "Paste the encrypted text below:",
    height=150,
    placeholder="Enter your ciphertext here..."
)

# ---------- FUNCTIONS ----------
def plot_letter_frequency(text, title):
    text = ''.join(filter(str.isalpha, text.upper()))
    freq = Counter(text)

    letters = sorted(freq.keys())
    counts = [freq[l] for l in letters]

    fig, ax = plt.subplots()
    ax.bar(letters, counts)
    ax.set_title(title)
    ax.set_xlabel("Letters")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ---------- ACTION ----------
if st.button("🚀 Crack Cipher"):
    if not ciphertext.strip():
        st.warning("Please enter some ciphertext.")
    else:
        with st.spinner("Loading N-gram language model..."):
            quadgrams, floor = load_quadgrams(DATA_PATH)
            scorer = NGramScorer(quadgrams, floor)

        with st.spinner("Performing cryptanalysis..."):
            plaintext, key, score = crack_caesar(ciphertext, scorer)

        # ---------- RESULTS ----------
        st.success("✅ Cipher Cracked Successfully")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔓 Decrypted Plaintext")
            st.code(plaintext)

            st.markdown(f"""
            **🔑 Key Found:** `{key}`  
            **📊 Confidence Score:** `{round(score, 2)}`
            """)

        with col2:
            st.subheader("📈 Frequency Analysis")
            plot_letter_frequency(ciphertext, "Ciphertext Letter Frequency")
            plot_letter_frequency(plaintext, "Decrypted Text Letter Frequency")

# ---------- FOOTER ----------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px; color:gray;'>
    Cyber Security Mini Project • Classical Cryptanalysis Demonstration
    </p>
    """,
    unsafe_allow_html=True
)
