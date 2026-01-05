import streamlit as st
import pandas as pd
import plotly.express as px
import math
import re
import os
from collections import Counter

# --- MODULE 1: N-GRAM SCORING ENGINE ---
class NgramScorer:
    def __init__(self):
        self.probs = {}
        self.totals = {}

    def load_dataset(self, filename, n):
        """Standard loader for the specific format: 'WORD COUNT'"""
        if not os.path.exists(filename):
            st.error(f"❌ File not found: {filename}")
            return False
        
        counts = {}
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    ngram, count = parts[0].upper(), int(parts[1])
                    counts[ngram] = count
        
        total = sum(counts.values())
        self.totals[n] = total
        # Log probability to prevent tiny decimal errors (underflow)
        self.probs[n] = {k: math.log10(v/total) for k, v in counts.items()}
        # Default value for unseen n-grams
        self.probs[n]['<UNK>'] = math.log10(0.01 / total)
        return True

    def get_score(self, text, n=4):
        """Higher score = more likely to be English."""
        text = re.sub(r'[^A-Z]', '', text.upper())
        if not text or n not in self.probs: return -999.0
        
        score = 0
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            score += self.probs[n].get(ngram, self.probs[n]['<UNK>'])
        return score

# --- MODULE 2: CIPHER LOGIC ---
class CipherLogic:
    @staticmethod
    def caesar_decrypt(text, shift):
        res = ""
        for char in text.upper():
            if char.isalpha():
                res += chr((ord(char) - 65 - int(shift)) % 26 + 65)
            else:
                res += char
        return res

    @staticmethod
    def vigenere_decrypt(text, key):
        if not key: return text
        key = key.upper()
        res = []
        ki = 0
        for char in text.upper():
            if char.isalpha():
                shift = ord(key[ki % len(key)]) - 65
                res.append(chr((ord(char) - 65 - shift) % 26 + 65))
                ki += 1
            else:
                res.append(char)
        return "".join(res)

    @staticmethod
    def railfence_decrypt(cipher, rails):
        rails = int(rails)
        if rails < 2: return cipher
        # Create the pattern
        idx_map = [0] * len(cipher)
        rail, direction = 0, 1
        for i in range(len(cipher)):
            idx_map[i] = rail
            rail += direction
            if rail == 0 or rail == rails - 1:
                direction *= -1
        
        # Fill the rails
        result = [''] * len(cipher)
        counter = 0
        for r in range(rails):
            for i in range(len(cipher)):
                if idx_map[i] == r:
                    result[i] = cipher[counter]
                    counter += 1
        return "".join(result)

# --- MODULE 3: STREAMLIT FRONTEND ---
st.set_page_config(page_title="Cipher Cracker Pro", layout="wide")
st.title("🕵️‍♂️ Automated Classical Cipher Cracker")

# Sidebar
with st.sidebar:
    st.header("1. Data Configuration")
    scorer = NgramScorer()
    m_ok = scorer.load_dataset("monograms.txt", 1)
    b_ok = scorer.load_dataset("bigrams.txt", 2)
    q_ok = scorer.load_dataset("quadgrams.txt", 4)
    
    if m_ok and b_ok and q_ok:
        st.success("✅ Datasets loaded successfully!")
    
    cipher_type = st.selectbox("2. Select Cipher Type", ["Caesar", "Vigenère", "Rail Fence"])
    
    st.divider()
    manual_mode = st.toggle("🛠 Manual Mode (Force my Key)", value=False)
    
    if manual_mode:
        if cipher_type == "Caesar":
            user_key = st.number_input("Enter Shift (0-25)", 0, 25, 0)
        elif cipher_type == "Rail Fence":
            user_key = st.number_input("Enter Number of Rails", 2, 20, 3)
        else:
            user_key = st.text_input("Enter Keyword", "KEY")
    else:
        user_key = None
        st.info("Automated Mode: The tool will scan all keys and pick the best one.")

# Main Body
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ciphertext Input")
    raw_input = st.text_area("Paste text to crack:", height=250)
    # Pre-processing: Remove non-alpha for analysis, but keep for manual if desired
    clean_input = re.sub(r'[^A-Z]', '', raw_input.upper())
    process_btn = st.button("🚀 Process")

if process_btn and raw_input:
    results = []

    # --- CASE: MANUAL MODE ---
    if manual_mode:
        if cipher_type == "Caesar":
            dec = CipherLogic.caesar_decrypt(raw_input, user_key)
        elif cipher_type == "Rail Fence":
            dec = CipherLogic.railfence_decrypt(clean_input, user_key)
        else:
            dec = CipherLogic.vigenere_decrypt(raw_input, user_key)
        
        results.append({"Key": user_key, "Decrypted": dec, "Score": scorer.get_score(dec)})

    # --- CASE: AUTOMATED CRACKING ---
    else:
        with st.spinner("Cracking..."):
            if cipher_type == "Caesar":
                for s in range(26):
                    dec = CipherLogic.caesar_decrypt(clean_input, s)
                    results.append({"Key": s, "Decrypted": dec, "Score": scorer.get_score(dec)})

            elif cipher_type == "Rail Fence":
                for r in range(2, min(20, len(clean_input))):
                    dec = CipherLogic.railfence_decrypt(clean_input, r)
                    results.append({"Key": r, "Decrypted": dec, "Score": scorer.get_score(dec)})

            elif cipher_type == "Vigenère":
                # Check key lengths from 1 to 12
                for length in range(1, 13):
                    best_key = ""
                    for i in range(length):
                        col = clean_input[i::length]
                        best_col_shift, max_col_score = 0, -float('inf')
                        # For each column, find the best Caesar shift based on monograms
                        for s in range(26):
                            dec_col = CipherLogic.caesar_decrypt(col, s)
                            s_score = scorer.get_score(dec_col, n=1)
                            if s_score > max_col_score:
                                max_col_score, best_col_shift = s_score, s
                        best_key += chr(best_col_shift + 65)
                    
                    dec = CipherLogic.vigenere_decrypt(clean_input, best_key)
                    results.append({"Key": best_key, "Decrypted": dec, "Score": scorer.get_score(dec)})

    # Sorting and Reporting
    df = pd.DataFrame(results).sort_values("Score", ascending=False)
    best = df.iloc[0]

    with col2:
        st.subheader("Top Decryption Result")
        st.success(f"**Key Found/Used:** {best['Key']}")
        st.metric("Confidence Score", round(best['Score'], 2))
        st.text_area("Plaintext:", best['Decrypted'], height=200)

    # Visualization
    st.divider()
    v1, v2 = st.columns(2)
    
    with v1:
        st.subheader("Letter Frequency Analysis")
        counts = Counter(clean_input)
        fig1 = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={'x':'Letter', 'y':'Count'}, color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig1, use_container_width=True)

    with v2:
        st.subheader("Bigram Frequency (Top 10)")
        bigrams = [clean_input[i:i+2] for i in range(len(clean_input)-1)]
        top_bigrams = Counter(bigrams).most_common(10)
        df_b = pd.DataFrame(top_bigrams, columns=['Bigram', 'Count'])
        fig2 = px.bar(df_b, x='Bigram', y='Count', color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Attempts (Ranked by Fitness)")
    st.dataframe(df, use_container_width=True)