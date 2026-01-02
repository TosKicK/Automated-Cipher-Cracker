import string
from backend.ciphers.vigenere import decrypt_vigenere

ALPHABET = string.ascii_uppercase

def crack_vigenere(ciphertext, scorer, key_length):
    best_score = float("-inf")
    best_plaintext = ""
    best_key = ""

    def all_keys(prefix, length):
        if len(prefix) == length:
            yield prefix
            return
        for c in ALPHABET:
            yield from all_keys(prefix + c, length)

    # WARNING: exponential – keep key_length small (3–5)
    for key in all_keys("", key_length):
        plaintext = decrypt_vigenere(ciphertext, key)
        score = scorer.score(plaintext)

        if score > best_score:
            best_score = score
            best_plaintext = plaintext
            best_key = key

    return best_plaintext, best_key, best_score
