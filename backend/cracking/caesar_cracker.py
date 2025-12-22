from backend.ciphers.caesar import decrypt

def crack_caesar(ciphertext, scorer):
    best_score = float("-inf")
    best_text = ""
    best_key = 0

    for key in range(26):
        plaintext = decrypt(ciphertext, key)
        score = scorer.score(plaintext)

        if score > best_score:
            best_score = score
            best_text = plaintext
            best_key = key

    return best_text, best_key, best_score
