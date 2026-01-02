from backend.ciphers.rail_fence import decrypt_rail_fence

def crack_rail_fence(ciphertext, scorer, max_rails=10):
    best_score = float("-inf")
    best_plaintext = ""
    best_rails = 0

    for rails in range(2, max_rails + 1):
        plaintext = decrypt_rail_fence(ciphertext, rails)
        score = scorer.score(plaintext)

        if score > best_score:
            best_score = score
            best_plaintext = plaintext
            best_rails = rails

    return best_plaintext, best_rails, best_score
