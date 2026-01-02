import string

ALPHABET = string.ascii_uppercase

def encrypt_vigenere(plaintext, key):
    plaintext = plaintext.upper()
    key = key.upper()
    result = []
    ki = 0

    for ch in plaintext:
        if ch in ALPHABET:
            p = ALPHABET.index(ch)
            k = ALPHABET.index(key[ki % len(key)])
            result.append(ALPHABET[(p + k) % 26])
            ki += 1
        else:
            result.append(ch)

    return ''.join(result)


def decrypt_vigenere(ciphertext, key):
    ciphertext = ciphertext.upper()
    key = key.upper()
    result = []
    ki = 0

    for ch in ciphertext:
        if ch in ALPHABET:
            c = ALPHABET.index(ch)
            k = ALPHABET.index(key[ki % len(key)])
            result.append(ALPHABET[(c - k) % 26])
            ki += 1
        else:
            result.append(ch)

    return ''.join(result)
