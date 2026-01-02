def encrypt_rail_fence(text, rails):
    if rails <= 1:
        return text

    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1

    for ch in text:
        fence[rail].append(ch)
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1

    return ''.join(''.join(row) for row in fence)


def decrypt_rail_fence(ciphertext, rails):
    if rails <= 1:
        return ciphertext

    pattern = [[None] * len(ciphertext) for _ in range(rails)]
    rail = 0
    direction = 1

    for i in range(len(ciphertext)):
        pattern[rail][i] = '*'
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1

    index = 0
    for r in range(rails):
        for c in range(len(ciphertext)):
            if pattern[r][c] == '*' and index < len(ciphertext):
                pattern[r][c] = ciphertext[index]
                index += 1

    result = []
    rail = 0
    direction = 1
    for i in range(len(ciphertext)):
        result.append(pattern[rail][i])
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1

    return ''.join(result)
