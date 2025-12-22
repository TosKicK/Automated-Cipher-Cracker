import math
import os

def load_quadgrams(filepath):
    quadgrams = {}
    total = 0

    # Resolve absolute path safely
    filepath = os.path.abspath(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            quad, count = parts
            quadgrams[quad] = int(count)
            total += int(count)

    for quad in quadgrams:
        quadgrams[quad] = math.log10(quadgrams[quad] / total)

    floor = math.log10(0.01 / total)
    return quadgrams, floor
