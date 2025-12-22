class NGramScorer:
    def __init__(self, quadgrams, floor):
        self.quadgrams = quadgrams
        self.floor = floor

    def score(self, text):
        text = ''.join(filter(str.isalpha, text.upper()))
        score = 0

        for i in range(len(text) - 3):
            quad = text[i:i+4]
            score += self.quadgrams.get(quad, self.floor)

        return score
