import json

path = "data/english-quadgrams.txt"

with open(path, "rb") as f:
    for i in range(10):
        line = f.readline()
        print("RAW:", line[:100])
        try:
            decoded = line.decode("utf-8", errors="ignore")
            print("DECODED:", decoded)
            print("JSON:", json.loads(decoded))
        except Exception as e:
            print("ERROR:", e)
        print("-" * 50)
