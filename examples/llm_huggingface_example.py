from transformers import pipeline

def run_demo():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    texts = [
        "This drug is harmful",
        "This medicine is very effective",
        "Side effects are dangerous"
    ]

    for text in texts:
        result = classifier(text)
        print(f"Input: {text}")
        print(f"Output: {result}\n")

if __name__ == "__main__":
    run_demo()