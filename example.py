import wikipedia
import spacy
from tqdm import tqdm

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")


# Function to extract triplets using POS tagging
def extract_triplets_pos(page_content):
    doc = nlp(page_content)
    triplets = []

    proper_nouns = [token for token in doc if token.pos_ == "PROPN"]
    for i in range(len(proper_nouns) - 1):
        subj = proper_nouns[i]
        obj = proper_nouns[i + 1]
        relation_tokens = doc[subj.i + 1: obj.i]

        # Check if tokens between subj and obj contain a verb
        if any(token.pos_ == "VERB" for token in relation_tokens):
            relation = " ".join(
                token.text for token in relation_tokens if token.pos_ in ["VERB", "ADP"]
            )
            triplets.append((subj.text, relation, obj.text))

    return triplets


# Function to extract triplets using dependency trees
def extract_triplets_dependency(page_content):
    doc = nlp(page_content)
    triplets = []

    proper_noun_heads = [
        token
        for token in doc
        if token.pos_ == "PROPN" and token.dep_ != "compound"
    ]

    for h1 in proper_noun_heads:
        for h2 in proper_noun_heads:
            if h1 == h2:
                continue

            # Condition #1
            if (
                    h1.head == h2.head
                    and h1.dep_ == "nsubj"
                    and h2.dep_ == "dobj"
            ):
                triplets.append((h1.text, h1.head.text, h2.text))

            # Condition #2
            elif (
                    h1.head == h2.head.head
                    and h1.dep_ == "nsubj"
                    and h2.head.dep_ == "prep"
                    and h2.dep_ == "pobj"
            ):
                relation = f"{h1.head.text} {h2.head.text}"
                triplets.append((h1.text, relation, h2.text))

    return triplets


# Evaluate triplet extractors on Wikipedia pages
def evaluate_extractors(page_titles):
    results = {}

    for title in tqdm(page_titles, desc="Processing Wikipedia pages"):
        page_content = wikipedia.page(title).content

        # Extract triplets using POS tagging
        pos_triplets = extract_triplets_pos(page_content)

        # Extract triplets using dependency trees
        dep_triplets = extract_triplets_dependency(page_content)

        results[title] = {
            "POS": pos_triplets,
            "Dependency": dep_triplets,
        }

    return results


# Function to display random sample of triplets for validation
def sample_and_validate(triplets, sample_size=5):
    import random

    sampled_triplets = random.sample(triplets, min(sample_size, len(triplets)))
    for triplet in sampled_triplets:
        print("Triplet:", triplet)
        print("Valid (Y/N)?")
        input()  # User inputs validation


# Main execution
if __name__ == "__main__":
    wikipedia_titles = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]
    extraction_results = evaluate_extractors(wikipedia_titles)

    for title in wikipedia_titles:
        print(f"\nResults for {title}:")
        print("Using POS Tagging:", len(extraction_results[title]["POS"]), "triplets")
        print("Using Dependency Trees:", len(extraction_results[title]["Dependency"]), "triplets")

        print("\nSample Validation for POS Tagging:")
        sample_and_validate(extraction_results[title]["POS"], sample_size=5)

        print("\nSample Validation for Dependency Trees:")
        sample_and_validate(extraction_results[title]["Dependency"], sample_size=5)
