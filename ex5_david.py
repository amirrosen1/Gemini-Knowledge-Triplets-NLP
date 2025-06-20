import random
import wikipedia
import spacy

def extract_triplets(text):
    """
    Extracts triplets (Subject, Relation, Object) from the input text.

    Parameters:
        text (str): The text to extract triplets from.

    Returns:
        list of tuples: Extracted triplets in the form (Subject, Relation, Object).
    """
    # Analyze text with SpaCy
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    triplets = []

    # Iterate through tokens to find proper noun sequences
    proper_nouns = [token for token in doc if token.pos_ == "PROPN"]

    for i in range(len(proper_nouns) - 1):
        subj = proper_nouns[i]
        obj = proper_nouns[i + 1]

        # Get tokens between the proper noun pair
        between_tokens = doc[subj.i + 1: obj.i]

        # Check conditions for valid Relation
        if between_tokens and any(token.pos_ == "VERB" for token in between_tokens) and \
           all(token.pos_ != "PUNCT" for token in between_tokens):

            # Extract Relation as tokens with POS VERB or ADP
            relation_tokens = [token.text for token in between_tokens if token.pos_ in {"VERB", "ADP"}]
            relation = " ".join(relation_tokens)

            if relation:  # Ensure Relation is not empty
                triplets.append((subj.text, relation, obj.text))

    return triplets
## maybe optimize solution
def extract_triplets_optimized(text):
    """
    Extracts optimized triplets (Subject, Relation, Object) from the input text.

    Parameters:
        text (str): The text to extract triplets from.

    Returns:
        list of tuples: Extracted triplets in the form (Subject, Relation, Object).
    """
    # Analyze text with SpaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triplets = []

    # Helper function to filter proper nouns and adjacent relations
    def is_valid_relation(tokens):
        return (
            any(token.pos_ == "VERB" for token in tokens) and
            all(token.pos_ != "PUNCT" for token in tokens)
        )

    # Collect proper nouns and their potential relations
    proper_nouns = [token for token in doc if token.pos_ == "PROPN"]

    for i in range(len(proper_nouns) - 1):
        subj = proper_nouns[i]
        obj = proper_nouns[i + 1]

        # Extract tokens between subject and object
        between_tokens = doc[subj.i + 1: obj.i]

        # Validate and extract relations
        if is_valid_relation(between_tokens):
            relation_tokens = [
                token.text for token in between_tokens if token.pos_ in {"VERB", "ADP"}
            ]
            relation = " ".join(relation_tokens)

            # Filter out noisy or trivial relations
            if relation and len(relation_tokens) <= 3:
                triplets.append((subj.text, relation, obj.text))

    return triplets

def extract_triplets_dependency(text):
    """
    Extracts triplets (Subject, Relation, Object) based on dependency trees in the input text.

    Parameters:
        text (str): The text to extract triplets from.

    Returns:
        list of tuples: Extracted triplets in the form (Subject, Relation, Object).
    """
    # Analyze text with SpaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triplets = []

    # Helper function to construct proper noun sets
    def get_proper_noun_set(token):
        return {token.text}.union({child.text for child in token.children if child.dep_ == "compound"})

    # Identify all proper noun heads
    proper_noun_heads = [
        token for token in doc if token.pos_ == "PROPN" and token.dep_ != "compound"
    ]

    for i, h1 in enumerate(proper_noun_heads):
        for j, h2 in enumerate(proper_noun_heads):
            if i == j:
                continue

            # Extract proper noun sets for h1 and h2
            subj_set = get_proper_noun_set(h1)
            obj_set = get_proper_noun_set(h2)

            # Check Condition #1: h1 and h2 share the same head
            if h1.head == h2.head:
                head = h1.head
                if head.dep_ == "ROOT" and head.pos_ == "VERB":
                    if h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                        triplets.append((" ".join(subj_set), head.text, " ".join(obj_set)))

            # Check Condition #2: h1's parent is h2's grandparent
            h1_parent = h1.head
            h2_parent = h2.head
            if h1_parent == h2_parent.head:
                h = h1_parent
                h_prime = h2_parent
                if h.dep_ == "ROOT" and h.pos_ == "VERB":
                    if h1.dep_ == "nsubj" and h_prime.dep_ == "prep" and h2.dep_ == "pobj":
                        relation = f"{h.text} {h_prime.text}"
                        triplets.append((" ".join(subj_set), relation, " ".join(obj_set)))

    return triplets



def evaluate_extractors():
    wiki_pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]
    results = []

    for page in wiki_pages:
        text = wikipedia.page(page,auto_suggest=False).content

        # Extract triplets using both methods
        pos_triplets = extract_triplets(text)
        dep_triplets = extract_triplets_dependency(text)

        # Random samples for manual verification
        pos_sample = random.sample(pos_triplets, min(5, len(pos_triplets)))
        dep_sample = random.sample(dep_triplets, min(5, len(dep_triplets)))

        results.append({
            "Wikipedia Page": page,
            "POS-Based Triplets Count": len(pos_triplets),
            "Dependency-Based Triplets Count": len(dep_triplets),
            "POS Sample": pos_sample,
            "Dependency Sample": dep_sample
        })

    return results


if __name__ == "__main__":
    evaluation_results = evaluate_extractors()
    for result in evaluation_results:
        print(f"Page: {result['Wikipedia Page']}")
        print(f"POS-Based Triplets Count: {result['POS-Based Triplets Count']}")
        print(f"Dependency-Based Triplets Count: {result['Dependency-Based Triplets Count']}")
        print("POS Sample:", result['POS Sample'])
        print("Dependency Sample:", result['Dependency Sample'])
        print("-" * 50)
