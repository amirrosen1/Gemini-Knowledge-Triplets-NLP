import random
import wikipedia
import spacy
from tqdm import tqdm


def extract_entity_relations(text_data):
    """
    Extract (Entity1, Relation, Entity2) tuples using POS-based rules.

    Parameters:
        text_data (str): Input text to analyze.

    Returns:
        list of tuples: Extracted triplets (Entity1, Relation, Entity2).
    """
    nlp_tool = spacy.load("en_core_web_sm")
    analyzed_doc = nlp_tool(text_data)
    extracted_relations = []

    # Find sequences of proper nouns
    proper_entities = [word for word in analyzed_doc if word.pos_ == "PROPN"]

    for idx in range(len(proper_entities) - 1):
        entity_one = proper_entities[idx]
        entity_two = proper_entities[idx + 1]

        # Analyze tokens between the two proper nouns
        tokens_in_between = analyzed_doc[entity_one.i + 1: entity_two.i]

        if tokens_in_between and any(token.pos_ == "VERB" for token in tokens_in_between) and \
                all(token.pos_ != "PUNCT" for token in tokens_in_between):

            # Construct the relation from VERB and ADP tokens
            relation_terms = [token.text for token in tokens_in_between if token.pos_ in {"VERB", "ADP"}]
            relation_string = " ".join(relation_terms)

            if relation_string:
                extracted_relations.append((entity_one.text, relation_string, entity_two.text))

    return extracted_relations


def extract_relation_tuples_with_dependencies(input_text):
    """
    Extract (Entity1, Relation, Entity2) tuples using dependency trees.

    Parameters:
        input_text (str): Input text to analyze.

    Returns:
        list of tuples: Extracted triplets (Entity1, Relation, Entity2).
    """
    nlp_tool = spacy.load("en_core_web_sm")
    parsed_text = nlp_tool(input_text)
    dependency_triplets = []

    def collect_proper_entity(node):
        return {node.text}.union({child.text for child in node.children if child.dep_ == "compound"})

    # Identify proper noun heads
    entity_heads = [token for token in parsed_text if token.pos_ == "PROPN" and token.dep_ != "compound"]

    for head_one in entity_heads:
        for head_two in entity_heads:
            if head_one == head_two:
                continue

            entity_one_group = collect_proper_entity(head_one)
            entity_two_group = collect_proper_entity(head_two)

            # Check Condition 1: Shared head with nsubj and dobj dependencies
            if head_one.head == head_two.head:
                shared_root = head_one.head
                if shared_root.dep_ == "ROOT" and shared_root.pos_ == "VERB":
                    if head_one.dep_ == "nsubj" and head_two.dep_ == "dobj":
                        dependency_triplets.append(
                            (" ".join(entity_one_group), shared_root.text, " ".join(entity_two_group)))

            # Check Condition 2: Parent-child-grandparent relationship
            if head_one.head == head_two.head.head:
                parent_node = head_one.head
                grandparent_node = head_two.head
                if parent_node.dep_ == "ROOT" and parent_node.pos_ == "VERB":
                    if head_one.dep_ == "nsubj" and grandparent_node.dep_ == "prep" and head_two.dep_ == "pobj":
                        relation_text = f"{parent_node.text} {grandparent_node.text}"
                        dependency_triplets.append(
                            (" ".join(entity_one_group), relation_text, " ".join(entity_two_group)))

    return dependency_triplets


def evaluate_methods():
    """
    Evaluate triplet extraction methods on specified Wikipedia pages.
    """
    test_pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]
    evaluation_data = []

    # Use tqdm to visualize the progress of the loop
    for page_name in tqdm(test_pages, desc="Processing Wikipedia Pages"):
        page_content = wikipedia.page(page_name, auto_suggest=False).content

        pos_results = extract_entity_relations(page_content)
        dep_results = extract_relation_tuples_with_dependencies(page_content)

        pos_samples = random.sample(pos_results, min(5, len(pos_results)))
        dep_samples = random.sample(dep_results, min(5, len(dep_results)))

        evaluation_data.append({
            "Page": page_name,
            "POS Count": len(pos_results),
            "Dependency Count": len(dep_results),
            "POS Samples": pos_samples,
            "Dependency Samples": dep_samples
        })

    return evaluation_data


if __name__ == "__main__":
    results = evaluate_methods()
    for result in results:
        print(f"Page: {result['Page']}")
        print(f"POS Count: {result['POS Count']}")
        print(f"Dependency Count: {result['Dependency Count']}")
        print("POS Samples:", result['POS Samples'])
        print("Dependency Samples:", result['Dependency Samples'])
        print("=" * 200)
