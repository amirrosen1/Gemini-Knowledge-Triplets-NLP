import wikipedia
import json
from google.cloud import aiplatform
from google.auth import load_credentials_from_file


def initialize_gemini(project_id, key_file, location="us-central1"):
    """
    Initialize the Gemini API client with service account authentication.
    """
    # Load credentials from the service account key file
    credentials = load_credentials_from_file(
        key_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )[0]

    # Initialize AI Platform with the credentials
    aiplatform.init(project=project_id, location=location, credentials=credentials)

    # Define the endpoint for Gemini
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/pelagic-campus-448812-p2/locations/us-central1/endpoints/your-endpoint-id"  # Replace with your actual endpoint ID
    )
    return endpoint


def extract_triplets_with_gemini(endpoint, page_content, max_length=5000):
    """
    Use the Gemini API to extract triplets from the Wikipedia page content.
    Handles long content by truncating.
    """
    truncated_content = page_content[:max_length]  # Truncate content if too long

    prompt = f"""
    Extract triplets (Subject, Relation, Object) from the following text:
    {truncated_content}
    Output the triplets in JSON format, where each triplet is an object with keys: Subject, Relation, and Object.
    """
    response = endpoint.predict(instances=[{"content": prompt}])
    return response.predictions[0]  # Return the extracted triplets


def fetch_wikipedia_content(pages):
    """
    Fetch Wikipedia content for the given list of page titles.
    """
    content = {}
    for page in pages:
        try:
            # Explicitly disable auto-suggest to avoid incorrect page matches
            content[page] = wikipedia.page(page, auto_suggest=False).content
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation error for {page}: {e}")
        except wikipedia.PageError as e:
            print(f"Page error for {page}: {e}")
        except Exception as e:
            print(f"Error fetching content for {page}: {e}")
    return content


def save_to_json(data, filename):
    """
    Save data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    # Set up Google Cloud project details
    project_id = "pelagic-campus-448812-p2"  # Replace with your actual project ID
    key_file = "pelagic-campus-448812-p2-8ae4a914be34.json"  # Path to your JSON key file

    # Initialize the Gemini API client
    endpoint = initialize_gemini(project_id, key_file)

    # Define Wikipedia pages to fetch
    pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]  # Ensure proper titles

    # Fetch Wikipedia content and extract triplets
    print("Fetching Wikipedia pages...")
    wikipedia_content = fetch_wikipedia_content(pages)

    print("Extracting triplets using Gemini...")
    all_triplets = {}
    for page, content in wikipedia_content.items():
        if not content:
            print(f"Skipping {page} due to missing content.")
            continue
        print(f"Processing: {page}")
        try:
            triplets = extract_triplets_with_gemini(endpoint, content)
            all_triplets[page] = triplets
        except Exception as e:
            print(f"Error processing page {page}: {e}")

    # Save the extracted triplets to a JSON file
    save_to_json(all_triplets, "gemini_results.json")
    print("Results saved to gemini_results.json")


if __name__ == "__main__":
    main()
