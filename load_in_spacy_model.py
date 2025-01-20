import spacy
import os

def save_spacy_model(model_name="en_core_web_lg", save_path="spacy_model"):
    try:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Download the model if not already present
        spacy.cli.download(model_name)
        
        # Load the model
        nlp = spacy.load(model_name)
        
        # Save the model to disk
        nlp.to_disk(save_path)
        print(f"SpaCy model {model_name} saved to {save_path}")
    except Exception as e:
        print(f"Error saving SpaCy model: {e}")

# Only run this once to generate the saved model
if __name__ == "__main__":
    save_spacy_model()
