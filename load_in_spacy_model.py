import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Save the model to a directory
nlp.to_disk("spacy_model_again")
