# Import required libraries
import pandas as pd
from collections import OrderedDict # For extracting best hyperparameter string representation from saved .csv file to original dtype
import numpy as np
import joblib
from tqdm import tqdm  # Monitors progress of long processing operations with a bar

# Import required libraries for text processing
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
# Create set out of English stopwords
stop_words = set(stopwords.words('english'))
import spacy
import subprocess
import os
# Emotion lexicon for 8 categories of emotions
from nrclex import NRCLex
# Textstat for extracting readability scores
import textstat

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Feature extraction and modellibraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# Web app libraries
import altair as alt
import streamlit as st

# LIME explanation libraries
from lime.lime_text import LimeTextExplainer



class BasicFeatureExtractor:
    """
        Functions for extracting basic features such as frequency counts for parts of speech
        and punctuation marks for pre-processing a fake news text or entire DataFrame 
        for analysis and classification.
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def extractExclamationPointFreqs(self, text):
        """
        A helper function for extracting normalized (by text length in number of tokens)
        frequencies of exclamation points per inputted news text.
            Input Parameters:
                text (str): the news text to extract exclamation point frequencies from.
    
            Output:
                excl_point_freq (float): the normalized exclamation point frequency for the text.
                Normalized by num of word tokens to handle varying text length datasets.
        """
        # Count the number of exclamation points in the text
        exclamation_count = text.count("!")
        # Count word tokens for text length
        word_tokens = word_tokenize(text)
        text_length = len(word_tokens)
        # Normalize the exclamation point frequency
        return exclamation_count / text_length if text_length > 0 else 0 # Handle division-by-zero errs


    def extractThirdPersonPronounFreqs(self, text):
        """
        Extracts the normalized frequency counts of third-person pronouns in the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract pronoun features from.
            
            Output:
                float: Normalized third-person pronoun frequency.
        """
        # Define the list of third-person pronouns
        third_person_pronouns = [
            "he", "she", "it", "they", "he's", "she's", "it's", "they're",
            "they've", "they'd", "they'll", "his", "her", "its", "their",
            "hers", "theirs", "him", "them", "one", "one's", "he'd", "she'd"
        ]

        # Tokenize the text into words
        word_tokens = word_tokenize(text)
        text_length = len(word_tokens)

        # Count frequency of third-person pronouns in the news text, lower to match the list above
        third_person_count = sum(1 for token in word_tokens if token.lower() in third_person_pronouns)

        # Normalize the frequency by text length in word tokens
        return third_person_count / text_length if text_length > 0 else 0



    def extractNounToVerbRatios(self, text):
        """
        Calculates the ratio of all types of nouns to all types of verbs in the text
        using the Penn Treebank POS Tagset and the SpaCy library with the
        "en_core_web_lg" model.
        
            Input Parameters:
                text (str): the news text to extract noun-verb ratio features from.
            
            Output:
                float: Noun-to-verb ratio, or 0.0 if no verbs are present.
        """
        # Convert the text to an NLP doc object using the SpaCy library.
        doc = self.nlp(text)
        
        # Define the Penn Treebank POS tag categories for nouns and verbs
        # Reference here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        
        # Count nouns and verbs based on the Penn Treebank tags
        noun_count = sum(1 for token in doc if token.tag_ in noun_tags)
        verb_count = sum(1 for token in doc if token.tag_ in verb_tags)
        
        # Compute and return the noun-to-verb ratio (should be higher for fake news, more nouns)
        return noun_count / verb_count if verb_count > 0 else 0.0


    def extractCARDINALNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of CARDINAL named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the CARDINAL named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of CARDINAL named entities.
        """
        # Process the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
         # Count how many named entities have the label "CARDINAL"
        cardinal_entity_count = sum(1 for entity in doc.ents if entity.label_ == "CARDINAL")

        # Tokenize the text and calculate its length
        word_tokens = [token for token in doc]
        text_length = len(word_tokens)

        # Return the normalized frequency of CARDIAL named entities
        return cardinal_entity_count / text_length if text_length > 0 else 0.0


    def extractPERSONNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of PERSON named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the PERSON named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of PERSON named entities.
        """
        # Process the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
        
        # Count how many named entities have the label "PERSON"
        person_entity_count = sum(1 for entity in doc.ents if entity.label_ == "PERSON")
        
        # Tokenize the text and calculate its length
        word_tokens = [token for token in doc]
        text_length = len(word_tokens)
        
        # Return the normalized frequency of PERSON named entities
        return person_entity_count / text_length if text_length > 0 else 0.0


    def extractPositiveNRCLexiconEmotionScore(self, text):
        """
        Extracts the POSITIVE emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract POSITIVE emotion score from.
            
            Output:
                float: POSITIVE emotion score.
        """
        # Create an NRC Emotion Lexicon object
        emotion_obj = NRCLex(text)
        
        # Return the POSITIVE emotion score (use "get" to default to 0.0 if not found)
        return emotion_obj.affect_frequencies.get("positive", 0.0)


    def extractTrustNRCLexiconEmotionScore(self, text):
        """
        Extracts the TRUST emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract TRUST emotion score from.
            
            Output:
                float: TRUST emotion score.
        """
        
        # Create an NRC Emotion Lexicon object
        emotion_obj = NRCLex(text)
        
        # Return the TRUST emotion score (use "get" to default to 0.0 if not found)
        return emotion_obj.affect_frequencies.get("trust", 0.0)


    # Helper Function for Flesch-Kincaid Grade Level score
    def extractFleschKincaidGradeLevel(self, text):
        """
        Extracts the Flesch-Kincaid Grade Level score for the input text.
        
        Input Parameters:
            text (str): the news text to calculate the Flesch-Kincaid Grade Level score.
        
        Output:
            float: the Flesch-Kincaid Grade Level score for the text.
        """
        return textstat.flesch_kincaid_grade(text)
        
    # Helper Function for Difficult Words score
    def extractDifficultWordsScore(self, text):
        """
        Extracts the number of difficult words in the input text using the textstat library.
        
        Input Parameters:
            text (str): the news text to calculate the difficult words score.
        
        Output:
            float: the number of difficult words score for the text.
        """
        return textstat.difficult_words(text)

    def extractCapitalLetterFreqs(self, text):
        """
        Extracts the normalized frequency of capital letters in the input text.
        Normalized by the total number of tokens to account for varying text lengths.
    
            Input Parameters:
                text (str): The news text to extract capital letter frequencies from.
            
            Output:
                float: Normalized frequency of capital letters in the text.
        """
        # Count the number of capital letters in the text
        capital_count = sum(1 for char in text if char.isupper())
        
        # Tokenize the text into words to calculate its length
        word_tokens = word_tokenize(text)
        text_length = len(word_tokens)
        
        # Normalize the frequency of capital letters
        return capital_count / text_length if text_length > 0 else 0.0
    
    def extractBasicFeatures(self, df, dataset_name, root_save_path="../FPData/BasicFeatureExtractionDFs",):
        """
        Adds new columns to an inputted DataFrame that stores the frequency of 
        or score for various features derived from the "text" column, hopefully for 
        improved classification using a Passive Aggressive Classifier model.

            Input Parameters:
                df (pd.DataFrame): DataFrame with a "text" column containing news texts.
    
            Output:
                pd.DataFrame: a new DataFrame with the additional feature columns
        """
        # Create a copy of the original DataFrame
        basic_feature_df = df.copy()

        # Get dataset name without the ".csv" at the end
        dataset_name_without_csv_extension = dataset_name.split('.')[0]
        print(f"Extracting features for {dataset_name_without_csv_extension}...")
        
        # Apply the exclamation point frequency helper function...
        print("Extracting normalized exclamation point frequencies...")
        basic_feature_df["exclamation_point_frequency"] = basic_feature_df["text"].progress_apply(self.extractExclamationPointFreqs)
        
        # Apply the third-person pronoun frequency helper function...
        print("Extracting third person pronoun frequencies...")
        basic_feature_df["third_person_pronoun_frequency"] = basic_feature_df["text"].progress_apply(self.extractThirdPersonPronounFreqs)

        # Apply the noun-to-verb ratio helper functionn
        print("Extracting noun-to-verb ratio frequencies...")
        basic_feature_df["noun_to_verb_ratio"] = basic_feature_df["text"].progress_apply(self.extractNounToVerbRatios)

        # Apply the CARDINAL named entity frequency helper function to create a new column
        print("Extracting CARDINAL named entity frequencies...")
        basic_feature_df["cardinal_named_entity_frequency"] = basic_feature_df["text"].progress_apply(self.extractCARDINALNamedEntityFreqs)

        # Apply the PERSON named entity frequency helper function to create a new column
        print("Extracting PERSON named entity frequencies...")
        basic_feature_df["person_named_entity_frequency"] = basic_feature_df["text"].progress_apply(self.extractPERSONNamedEntityFreqs)

        # Apply the NRC Lexicon to get the POSITIVE emotion scores for each text
        print("Extracting NRC Lexicon POSITIVE emotion scores...")
        basic_feature_df["nrc_positive_emotion_score"] = basic_feature_df["text"].progress_apply(self.extractPositiveNRCLexiconEmotionScore)

        # Apply the NRC Lexicon to get the TRUST emotion scores for each text
        print("Extracting NRC Lexicon TRUST emotion scores...")
        basic_feature_df["nrc_trust_emotion_score"] = basic_feature_df["text"].progress_apply(self.extractTrustNRCLexiconEmotionScore)

        # Apply textstat package to get the Flesch-Kincaid U.S. Grade Level readability score
        print("Extracting Flesch-Kincaid U.S. Grade readability scores...")
        basic_feature_df["flesch_kincaid_readability_score"] = basic_feature_df["text"].progress_apply(self.extractFleschKincaidGradeLevel)

        # Apply textstat package to get the difficult_words readability score
        print("Extracting difficult words readability scores...")
        basic_feature_df["difficult_words_readability_score"] = basic_feature_df["text"].progress_apply(self.extractDifficultWordsScore)

        # Extract counts of capital letters for each text normalized by number of tokens
        print("Extracting normalized capital letter frequency scores..")
        basic_feature_df["capital_letter_frequency"] = basic_feature_df["text"].progress_apply(self.extractCapitalLetterFreqs)

        print("\n\n")

        # Save to disk
        basic_feature_df.to_csv(os.path.join(root_save_path, dataset_name), index=False)
        
        return basic_feature_df

    def extractExtraFeaturesColumns(self, df):
        return df[[
        "exclamation_point_frequency", "third_person_pronoun_frequency", "noun_to_verb_ratio",
        "cardinal_named_entity_frequency", "person_named_entity_frequency",
        "nrc_positive_emotion_score", "nrc_trust_emotion_score",
        "flesch_kincaid_readability_score", "difficult_words_readability_score",
        "capital_letter_frequency"
    ]]

    def extractFeaturesForSingleText(self, text):
        """
        Function that extracts all features for a single text instance.
    
        Input Parameters:
            text (str): The text to extract features from.
    
        Output:
            dict: A dictionary of extracted feature values.
        """
        feature_dict = {
            "exclamation_point_frequency": self.extractExclamationPointFreqs(text),
            "third_person_pronoun_frequency": self.extractThirdPersonPronounFreqs(text),
            "noun_to_verb_ratio": self.extractNounToVerbRatios(text),
            "cardinal_named_entity_frequency": self.extractCARDINALNamedEntityFreqs(text),
            "person_named_entity_frequency": self.extractPERSONNamedEntityFreqs(text),
            "nrc_positive_emotion_score": self.extractPositiveNRCLexiconEmotionScore(text),
            "nrc_trust_emotion_score": self.extractTrustNRCLexiconEmotionScore(text),
            "flesch_kincaid_readability_score": self.extractFleschKincaidGradeLevel(text),
            "difficult_words_readability_score": self.extractDifficultWordsScore(text),
            "capital_letter_frequency": self.extractCapitalLetterFreqs(text),
        }
        # Convert to DataFrame (single row)
        feature_df = pd.DataFrame([feature_dict])
        return feature_df




def explainPredictionWithLIME(trained_pipeline, text, feature_extractor, num_features=50, num_perturbed_samples=500):
    """
    Input Parameters:
        trained_pipeline (scikit-learn.Pipeline): a pre-trained Pipeline consisting of tuned TF-IDF, PA Classifier, and Calibrated
            Classifier for returning probabilities.
        text (str): the text to get the prediction for
        feature_extractor (instance of BasicFeatureExtractor class): the object containing a method for extracting the extra
            features for the news text
        num_features (int): number of "important features" contributing to the final prediction that the LIME explainer should
                            output.
        num_perturbed_samples (int): number of times LIME should perturb-and-test the entered original text to see how
                                     changing certain features leads to differences in predicted probabilities
    Output:
        dict: stores different outputs based on the LIME explainer, such as the explanation object, original array of real-fake
              prediction probabilities, a list of tuples for the importance scores for each word and extra feature etc.
    """
    
    print("Initializing the LIME explainer instance...")
    # Instantiate the Lime Text Explainer
    text_explainer = LimeTextExplainer(class_names=["real", "fake"])
    
    print("Extracting features for the input text...")
    # Extract the linguistic features for the single text inputted as "text" in the args
    single_text_df = feature_extractor.extractFeaturesForSingleText(text)
    single_text_df["text"] = text
    
    # Store the original extra features for generating later LIME explanations of how much they contributed to the final prediction
    extra_features = single_text_df.drop("text", axis=1) # Drop the text column to get only extra features columns
    extra_feature_names = extra_features.columns  # Extract the names of the extracted features

    def predictProbaWrapper(texts):
        """
        LIME works by generating perturbations  of the original news text.
        This helper function processes (extracts features) and then predicts the probabilities (2-element array) for
        each of these 'x' perturbed texts.
        Then it uses the trained Pipeline to return arrays of probabilities [prob_real, prob_fake] for each perturbed text.
        """
        
        print(f"Predicting probabilities for {len(texts)} perturbed LIME texts...")
        
        df_list = []  # Store the DataFrames for extracted features + text for all the perturbed texts
        
        # Iterate over perturbed LIME-texs
        for t in texts:
            # Extract features for each perturbation of original text
            df = feature_extractor.extractFeaturesForSingleText(t)
            # Add text column to extracted features single-row DataFrame
            df["text"] = t
            # Add single row DataFrame storing extracted features and text to list for concatenating all perturbed sample information
            df_list.append(df)
            
        # Combine the single-row DataFrames for all perturbed LIME texts into a single DataFrame, vertically stack using concat
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        # Extract the probabilities of all the perturbed feature-text LIME samples using the pipeline
        probs_for_texts = trained_pipeline.predict_proba(combined_df)
        
        return probs_for_texts

    # Use the model to get the prediction for real or fake news for the text (as there is a single text only, access first elem with [0])
    prediction = trained_pipeline.predict(single_text_df)[0]
    text_representation_of_prediction = "Fake News" if prediction == 1 else "Real News"
    print(f"Main prediction: {prediction} = {text_representation_of_prediction}.\nGenerating LIME explanation for the text...")

    # Generate the LIME explanation
    explanation = text_explainer.explain_instance(
        text, # Single original text
        predictProbaWrapper, # Wrapped for extracting the extra features for each perturbed sample of the text and getting prediciton probs
        num_features=num_features, # Number of top word-features to output in explanation (default = 50)
        top_labels=1, # Explain only the top-predicted label
        num_samples=num_perturbed_samples, # Number of perturbations; a higher value inc. accuracy but takes more time. Default = 500
        labels=[prediction] # Get the predicted label for this news text (to explain it)
    )
    
    # Get the text feature explanations
    print("Generating LIME explanations based on 500 perturbed samples...")
    text_features = explanation.as_list(label=prediction)
    # Convert the outputted text_features to a list of tuples storing (feature_name<word>, importance_score)
    text_feature_list = [(feature[0], feature[1]) for feature in text_features
                             if feature[0].lower() not in stop_words] # Filter out text features that are stopwords like "the", "an", etc.
    # Sort the text_feature_list by absolute value of importance scores in descending order
    text_feature_list_sorted = sorted(text_feature_list, key=lambda x: abs(x[1]), reverse=True)
    print(f"text_feature_list_sorted: {text_feature_list_sorted }")
    
    # Generate explanations for extra features based on perturbing them
    print("\nCalculating extra feature importance scores...")

    # Store extra features'importance in this list of tuples (feature_name, feature_importance)
    extra_feature_importances= []

    # Get the probability array [real_news_probability, fake_news_probability] for the single, unperturbed news text
    original_text_probability_array = trained_pipeline.predict_proba(single_text_df)[0]

    # Iterate through the extra features
    for feature in extra_feature_names:
        # Create perturbed version of the DataFrame storing the extra features
        perturbed_df = single_text_df.copy()
        # Extract the original value of the feature whose importance would like to evalute
        original_value = perturbed_df[feature].iloc[0] # Single row frame, so get the row 0 out
        
        # Calculate this particular feature's importance for the prediction by setting its value to 0, and predicting
        # the probability array using the Pipeline for real vs fake news WITHOUT this feature (i.e. set to 0)
        perturbed_df[feature] = 0
        perturbed_probability_array = trained_pipeline.predict_proba(perturbed_df)[0]
        
        # Calculate the feature's importance as difference in probabilities of the prediction (i.e. class with the highest probability)
        importance = original_text_probability_array[prediction] - perturbed_probability_array[prediction]

        # Append importance of the feature to the extra_feature_importances list-of-tuples
        extra_feature_importances.append((feature, importance))
    
    # Sort extra features by absolute value of the feature importance scores (at index 1 in the name-score tuples) in descending order
    extra_feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Extra feature importance scores (sorted in descending order):", extra_feature_importances)

    # Visualize the text with highlighted words based on their importance scores
    highlighted_text = highlightText(text, text_feature_list_sorted, prediction)
    
    # Combine both explanations in the return
    return {
        "explanation_object": explanation,
        "text_features_list": text_feature_list,
        "extra_features_list": extra_feature_importances,
        "highlighted_text": highlighted_text,
        "probabilities": original_text_probability_array,
        "main_prediction": prediction
    }


def highlightText(text, text_feature_list, prediction):
    """
    A helper function for highlighting the words in the input text based on their importance scores and their impact on the prediction.
    Red for words pushing towards fake news (label=1) and blue for words pushing towards real news (label=0).

    Input Parameters:
        text (str): the text to highlight
        text_feature_list (list of tuples): list of word-feature, importance-score tuples outputted by LIME important feature detector
        prediction (int): 0 if real news, 1 if fake news
    Output:
        str: HTML formatted string with highlighted text
    """
    
    # A list to store the dictionaries containing positions, colors and opacities for highlighting parts of the text
    highlight_positions = []
    
    # Calculate the maximum absolute importance score
    max_importance = max(abs(importance) for feature, importance in text_feature_list)

    # Find all positions to highlight first, then sort them
    for feature, importance in text_feature_list:
        pos = 0  # Start position for searching
        while True:
            # Find the next occurrence of the feature
            pos = text.lower().find(feature.lower(), pos)
            if pos == -1:  # No more occurrences found
                break
                
            # Check word boundaries
            boundary_positions = detectWordBoundaries(text, pos, feature)
            if boundary_positions is None:
                pos += 1  # Move to next position
                continue
                
            word_start_pos, word_end_pos = boundary_positions
            
            # Add position for highlighting
            highlight_positions.append({
                "start": word_start_pos,
                "end": word_end_pos,
                "color": "red" if importance > 0 else "blue",
                "opacity": abs(importance) / max_importance if max_importance != 0 else 0,
                "text": text[word_start_pos:word_end_pos]
            })
            
            pos = word_end_pos  # Move past this word

    # Sort positions by start position
    highlight_positions.sort(key=lambda x: x["start"])
    
    # Merge adjacent highlights of the same color
    merged_positions = []
    if highlight_positions:
        current = highlight_positions[0]
        
        for next_pos in highlight_positions[1:]:
            # If next position starts right after current ends (or overlaps) and has same color
            if (next_pos["start"] <= current["end"] + 1 and
                next_pos["color"] == current["color"]):
                # Merge by extending current position
                current["end"] = max(current["end"], next_pos["end"])
                current["opacity"] = max(current["opacity"], next_pos["opacity"])
                current["text"] = text[current["start"]:current["end"]]
            else:
                # If not mergeable, add current to results and move to next
                if next_pos["start"] > current["end"]:  # No overlap
                    merged_positions.append(current)
                    current = next_pos
                else:  # Overlap but different colors - keep the one with higher opacity
                    if next_pos["opacity"] > current["opacity"]:
                        current = next_pos
        
        merged_positions.append(current)  # Add the last position

    # Build the final highlighted text
    result = []
    last_end = 0
    
    for pos in merged_positions:
        # Add non-highlighted text
        if pos["start"] > last_end:
            result.append(text[last_end:pos["start"]])
        
        # Add highlighted text
        color = "rgba(255, 0, 0," if pos["color"] == "red" else "rgba(100,149,237,"  # red or cornflowerblue
        background_style = f"{color}{pos['opacity']})"
        
        result.append(
            f"<span style='background-color:{background_style}; font-weight: bold'>"
            f"{pos['text']}</span>"
        )
        
        last_end = pos["end"]
    
    # Add any remaining text
    if last_end < len(text):
        result.append(text[last_end:])

    # Return the final string (raw HTML)
    return "".join(result)



def detectWordBoundaries(text, word_start_pos, word):
    """
    Helper function to ensure matching whole words only when highlighting text
    Returns the start and end position if it's a valid word boundary, None otherwise

    Input Parameters:
        text (str): the whole text the sub-string to highlight is part of
        word_start_pos (int): starting position of word (feature)
        word (str): the word feature to highlight

    Output:
        tuple of word_start_pos (int), word_end_pos (int): if word is indeed a word surrounded by a boundary
        None: if invalid word boundary
    """
    # Define a set of punctuation characters for detecting word boundaries, such as space, exclamation mark, hyphen
    boundary_chars = set(' .,!?;:()[]{}"\n\t-')
    
    # Check if the word is at the proper word boundaries: either position is the start_position (0) or the
    # previous character before the word is inside of the boundary_chars set for detecting w boundaries
    start_check = word_start_pos == 0 or text[word_start_pos - 1] in boundary_chars
    # Set end position to the character after the word to check if it is a boundary-marking characters
    word_end_pos = word_start_pos + len(word)
    end_check = word_end_pos == len(text) or text[word_end_pos] in boundary_chars # Check if end is either text end or boundary char

    # If word boundary is around this word, then return the starting and end position of the word in the text
    if start_check and end_check:
        return word_start_pos, word_end_pos

    # If not a boundary, return None
    return None


def displayAnalysisResults(explanation_dict, container, news_text, feature_extractor, FEATURE_EXPLANATIONS):

    """
    Displays analysis results including prediction, confidence scores, and feature importance charts.
    
    Input Parameters:
        explanation_dict (dict): dict containing LIME explanation results
        container: Streamlit app container to display the results in
        news_text (str): text to generate prediction and explanation for
        feature_extractor (instance of BasicFeatureExtractor): for processing inputted text to get semantic and linguistic features
        FEATURE_EXPLANATIONS (dict): natural language explanations of the different non-word semantic and linguistic features
    """
    
    # Convert news category label from 0/1 to text labels
    main_prediction = "Fake News" if explanation_dict["main_prediction"] == 1 else "Real News"
    
    # Extract [real, fake] probability array returned from LIME explainer function
    probs = explanation_dict["probabilities"]
    
    # Display the results based on LIME explainer output
    container.subheader("Text Analysis Results")
    container.write(f"**General Prediction:** {main_prediction}")
    container.write(f"**Confidence Scores:**")
    container.write(f"- Real News: {probs[0]:.2%}")
    container.write(f"- Fake News: {probs[1]:.2%}")
    
    # Display highlighted text in an expandable box
    with st.expander("View Highlighted Text"):
        # Format the expandable scroll-box to show highlighted (blue=real, red=fake) text outputted by LIME Explainer, to allow y-scrolling and padding
        st.markdown("""
            <div style='height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>
                {}
            </div>  
            """.format(explanation_dict["highlighted_text"]), unsafe_allow_html=True)
    
    # Creating bar charts for feature importance analysis...
    container.subheader("Feature Importance Analysis")

    # Create two columns for side-by-side bar charts
    col1, col2 = container.columns(2)
    
    # First column: word/text-based features
    with col1:
        col1.write("### Top Text Features")
        # Get the top 10 text features identified by the LIME algorithm into a DataFrame for easier sorting and filtering
        text_features_df = pd.DataFrame(
            explanation_dict["text_features_list"],
            columns=["Feature", "Importance"]
        )
        text_features_df = text_features_df.nlargest(10, "Importance")
        
        # Creates a bar chart for most important text features using the Altair visualization library using text_features_df
        # Q = specifies this feature/value is quantitative, N = specifies it is nominal/categorical
        # sort="-x" = sort features by x=value (LIME score or "Importance")
        text_chart = alt.Chart(text_features_df).mark_bar().encode( # Use mark_bar to create bar_chart
            x=alt.X("Importance:Q", title="Impact Strength"),
            y=alt.Y("Feature:N", sort="-x", title=None,
                    axis=alt.Axis(
                        labelLimit=200,
                        labelFontSize=10,
                        labelAngle=0,
                        ticks=False
                    )), # Customizes the y-axis appearance for label length threshold, size, removing tick-marks etc.
            color=alt.condition(
                alt.datum.Importance > 0,
                alt.value("red"),
                alt.value("blue")
            ), # Sets the bar colors based on Importance value --> if Importance is positive, set to red (pushes to fake news), if negative set to blue (push to real news)
            tooltip=["Feature", "Importance"] # Add "tooltips": explanations that appear when hovering above the bar
        ).properties(
            title="Top 10 Word Features",
            height=300,
            width=400
        ) # Sets the chart title and size
        
        # Show the chart now in the Streamlit column
        col1.altair_chart(text_chart, use_container_width=True)

    # Column for analyzing importance of non-text based features
    with col2:
        col2.write("### Top Extra Features")
        
        # Create a DataFrame from the extra features list returned by the LIME explainer function
        extra_features_df = pd.DataFrame(
            explanation_dict["extra_features_list"],
            columns=["Feature", "Importance"]
        )
        
        # Sort the features by absolute importance and get the top 10 features out
        extra_features_df["Absolute Importance"] = extra_features_df["Importance"].abs() # Create new absolute importance column first
        extra_features_df = extra_features_df.nlargest(10, "Absolute Importance") # Extract the 10 largest features (most importance) using the new column
        
        # Add the original feature name before mapping to the explanations
        extra_features_df["Original Feature"] = extra_features_df['Feature']
        
        # Creates a more accessible feature name mapping for readability for users to understand better
        feature_name_mapping = {
            "exclamation_point_frequency": "Exclamation Point Usage",
            "third_person_pronoun_frequency": "3rd Person Pronoun Usage",
            "noun_to_verb_ratio": "Noun/Verb Ratio",
            "cardinal_named_entity_frequency": "Number Usage",
            "person_named_entity_frequency": "Person Name Usage",
            "nrc_positive_emotion_score": "Sentiment/Emotion Polarity",
            "nrc_trust_emotion_score": "Trust Score",
            "flesch_kincaid_readability_score": "Readability Grade",
            "difficult_words_readability_score": "Complex Words",
            "capital_letter_frequency": "Capital Letter Usage"
        }
        
        # Maps the features to their more readable names
        extra_features_df["Feature"] = extra_features_df["Feature"].map(feature_name_mapping).fillna(extra_features_df["Feature"])
        
        # Maps the explanations to their natural language explanation for users
        extra_features_df["Explanation"] = extra_features_df["Original Feature"].map(FEATURE_EXPLANATIONS)
        
        # Creates the bar chart with enhanced tooltips for explaining what features mean to users
        extra_chart = alt.Chart(extra_features_df).mark_bar().encode(
            x=alt.X("Importance:Q", title="Impact Strength"),
            y=alt.Y("Feature:N",
                    title=None,
                    sort=alt.EncodingSortField(
                        field="Absolute Importance", # Sort features by ABSOLUTE value of importance in descending order (most important to least important_
                        order="descending"
                    ),
                    axis=alt.Axis(
                        labelLimit=200,
                        labelFontSize=10,
                        labelAngle=0,
                        ticks=False
                    )), #  Format the axis labels
            color=alt.condition(
                alt.datum.Importance > 0,
                alt.value("red"),
                alt.value("blue")
            ), # Set bar colors
            tooltip=["Feature", "Importance", "Explanation"]  # Add hoverable tooltip explanations
        ).properties(
            title="Top 10 Linguistic Features",
            height=300,
            width=400
        ) # Add title and configure chart size
        
        # Show the chart in the column
        col2.altair_chart(extra_chart, use_container_width=True)
        
        # Add a legend explanation for the color-coded bar charts and highlighted text
        container.markdown(f"""
            **Color Legend:**
            - 🔵 **Blue bars**: Features pushing towards real news classification
            - 🔴 **Red bars**: Features pushing towards fake news classification
            
            The length of each bar and color represent how strongly this feature
            in the news text pushes the classifier towards a REAL or FAKE prediction.
            For more details about the raw, scaled scores of these features and
            explanations of their distributions in real vs fake training data,
            please click below or on the Visualizations tab.
        """)
        
                
        # Extract the actual feature values for the single text news prediction
        single_text_df = feature_extractor.extractFeaturesForSingleText(news_text)
        

        with col2.expander("*View More Detailed Feature Score Explanations*"):
            
            # Iterate over extra features to explain each one
            for _, row in extra_features_df.iterrows():
                # Get the raws-score/actual feature value for the news text
                actual_value = single_text_df[row["Original Feature"]].iloc[0]
                # Determine its color based on its importance value: red if pushing towards positive/fake news, else blue
                importance_color = "red" if row["Importance"] > 0 else "blue"
                # Explain the feature importance value to users
                if row["Importance"] > 0:
                    importance_explanation = "(pushing towards fake news)"
                elif row["Importance"] < 0:
                    importance_explanation = "(pushing towards real news)"
                else:
                    importance_explanation = "(neutral/has no impact)"
                
                # Add text explaining what exactly each semantic and linguistic feature means
                container.markdown(f"""
                    **{row['Feature']}**
                    - Raw Score: {actual_value:.4f}
                    - Impact on Classification: <span style='color:{importance_color}'>{row["Importance"]:.4f} {importance_explanation}</span>
                    - {FEATURE_EXPLANATIONS[row["Original Feature"]]}
                    ---
                """, unsafe_allow_html=True)
                
