# Import required libraries
import pandas as pd
from collections import OrderedDict # For extracting best hyperparameter string representation from saved .csv file to original dtype
import numpy as np
import joblib
from tqdm import tqdm  # Monitors progress of long processing operations with a bar
import streamlit as st
from newspaper import Article
import altair as alt
import matplotlib.pyplot as plt
import joblib
import spacy
import subprocess

#@st.cache_resource
#def downloadSpacyModel():
#    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#    print("Downloaded Spacy model")
#
#downloadSpacyModel()

from lime_functions import BasicFeatureExtractor, explainPredictionWithLIME, highlightText, detectWordBoundaries, displayAnalysisResults


FEATURE_EXPLANATIONS = {
    "exclamation_point_frequency": "Normalized frequency of exclamation marks. Higher raw scores may indicate more emotional or sensational writing, more associated with fake news in the training data.",
    "third_person_pronoun_frequency": "Normalized frequency of third-person pronouns (he, she, they, etc.). Higher raw scores may indicate narrative and story-telling style. More positive scores associated with fake news than real news in training data.",
    "noun_to_verb_ratio": "Ratio of nouns to verbs. Higher values suggest more descriptive rather than action-focused writing. Higher scores (more nouns to verbs) associated more with real news than fake news in training data. Negative values more associated with fake news than real news.",
    "cardinal_named_entity_frequency": "Normalized frequency of numbers and quantities. Higher scores indicate higher level of specific details, more associated with real news. ",
    "person_named_entity_frequency": "Normalized frequency of person names. Shows how person-focused the text is, higher scores more associated with fake news.",
    "nrc_positive_emotion_score": "Measure of positive emotional content using NRC lexicon. Higher values indicate more positive tone, and more positive tone is associated more with real news than fake news.",
    "nrc_trust_emotion_score": "Measure of trust-related words using NRC lexicon. Higher values suggest more credibility-focused language, and is more associated with real news than fake news.",
    "flesch_kincaid_readability_score": "U.S. grade level required to understand the text. Higher scores indicate more complex writing, which is associated more with real news in the training data.",
    "difficult_words_readability_score": "Count of complex words. Higher values indicate more sophisticated vocabulary, associated more with real news in the training data.",
    "capital_letter_frequency": "Normalized frequency of capital letters. Higher values might indicate more emphasis and acronyms. Associated more with real news in training data"
}


# Load the pipeline
@st.cache_resource  # Cache the loaded pipeline
def load_pipeline():
    return joblib.load("iteration2_lime_model.pkl")
    
feature_extractor = BasicFeatureExtractor()
pipeline = load_pipeline()



# Title of the app
st.title("Fake News Detection App")

# Create tabs
tabs = st.tabs(["Enter News as URL", "Paste in Text Directly", "Key Pattern Visualizations",
                "Word Clouds: Real vs Fake", "How it Works..."])

# First tab: News Input as URL
with tabs[0]:
    st.header("Paste URL to News Text Here")
    url = st.text_area("Enter news URL for classification", placeholder="Paste your URL here...", height=68)
    
    # Add a slider to let users select the number of perturbed samples for LIME explanations
    num_perturbed_samples = st.slider(
        "Select the number of perturbed samples for explanation",
        min_value=25,
        max_value=500,
        value=100,  # Default value
        step=25, # Step size of 25
        help="Increasing the number of samples will make the outputted explanations more accurate but may take longer to process."
    )
    
    st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to compute.")
    
    if st.button("Classify", key="classify_button"):
        if url.strip():  # Check if input is not empty
            try:
                # Extract news text from URL using the newspaper3k library
                with st.spinner("Extracting news text from URL..."):
                    article = Article(url)
                    article.download()
                    article.parse()
                    news_text = article.text
                    
                    # Show the original text in an expander
                    with st.expander("View Original Text"):
                        st.text_area("Original News Text", news_text, height=300)  # Use a text_area to display large text
                    
                    # Generating prediction and explanations
                    with st.spinner("Analyzing text..."):
                        explanation_dict = explainPredictionWithLIME(
                            pipeline,
                            news_text,
                            feature_extractor,
                            num_perturbed_samples=num_perturbed_samples
                        )
                        
                        displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)
            
            except Exception as e:
                st.error(f"Error extracting the news text: {e}. Please try a different text.")
        else:
            st.warning("Warning: Please enter some valid news text for classification!")


# Second tab: News Input directly as text
with tabs[1]:
    st.header("Paste News Text Directly")
    news_text = st.text_area("Paste the news text for classification", placeholder="Paste your news text here...", height=300)
    
    # Add slider to let users select the number of perturbed samples for LIME explanations
    num_perturbed_samples = st.slider(
        "Select the number of perturbed samples for explanation",
        min_value=25,
        max_value=500,
        value=100,  # Default value
        step=25,
        help="Increasing the number of samples will make the outputted explanations more accurate but may take longer to process"
    )
    
    st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to compute.")
    
    if st.button("Classify", key="classify_button_text"):
        if news_text.strip():  # Check if input is not empty
            try:
                # Use the entered text as news text directly for classification
                with st.spinner(f"Analyzing text with {num_perturbed_samples} perturbed samples..."):
                    explanation_dict = explainPredictionWithLIME(
                        pipeline,
                        news_text,
                        feature_extractor,
                        num_perturbed_samples=num_perturbed_samples
                    )
                    
                    displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)
                   
                     
            except Exception as e:
                st.error(f"Error analyzing the text: {e}")
        else:
            st.warning("Warning: Please enter some valid news text for classification!")

# Third tab: Visualizations of REAL vs FAKE news patterns
with tabs[2]:
    st.header("Key Patterns in the Training Data: Real (Blue) vs Fake (Red) News")
    st.write("These visualizations show the main trends and patterns between real and fake news articles in the training data.")
    
    # Capital Letter Usage
    st.subheader("Capital Letter Usage")
    caps_img = plt.imread("all_four_datasets_capitals_bar_chart_real_vs_fake.png")
    st.image(caps_img, caption="Mean number of capital letters in real vs fake news", use_container_width=True)
    st.write("Real news tended to use more capital letters, perhaps due to including more proper nouns and technical acronyms.")
    
    # Third Person Pronoun Usage
    st.subheader("Third Person Pronoun Usage")
    pronouns_img = plt.imread("all_four_datasets_third_person_pronouns_bar_chart_real_vs_fake.png")
    st.image(pronouns_img, caption="Frequency of third-person pronouns in real vs fake news", use_container_width=True)
    st.write("Fake news often uses more third-person pronouns (e.g him, his, her), which may suggest a more 'storytelling' kind of narrative style.")

    # Exclamation Point Usage
    st.subheader("Exclamation Point Usage")
    exclaim_img = plt.imread("all_four_datasets_exclamation_points_bar_chart_real_vs_fake.png")
    st.image(exclaim_img, caption="Frequency of exclamation points in real vs fake news", use_container_width=True)
    st.write("Fake news tends to use more exclamation points, which indicates a more sensational and inflammatory writing.")
    
    # Emotion counts
    st.subheader("Emotional Content using NRC Emotion Lexicon")
    emotions_img = plt.imread("all_four_datasets_emotions_bar_chart_real_vs_fake.png")
    st.image(emotions_img, caption="Emotional content comparison between real and fake news", use_container_width=True)
    st.write("Fake news (in this dataset) often showed less positive emotion and less trust than real news.")

    # Add an expander with more detailed explanation
    with st.expander("📊 Details about these visualizations"):
        st.write("""
        These visualizations are based on analysis of our training dataset containing thousands of verified real and fake news articles
        from four benchmark fake news datasets. 
        The charts show **normalized** frequencies, meaning the counts are adjusted for article length to allow fair comparison across
        news texts of varying lengths.
        
        Summary of Trends:
        
        - Capital letters: Higher frequencies in real news due to using more proper nouns and techical acronyms
        - Third-person pronouns: More common in fake news, suggesting storytelling-like narrative style and person-focused content
        - Exclamation points: More frequent in fake news, indicating sensational inflammatory style
        - Emotional content: Fake news tends to have more negative emotional connotations and reduced trust scores
        
        Note: While these patterns are statistically significant in THIS dataset, they should be considered alongside other features
        (i.e. word feature importance) in the analysis, as well as remembering that more recent fake news may exhibit different 
        trends.
        
        More feature analysis coming soon!
        """)

# Fourth tab: Word Cloud patterns
with tabs[3]:
    st.header("Most Common Named Entities: Real vs Fake News")
    st.write("These word clouds visualize the most frequent named entities (e.g. people, organizations, countries) in real and fake news articles from our training data. The size of each word is proportional to how frequently it appears.")
    
    st.subheader("Named Entities Appearing ONLY in Real News and NOT in Fake News")
    real_cloud_img = plt.imread("combined_four_set_training_data_real_news_named_entities_wordcloud.png")
    st.image(real_cloud_img, caption="Most frequent entities in real news not in fake news", use_container_width=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Named Entities Appearing ONLY in Fake News and NOT in Real News")
    fake_cloud_img = plt.imread("combined_four_set_training_data_fake_news_named_entities_wordcloud.png")
    st.image(fake_cloud_img, caption="Most frequent entities in fake news not in real news", use_container_width=True)
   
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add an expander with methodology explanation
    with st.expander("ℹ️ A Note on these Word Clouds"):
        st.write("""
        The larger a named entitity appears, the more frequently it occurred in that category of news. 
        Colors are used only for visual distinction and don't carry specific meaning.
        """)

with tabs[4]:
    st.header("How Does this App Work?")
    
    st.write("""
    This app uses an Explainable AI (XAI) technique called LIME (Local Interpretable Model-agnostic Explanations) to explain its predictions.
    Let's break down how it works in simple stages:
    """)
    
    # Section 1
    st.subheader("🔍 The Basic Idea Behind LIME Explanations")
    st.write("""
    When the app analyzes a news text, it doesn't just give you a "real" or "fake" prediction - it explains WHY it made 
    that decision by showing which word features, or linguistic features, of the text were the most important for the classification.
    """)
    
    st.subheader("🍋‍🟩 LIME and the 'Perturbation' Process")
    st.write("""
    Imagine you're trying to understand why a chef thinks a meal he has made tastes good. He might try removing different ingredients 
    one at a time to see what affects the texture and taste the most. LIME does something similar, but with data samples:

    1. It takes the news texts and creates many different versions by randomly removing or changing the features in the text
    2. It runs the changed/perturbed text through a machine-learning classifier and evaluates how much changing the word affects
    the real vs fake news probability scores outputted by the model
    3. If changing a particular feature (such as references to people or emotion-bearing words) greatly affects the prediction, 
       that feature is considered more important for the final decision
    """)
    
    st.subheader("📊 Included Features ")
    st.write("""
    This model considers several key features in text-based news articles:

    - Writing style (e.g. capital letters, exclamation points)
    - Language patterns (third-person pronouns, noun-to-verb ratios)
    - Named entities (people, cardinal numbers)
    - Emotion words (using the NRC emotion dictionary)
    - Text complexity (readability scores)
    """)
    
    with st.expander("🤔 Why These Features?"):
        st.write("""
        These features were chosen based on extensive research and analysis into the differences between real and fake news
        over four well-known fake news datasets.
        
        - Fake news often uses more sensational, inflammatory language and exclamation points
        - Real news tends to include more specific details, such as more nouns than verbs, as well as more numbers
        - The writing style (like using more third person pronouns) can be an indicator of disinformation
        - Readability and difficult word usage can also differ between real and fake news
        
        However, fake news is constantly evolving, and no one feature is enough to understand how the content
        differs from that of real news. Rather, it's the combination of many features that helps make the determination.
        """)
    
    st.subheader("⚠️ Important Notes and Ethical Information on How to Use the App")
    st.write("""
    - The app's explanations are based on patterns found in our training data
    - These patterns might not apply to all cases or newer forms of fake news
    - The app is a tool to assist and enrich human judgment, not replace it
    - Use fact-checking and claim-busting websites to check out sources
    """)

