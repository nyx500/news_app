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
        value=50,  # Default value
        step=25, # Step size of 25
        help="Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!"
    )
    
    st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to output.")
    
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
                st.error(f"Error extracting the news text: {e}. Please try a different text!")
        else:
            st.warning("Warning: Please enter some valid news text for classification!")


# Second tab: News Input directly as text
with tabs[1]:
    st.header("Paste News Text In Here Directly")
    news_text = st.text_area("Paste the news text for classification", placeholder="Paste your news text here...", height=300)
    
    # Add slider to let users select the number of perturbed samples for LIME explanations
    num_perturbed_samples = st.slider(
        "Select the number of perturbed samples to use for the explanation",
        min_value=25,
        max_value=500,
        value=50,  # Default value is 50, sweet spot between time and accuracy of explanations
        step=25,
        help="Warning: Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!"
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
        
        Main Trends:
        
        - Capital letters: Higher frequencies in real news due to using more proper nouns and techical acronyms
        - Third-person pronouns: More common in fake news, suggesting storytelling-like narrative style and person-focused content
        - Exclamation points: More frequent in fake news, indicating sensational inflammatory style
        - Emotional content: Fake news tends to have more negative emotional connotations and reduced trust scores
        
        Note: While these patterns are statistically significant in THIS dataset, they should be considered alongside other features
        (i.e. word feature importance) in the analysis, as well as remembering that more recent fake news may exhibit different 
        trends.
        
        More feature analysis coming soon!
        """)

# Fourth tab: Word Clouds
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
    st.header("🤷 How Does this Application Work?")
    
    st.write("""
    An algorithm called LIME (Local Interpretable Model-agnostic Explanations) is used in this app to explain the
    individual predictions made for a news item (i.e. whether the news text is real or fake).
    Let's explain how it works step-by-step here:
    """)
    
    st.subheader("⚙️ The Core Concept Behind LIME")
    st.write("""
    Whenever this app analyzes a news text, it doesn't just tell you if the news is "fake news" or "real news". The core purpose of
    LIME is to explain which features of the text led the model to make the outputted decision.
    As such, highlights WHICH word features, or more high-level semantic and linguistic features (such as use of certain punctuation marks)
    , in the news text led to the outputted classification. Furthermore, the algorithm also outputs a probability for the text being fake news,
    thus displaying the confidence  likely it is to be fake, rather than a simple "fake" or "real" label.
    """)
    
    st.subheader("🍋‍🟩 How Does LIME Generate the Explanations?")
    st.write("""
    LIME removes certain words or linguistic features in the news text one-by-one, and runs the trained machine-learning model to see
    how the outputted probabilities change when the text has been slightly changed.

    (a) LIME randomly removes words or linguistic features from the news input
    (b) It then runs the altered versions of the news texts through the classifier and records how much changing these individual features
    has impacted the final prediction
    (c) If changing a specific feature (e.g. emotion score) has a big impact on the final predicted probability, this feature is then assigned a higher importance
    scor. This importance is then visualized using the graphs and highlighted text
    """)
    
    st.subheader("📈 Which features have been included here?")
    st.write("""
    This model classifies news articles based on certain features which were found to be the most useful for discriminating 
    between real and fake news based on an extensive exploratory data analysis:

    - Individual words and groups of words (bigrams and trigrams) that appear more frequently in fake than real news
    - Use of punctuation (exclamation marks) and capital letters
    - Grammatical and linguistic patterns such as noun-to-verb ratio
    - Frequency of PERSON and NUMBER named entities
    - Positive emotion scores using the NRC Lexicon
    - Text readability (how complex the text is to read, e.g. how many difficult words are used based on a list from the "textstat" Python library)
    """)
    
    with st.expander("⁉️ Why Were THESE Particular Features Chosen?"):
        st.write("""
        These features were engineered based on a detailed exploratory analysis focusing on the key differences between real and fake news
        over four benchmark datasets: WELFake (general news), Constraint (COVID-19 related health news), PolitiFact (political news),
        and GossipCop (celebrity and entertainment news).
        
        - Fake news is often associated with a more sensational style (e.g. using more exclamation points) than real news, and more "clickbaity" language
        - Real news tends to use more nouns than verbs, as well as more references to numbers, signalling a more factal style
        - Narrative style (e.g. using more third-person pronouns for a more "storytelling" style) can be an indicator of fake news
        - How easy the text is to read and difficult word usage can also help the classifier distinguish between real and fake news,
        as fake news is often easier to digest and less challenging.
        """)
        
    st.subheader("😐 Disclaimer: Limitations of the Model")
    st.write("""
        Please bear in mind that patterns characterizing disinformation and fake news are rapidly evolving, particularly with the rise of generative AI.
        No single feature is sufficient to distinguish between real and fake news: it is rather a combination of the features that the classifier uses to
        make the predictions. The patterns highlighted in these explanations are based on this specific training data from four well-known fake news datasets; however,
        they may not apply to newer forms of disinformation, particularly given the rise of new technologies for disinformation production. It is recommended to
        also use fact-checking and claim-busting websites to check out whether the sources of information are legitimate.
    """)
    
