import os
import nltk

def ensure_nltk_data():
    """Ensure NLTK data is downloaded to the correct location"""
    nltk_data_dir = '/home/mferguson8/nltk_data'
    
    # Create directory if it doesn't exist
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Set NLTK data path
    os.environ['NLTK_DATA'] = nltk_data_dir
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Download required NLTK data
    required_packages = ['punkt', 'wordnet', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir)

# Call this function before importing other NLTK components
ensure_nltk_data()

import random
from nltk.corpus import wordnet, stopwords
from nltk.classify import NaiveBayesClassifier

########################################
# Functionality: Storytelling, Jokes, and Factual Answers
########################################

def tell_story():
    """Randomly selects and returns a story from a predefined list."""
    stories = [
        "Once upon a time, in a faraway land, there was a curious explorer who discovered a secret garden filled with wonders.",
        "In a small village, a young dreamer crafted incredible inventions that made life a bit more magical for everyone.",
        "Long ago, a solitary wanderer embarked on a journey to find the meaning of happiness, meeting friends along the way."
    ]
    return random.choice(stories)

def tell_joke():
    """Randomly selects and returns a joke from a predefined list."""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my computer I needed a break, and now it won’t stop sending me KitKat ads.",
        "Why did the programmer quit his job? Because he didn't get arrays."
    ]
    return random.choice(jokes)

def answer_factually(question):
    """Returns a factual answer from a predefined dictionary based on the user's question."""
    facts = {
        "what is the capital of france": "The capital of France is Paris.",
        "who wrote hamlet": "Hamlet was written by William Shakespeare.",
        "what is the speed of light": "The speed of light is approximately 299,792 kilometers per second."
    }
    key = question.lower().strip(' ?')
    return facts.get(key, "I'm not sure about that. Could you ask something else or phrase it differently?")

########################################
# NLTK Integration: Synonyms and Lexical Resources
########################################

def get_synonyms(word):
    """
    Return a list of synonyms for the given word using NLTK's WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def respond_with_synonyms(query):
    """
    Return a response containing synonyms for key words in the query.
    """
    tokens = nltk.word_tokenize(query)
    response = "Here are some synonyms for key words in your query:\n"
    for token in tokens:
        syns = get_synonyms(token)
        if syns:
            response += f"- {token}: {', '.join(syns[:5])}\n"
    return response

########################################
# Naïve Bayes Classifier for Text Classification
########################################

def extract_features(text):
    """
    Convert input text into a dictionary of features, excluding stopwords.
    """
    words = nltk.word_tokenize(text.lower())
    features = {}
    for word in words:
        if word not in stopwords.words('english'):
            features[word] = True
    return features

# Sample training data for the classifier.
train_data = [
    ("hello there", "greeting"),
    ("hi, how are you", "greeting"),
    ("good morning", "greeting"),
    ("tell me a joke", "joke"),
    ("make me laugh", "joke"),
    ("i need a funny story", "joke"),
    ("what is the capital of france", "fact"),
    ("who wrote hamlet", "fact"),
    ("how fast is the speed of light", "fact"),
]

# Declare the classifier variable without initializing it yet.
nb_classifier = None

def classify_input(user_input):
    """
    Use the Naïve Bayes classifier to classify the user's input into an intent.
    """
    features = extract_features(user_input)
    return nb_classifier.classify(features)

def generate_response(user_input):
    """
    Generate a response based on the classified intent of the user input.
    """
    intent = classify_input(user_input)
    if intent == "greeting":
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! Ready to chat?"
        ]
        return random.choice(responses)
    elif intent == "joke":
        return tell_joke()
    elif intent == "fact":
        return answer_factually(user_input)
    else:
        # Fallback response if classification is uncertain.
        return "I'm not sure what you're asking. Can you please elaborate?"

########################################
# Extra Flair: Personal Touches
########################################

def share_fun_fact():
    """Return a random fun fact."""
    fun_facts = [
        "Did you know that honey never spoils?",
        "Bananas are berries, but strawberries aren't.",
        "Oxford University is older than the Aztec Empire."
    ]
    return random.choice(fun_facts)

def personalized_response(user_input):
    """
    Return a personalized response based on keywords in the input or using classifier intent.
    """
    lower_input = user_input.lower()
    if "fact" in lower_input:
        return answer_factually(user_input)
    elif "synonym" in lower_input:
        tokens = user_input.split()
        if len(tokens) > 1:
            return f"Synonyms for '{tokens[1]}': " + ", ".join(get_synonyms(tokens[1])[:5])
        else:
            return "Please specify a word to find synonyms for."
    elif "fun fact" in lower_input:
        return share_fun_fact()
    else:
        # Fallback to classifier-based response.
        return generate_response(user_input)

########################################
# Main Chatbot Interactive Loop
########################################

def chatbot():
    """
    Runs an interactive chatbot loop in the terminal.
    """
    print("Welcome to the Multimodal Chatbot! (Type 'exit' to quit.)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = personalized_response(user_input)
        print("Chatbot:", response)

########################################
# Entry Point
########################################

if __name__ == "__main__":
    # Now initialize training features and train the classifier.
    training_features = [(extract_features(text), label) for (text, label) in train_data]
    nb_classifier = NaiveBayesClassifier.train(training_features)

    print("Most Informative Features:")
    try:
        nb_classifier.show_most_informative_features(5)
    except AttributeError:
        print("Unable to display most informative features. Ensure binary features are used.")

    chatbot()
