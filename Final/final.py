import nltk, random, string
from nltk.corpus import wordnet, stopwords
import os
from nltk.tokenize import RegexpTokenizer
from nltk.classify import NaiveBayesClassifier
from PIL import Image, ImageFilter, ImageOps

# 1) Download needed NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2) Ensure NLTK can find your data
nltk.data.path.append('/root/nltk_data')

# 3) Tokenizer + Stopwords setup
tokenizer = RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))

# Define extract_features first
def extract_features(text):
    """Extract features from text using RegexpTokenizer"""
    tokens = tokenizer.tokenize(text.lower())
    return {f"contains({t})": True for t in tokens if t not in stop_words}

# 4) Train a tiny Naïve Bayes intent classifier
train_data = [
    ("tell me a story", "story"), ("i want a story", "story"),
    ("share your personal narrative", "story"), ("tell me a joke", "joke"),
    ("make me laugh", "joke"), ("give me a witty saying", "joke"),
    ("what is the capital of france", "fact"),
    ("who is the president of the united states", "fact"),
    ("synonym for happy", "synonym"), ("what's a synonym for sad", "synonym"),
    ("exit", "exit"), ("quit", "exit")
]

# Now we can safely create training features
training_features = [(extract_features(text), label) for (text, label) in train_data]
classifier = NaiveBayesClassifier.train(training_features)

# 5) Factual Q/A
fact_dict = {
    "capital_of_france": "The capital of France is Paris.",
    "president_of_the_united_states": "The President of the United States is Joe Biden."
}
def answer_fact(q):
    q = q.lower()
    if "capital" in q and "france" in q:
        return fact_dict["capital_of_france"]
    if "president" in q and ("united states" in q or "usa" in q):
        return fact_dict["president_of_the_united_states"]
    return "I'm not sure about that. Try asking something else!"

# 6) Content Pools
story = (
    "Once upon a time, in a small mountain village, there lived a curious student "
    "who dreamed of bridging the worlds of code and storytelling..."
)
jokes = [
    "Why did the programmer quit his job? Because he didn't get arrays.",
    "I would tell you a UDP joke, but you might not get it.",
    "Debugging: Removing the needles from the haystack."
]
aphorisms = [
    "Code is like humor. When you have to explain it, it’s bad.",
    "In theory, theory and practice are the same. In practice, they’re not.",
    "Simplicity is the soul of efficiency."
]
ascii_art = r"""
   ____ _           _   _
  / ___| |__   __ _| |_| |__
 | |   | '_ \ / _` | __| '_ \
 | |___| | | | (_| | |_| | | |
  \____|_| |_|\__,_|\__|_| |_|
"""
emojis = ["🤖","🎉","🚀","💡","😊"]

def get_synonyms(w):
    lemmas = set()
    for syn in wordnet.synsets(w):
        for lm in syn.lemmas():
            name = lm.name().replace('_',' ')
            if name.lower()!=w.lower():
                lemmas.add(name)
    return list(lemmas)

# Image handling setup
PRESET_DIR = 'preset_images'
current_image = None

def list_preset_images():
    if not os.path.isdir(PRESET_DIR): return []
    return [f for f in os.listdir(PRESET_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))]

def load_custom_image(path):
    global current_image
    try:
        current_image = Image.open(path)
        return True
    except Exception as e:
        print(f"Error loading image: {e}")
        return False

def load_preset_image(filename):
    return load_custom_image(os.path.join(PRESET_DIR, filename))

def apply_filter(filter_name):
    global current_image
    if current_image is None:
        print("No image loaded.")
        return None
    if filter_name == 'edge': return current_image.filter(ImageFilter.FIND_EDGES)
    if filter_name == 'grayscale': return current_image.convert('L')
    if filter_name == 'invert': return ImageOps.invert(current_image.convert('RGB'))
    print("Unknown filter.")
    return None

def creative_effect():
    global current_image
    if current_image is None:
        print("No image loaded.")
        return None
    return current_image.filter(ImageFilter.EMBOSS)

def validate_image_path(path):
    """Validate and normalize image path"""
    # Convert relative path to absolute
    abs_path = os.path.abspath(os.path.expanduser(path))
    # Check if file exists and is an image
    if not os.path.isfile(abs_path):
        return None
    if not any(abs_path.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        return None
    return abs_path

def handle_image_command(user_input):
    """Handle image-related commands"""
    # Check if input is a direct path
    path = validate_image_path(user_input)
    if path:
        if load_custom_image(path):
            return "image"  # Return intent instead of message
    return None

# Chat loop
def chatbot():
    print(ascii_art)
    print("ChatPy 🤖: Hello! Ask for a story, joke, fact, synonym, image commands, give the location of an image, or 'exit'.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input: 
            continue

        # Try handling as image path first
        image_intent = handle_image_command(user_input)
        if image_intent:
            print(f"ChatPy 🎨: Successfully loaded image!")
            print("Available commands: 'apply filter', 'creative effect'\n")
            continue

        # Handle image-related commands
        if user_input.lower().startswith('apply filter'):
            if current_image is None:
                print("ChatPy ❌: No image loaded. Load an image first.\n")
                continue
            filt = input("Filter (edge, grayscale, invert): ").lower().strip()
            out_img = apply_filter(filt)
            if out_img:
                out_file = f"out_{filt}.png"
                out_img.save(out_file)
                print(f"ChatPy ✨: Applied {filt} filter and saved as {out_file}\n")
            continue

        # Regular intent classification
        intent = classifier.classify(extract_features(user_input))
        
        if intent == "story":
            print(f"ChatPy {random.choice(emojis)}: {story}\n")
        elif intent == "joke":
            print(f"ChatPy {random.choice(emojis)}: {random.choice(jokes)}\n")
        elif intent == "fact":
            if "france" in user_input.lower():
                print(f"ChatPy {random.choice(emojis)}: {fact_dict['capital_of_france']}\n")
            elif "president" in user_input.lower():
                print(f"ChatPy {random.choice(emojis)}: {fact_dict['president_of_the_united_states']}\n")
            else:
                print(f"ChatPy {random.choice(emojis)}: I'm not sure about that fact. Try asking about France or the US President!\n")
        elif intent == "exit":
            print(f"ChatPy {random.choice(emojis)}: Goodbye!\n")
            break
        else:
            print(f"ChatPy {random.choice(emojis)}: {random.choice(aphorisms)}\n")

if __name__ == "__main__":
    chatbot()
