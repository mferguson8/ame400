import nltk
nltk.download('book')

from nltk.book import *
from nltk.corpus import wordnet, swadesh, movie_reviews
import random

# ============================================================
# Exercise 1: Tutorial demonstration code
# ============================================================
print("\n========== Exercise 1: Tutorial Demonstration ==========\n")

# Display basic information for Text 1 and Text 2
print("Text 1:")
print(text1)
print("\nText 2:")
print(text2)

# Concordance and similar words for "monstrous"
print("\n--- Concordance for 'monstrous' in text1 ---")
text1.concordance("monstrous")
print("\n--- Similar words to 'monstrous' in text1 ---")
text1.similar("monstrous")
print("\n--- Similar words to 'monstrous' in text2 ---")
text2.similar("monstrous")
print("\n--- Common contexts for 'monstrous' and 'very' in text2 ---")
text2.common_contexts(["monstrous", "very"])

# ============================================================
# Exercise 2: Analyze the usage of the word "however" across texts 1 to 9
# ============================================================
print("\n========== Exercise 2: Concordance for 'however' ==========\n")
for i, txt in enumerate([text1, text2, text3, text4, text5, text6, text7, text8, text9], 1):
    print(f"\n--- Concordance for 'however' in text{i} ---")
    txt.concordance("however")
    
# ============================================================
# WordNet examples (used in tutorial)
# ============================================================
print("\n========== WordNet Examples ==========\n")
synsets_pain = wordnet.synsets("pain")
if synsets_pain:
    print("\nDefinition of the first synset for 'pain':")
    print(synsets_pain[0].definition())
    print("\nExamples of 'pain':")
    print(synsets_pain[0].examples())
else:
    print("No synsets found for 'pain'.")

synonyms = []
for syn in wordnet.synsets('pain'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print("\nSynonyms of 'pain':")
print(set(synonyms))

antonyms = []
for syn in wordnet.synsets('pain'):
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print("\nAntonyms of 'pain':")
print(set(antonyms))

# ============================================================
# Exercise 4: Translation of words using the Swadesh wordlist
# ============================================================
print("\n========== Exercise 4: Word Translation Using Swadesh ==========\n")
print("\nAvailable Swadesh file IDs:")
print(swadesh.fileids())
print("\nA sample of English words from the Swadesh list:")
print(swadesh.words('en')[:10])

# Invert the Swadesh entries to translate from English to the target languages
fr_trans = dict((v, k) for (k, v) in swadesh.entries(['fr', 'en']))
es_trans = dict((v, k) for (k, v) in swadesh.entries(['es', 'en']))
it_trans = dict((v, k) for (k, v) in swadesh.entries(['it', 'en']))
sl_trans = dict((v, k) for (k, v) in swadesh.entries(['sl', 'en']))

words_to_translate = ["mountain", "wind", "eat", "forest"]
translations = {}
for word in words_to_translate:
    translations[word] = {
        "French": fr_trans.get(word, "Not found"),
        "Spanish": es_trans.get(word, "Not found"),
        "Italian": it_trans.get(word, "Not found"),
        "Slovenian": sl_trans.get(word, "Not found")
    }

print("\nTranslations for the words:")
for word, trans in translations.items():
    print(f"{word}: {trans}")

# ============================================================
# Exercise 3: Compare differences between a pair of texts
# ============================================================
print("\n========== Exercise 3: Compare Texts ==========\n")
# Example: Compare vocabulary size of text1 and text2.
print("Vocabulary size of text1:", len(set(text1)))
print("Vocabulary size of text2:", len(set(text2)))
print("\n(Provide further analysis in your report for Exercise 3.)")

# ============================================================
# Document Classification Example using the movie_reviews corpus
# (Included as part of the tutorial demonstration)
# ============================================================
print("\n========== Document Classification Example ==========\n")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    return {f'contains({word})': (word in document_words) for word in word_features}

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("\nDocument classification accuracy on test set:")
print(nltk.classify.accuracy(classifier, test_set))
print("\nMost informative features:")
classifier.show_most_informative_features(5)

if __name__ == "__main__":
    pass
