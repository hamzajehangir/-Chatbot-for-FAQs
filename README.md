# -Chatbot-for-FAQs
Create a chatbot that can answer frequently asked  questions (FAQs) about a particular topic or product.  Use natural language processing (NLP) techniques and  pre-built libraries like NLTK or SpaCy to understand and  generate responses.
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import random
import string

# Load the FAQ data
faqs = [
    {"question": "What is the product?", "answer": "Our product is a revolutionary new gadget that combines AI technology with user-centric design.", "category": "product_info"},
    {"question": "How does it work?", "answer": "It uses advanced AI algorithms to learn your habits and preferences, making your life easier and more convenient.", "category": "product_info"},
    {"question": "Is it compatible with my phone?", "answer": "Yes, it is compatible with most smartphones, including iOS and Android devices.", "category": "compatibility"},
    {"question": "How much does it cost?", "answer": "The product costs $99.99, with a special introductory offer of $79.99 for early adopters.", "category": "pricing"},
    {"question": "What are the benefits of using this product?", "answer": "Our product offers a range of benefits, including increased productivity, improved organization, and enhanced user experience.", "category": "benefits"},
    {"question": "Is it secure?", "answer": "Yes, our product uses state-of-the-art encryption and secure servers to protect your data and ensure your privacy.", "category": "security"},
    # Add more FAQs here
]

# Preprocess the FAQ data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return '.join(tokens)

faq_questions = [preprocess_text(faq["question"]) for faq in faqs]
faq_answers = [faq["answer"] for faq in faqs]
faq_categories = [faq["category"] for faq in faqs]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

# Train a Naive Bayes classifier for category classification
X_train, X_test, y_train, y_test = train_test_split(faq_vectors, faq_categories, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Define a function to generate responses
def generate_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_match_index = similarities.argmax()
    category = nb_classifier.predict(user_vector)[0]
    response = faq_answers[best_match_index]
    return response, category

# Create a chatbot interface
def chatbot():
    print("Welcome to our chatbot! Ask us a question about our product.")
    while True:
        user_input = input("You: ")
        response, category = generate_response(user_input)
        print("Chatbot:", response)
        if category == "product_info":
            print("Would you like to know more about our product?")
        elif category == "compatibility":
            print("Check out our compatibility page for more information.")
        elif category == "pricing":
            print("We have a special offer for early adopters. Would you like to know more?")
        elif category == "benefits":
            print("Our product offers many benefits. Would you like to learn more?")
        elif category == "security":
            print("We take security seriously. Would you like to know more about our security features?")

# Run the chatbot
chatbot()
