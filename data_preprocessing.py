import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from termcolor import colored

nltk.data.path.append('C:\\Users\\tusha\\AppData\\Roaming\\nltk_data')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def print_preview(text, limit=100):
    if len(text) > limit:
        print(text[:limit] + "...")
    else:
        print(text)

def preprocess_text(text):
    text = text.lower()
    print(colored("\nStep 1: Lowercased text", 'blue'))
    print_preview(text)  

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    print(colored("\nStep 2: Removed non-alphanumeric characters", 'blue'))
    print_preview(text)  
    
    text = re.sub(r'\s+', ' ', text).strip()
    print(colored("\nStep 3: Removed extra spaces", 'blue'))
    print_preview(text) 
    
    tokens = word_tokenize(text)
    print(colored("\nStep 4: Tokenized text", 'green'))
    print_preview(" ".join(tokens), limit=100) 
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    print(colored("\nStep 5: Lemmatized tokens", 'yellow'))
    print_preview(" ".join(lemmatized_tokens), limit=100)  
    
    return lemmatized_tokens

def lemmitizeTokens(text):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in text]
    print(colored("\nLemmatized tokens", 'yellow'))
    print_preview(" ".join(lemmatized_tokens), limit=100)  
    return lemmatized_tokens
