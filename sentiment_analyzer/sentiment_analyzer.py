import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import RegexpTokenizer

# Path to the CSV file containing training data
DATA_FILE = os.path.join(os.path.dirname(__file__), "sentiment_data.csv")

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = MultinomialNB()

        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = MultinomialNB()

    def preprocess(self, text: str) -> str:
        # remove non-letters, lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # use regex tokenizer instead of word_tokenize
        tokens = self.tokenizer.tokenize(text.lower())
        stops = set(stopwords.words('english'))
        clean = [self.lemmatizer.lemmatize(t) for t in tokens if t not in stops]
        return ' '.join(clean)

    def train(self, messages, labels) -> float:
        # fit TF-IDF, train/test split, fit model, return accuracy
        X = self.vectorizer.fit_transform([self.preprocess(m) for m in messages])
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        return float(accuracy_score(y_test, preds))

    def predict(self, message: str) -> str:
        vec = self.vectorizer.transform([self.preprocess(message)])
        pred = self.model.predict(vec)[0]
        return pred  # e.g. 'positive' or 'negative'

    def partial_fit(self, message: str, label: str):
        # add to CSV, retrain on full data
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([
            df,
            pd.DataFrame({'text': [message], 'label': [label]})
        ], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        self.train(df['text'], df['label'])


def train_sentiment_analyzer():
    if not os.path.exists(DATA_FILE):
        print(f"Training data file {DATA_FILE} not found.")
        return
    df = pd.read_csv(DATA_FILE)
    global analyzer
    analyzer = SentimentAnalyzer()
    acc = analyzer.train(df['text'], df['label'])
    print("="*40)
    print(f"Sentiment analyzer trained with accuracy: {acc:.2f}")
    print("="*40)


def add_text_to_db(text: str, label: str) -> bool:
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=['text', 'label'])
    else:
        df = pd.read_csv(DATA_FILE)
    if not ((df['text'] == text).any()):
        df = pd.concat([
            df,
            pd.DataFrame({'text': [text], 'label': [label]})
        ], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return True
    return False


def classify_sentiment_loop():
    clear_console()
    if 'analyzer' not in globals():
        train_sentiment_analyzer()

    print("="*40)
    print("      Sentiment Analyzer Module")
    print("="*40)
    print("Type your text to classify (or 'exit' to quit).")
    print("If last prediction was wrong, enter 'w' to mark it opposite.")
    print("-"*40)

    last_text = None
    last_pred = None
    try:
        while True:
            inp = input("\nText: ")
            cmd = inp.strip().lower()

            if cmd == 'exit':
                print("\nExiting module.")
                input("Press Enter to continue...")
                return

            if cmd == 'w':
                if last_text and last_pred:
                    correct = 'negative' if last_pred == 'positive' else 'positive'
                    if add_text_to_db(last_text, correct):
                        print(f"Logged as {correct}. Retraining...")
                        train_sentiment_analyzer()
                    else:
                        print("Already in database.")
                else:
                    print("No previous text to correct.")
                continue

            last_text = inp
            last_pred = analyzer.predict(inp)
            emoji = '😊' if last_pred == 'positive' else '😠'
            print(f"Predicted: {last_pred} {emoji}")

    except KeyboardInterrupt:
        print("\nExiting module.")
        input("Press Enter to continue...")
