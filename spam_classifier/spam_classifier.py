# Contents of /AI-Learning/AI-Learning/spam_classifier/spam_classifier.py

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path to the CSV file containing training data
DATA_FILE = os.path.join(os.path.dirname(__file__), "spam_data.csv")

def clear_console():
    # Clears the console screen for better readability.
    # Works on both Windows and Unix-based systems.
    os.system('cls' if os.name == 'nt' else 'clear')

class SpamClassifier:
    # A simple spam classifier using CountVectorizer and Multinomial Naive Bayes.
    def __init__(self):
        # Initialize the vectorizer and the model
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, messages, labels):
        # Trains the classifier on the provided messages and labels.
        # Splits the data into training and test sets, fits the model, and returns accuracy.
        X = self.vectorizer.fit_transform(messages)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def predict(self, message):
        # Predicts whether a single message is spam or ham.
        transformed_message = self.vectorizer.transform([message])
        prediction = self.model.predict(transformed_message)
        return prediction[0]

    def partial_fit(self, message, label):
        # Adds a new message and label to the dataset and retrains the model.
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame({'message': [message], 'label': [label]})], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        self.train(df['message'], df['label'])

def train_spam_classifier():
    # Loads the training data from CSV, trains the classifier, and prints the accuracy.
    if not os.path.exists(DATA_FILE):
        print(f"Training data file {DATA_FILE} not found.")
        return
    df = pd.read_csv(DATA_FILE)
    global classifier
    classifier = SpamClassifier()
    accuracy = classifier.train(df['message'], df['label'])
    print("=" * 40)
    print(f"Spam classifier trained with accuracy: {accuracy:.2f}")
    print("=" * 40)

def add_message_to_db(message, label):
    # Adds a new message and its label to the CSV database if not already present.
    # Returns True if added, False if already present.
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=['message', 'label'])
    else:
        df = pd.read_csv(DATA_FILE)
    # Only add if not already present
    if not ((df['message'] == message).any()):
        df = pd.concat([df, pd.DataFrame({'message': [message], 'label': [label]})], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return True
    return False

def classify_message_loop():
    # Main loop for classifying messages.
    # - Prompts the user for messages to classify.
    # - Displays the classifier's prediction.
    # - If the previous classification was wrong, user can type 'w' to log the opposite label.
    # - Typing 'exit' or pressing Ctrl+C returns to the main menu.
    clear_console()
    global classifier
    if 'classifier' not in globals():
        train_spam_classifier()
    print("=" * 40)
    print("         Spam Classifier Module")
    print("=" * 40)
    print("Type your message to classify (type 'exit' to return to main menu):")
    print("If the previous classification was wrong, type 'w' as your next input to mark it as wrong (it will be logged as the opposite label).")
    print("-" * 40)
    last_message = None
    last_prediction = None
    try:
        while True:
            user_input = input("\nMessage: ")
            if user_input.strip().lower() == "exit":
                # Exit to main menu
                print("\nReturning to main menu.")
                input("Press Enter to continue...")
                return

            if user_input.strip().lower() == "w":
                # Mark previous message as the opposite label and retrain
                if last_message is not None and last_prediction is not None:
                    correct_label = "spam" if last_prediction == "ham" else "ham"
                    if add_message_to_db(last_message, correct_label):
                        print(f"Message logged as '\033[1m{correct_label}\033[0m' and classifier updated.")
                        train_spam_classifier()
                    else:
                        print("Message already in database.")
                else:
                    print("No previous message to mark as wrong.")
                continue

            # Classify the new message and store for possible correction
            last_message = user_input
            last_prediction = classifier.predict(user_input)
            print(f"Classified as: \033[1m{last_prediction}\033[0m")
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nReturning to main menu.")
        input("Press Enter to continue...")