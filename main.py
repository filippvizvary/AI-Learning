import os
import numpy as np

from number_guessing.number_guessing import play_game
from digit_recogniser_cnn.digit_recogniser_cnn import main as digit_cnn_main
from spam_classifier.spam_classifier import (
    train_spam_classifier,
    classify_message_loop
)
from sentiment_analyzer.sentiment_analyzer import (
    train_sentiment_analyzer,
    classify_sentiment_loop
)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_console()
        print("======================================")
        print(" Welcome to the AI Learning Environment")
        print("======================================\n")
        print("Which module would you like to run?")
        print("1. Number Guessing Game")
        print("2. Spam Classifier")
        print("3. Sentiment Analyzer")
        print("4. Handwritten Digit Recognizer (CNN)")
        choice = input("Enter 1, 2, 3, 4 or exit: ").strip()

        if choice == "1":
            play_game()

        elif choice == "2":
            train_spam_classifier()
            try:
                classify_message_loop()
            except KeyboardInterrupt:
                print("\nReturning to main menu.")

        elif choice == "3":
            train_sentiment_analyzer()
            try:
                classify_sentiment_loop()
            except KeyboardInterrupt:
                print("\nReturning to main menu.")

        elif choice == "4":
            digit_cnn_main()   # now runs train/detect menu inside the module

        elif choice.lower() == "exit":
            print("Exiting the AI Learning Environment. Goodbye!")
            break

        else:
            print("Invalid choice. Please select from available modules.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
