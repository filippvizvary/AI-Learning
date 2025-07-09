import os
from number_guessing.number_guessing import play_game
from spam_classifier.spam_classifier import train_spam_classifier, classify_message_loop

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
        print("3. Exit")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            play_game()
        elif choice == "2":
            train_spam_classifier()
            try:
                classify_message_loop()
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()