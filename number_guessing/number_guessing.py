import os

# Clears the console screen for better readability.
# Works on both Windows and Unix-based systems.
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Main function to run the number guessing game.
# The user thinks of a number between 1 and 100, and the program tries to guess it.
# The user provides feedback after each guess: high, low, or correct.
def play_game():
    clear_console()
    print("=" * 40)
    print("      Welcome to the Number Guesser!")
    print("=" * 40)
    print("Think of a number between 1 and 100, and I'll try to guess it!")
    print("After each guess, tell me if my guess is (h)igh, (l)ow, or (c)orrect.")
    print("Type 'exit' at any time to return to the main menu.")
    print("-" * 40)
    input("Press Enter when you're ready...")

    lower = 1      # Lower bound of the guessing range
    upper = 100    # Upper bound of the guessing range
    attempts = 0   # Number of guesses made
    guesses = []   # List to store all guesses for display

    try:
        while True:
            # Make a guess using binary search strategy
            guess = (lower + upper) // 2
            attempts += 1
            guesses.append(guess)
            print(f"\nAttempt #{attempts}: My guess is \033[1m{guess}\033[0m")
            feedback = input("Is it (h)igh, (l)ow, (c)orrect, or type 'exit' to return to menu? ").strip().lower()

            if feedback == 'c':
                # Correct guess: show summary and return to menu
                print("\n" + "=" * 40)
                print(f"🎉 I guessed your number in {attempts} attempts!")
                print("Here are my guesses:")
                print(" -> ".join(str(g) for g in guesses))
                print("=" * 40)
                input("Press Enter to return to the main menu...")
                return
            elif feedback == 'h':
                # Guess was too high: adjust upper bound
                upper = guess - 1
            elif feedback == 'l':
                # Guess was too low: adjust lower bound
                lower = guess + 1
            elif feedback == 'exit':
                # User wants to exit to main menu
                print("\nReturning to main menu.")
                return
            else:
                # Invalid input: prompt again
                print("Please enter 'h' for high, 'l' for low, 'c' for correct, or 'exit' to return to menu.")

            # If bounds are invalid, user may have made a mistake
            if lower > upper:
                print("\nHmm, something doesn't add up. Did you change your number?")
                input("Press Enter to return to the main menu...")
                return
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nReturning to main menu.")
        return

if __name__ == "__main__":
    play_game()