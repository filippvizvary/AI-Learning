import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

from digit_recogniser_cnn.digit_recogniser_cnn import DigitRecognizerCNN

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Initialize and train the CNN
    clear_console()
    dr = DigitRecognizerCNN()
    dr.train()        # default epochs=5, batch_size=128
    dr.evaluate()

    # Interactive image‐picker loop
    try:
        while True:
            print("\nSelect an image file of a handwritten digit...")
            root = tk.Tk()
            root.withdraw()  # hide main window
            filepath = filedialog.askopenfilename(
                title="Choose digit image",
                filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp")]
            )
            root.destroy()

            if not filepath:
                print("No file chosen. Exiting scanner.")
                break

            # Load, convert, resize, predict
            img = Image.open(filepath).convert("L").resize((28, 28))
            arr = np.array(img)
            prediction = dr.predict(arr)
            print(f"Predicted digit: {prediction}")

            again = input("Classify another image? (y/n): ").strip().lower()
            if again != 'y':
                break

    except KeyboardInterrupt:
        print("\nScanner interrupted. Returning to launcher.")

if __name__ == "__main__":
    main()
