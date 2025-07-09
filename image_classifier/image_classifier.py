import os
from PIL import Image
import requests
import traceback
from transformers import CLIPProcessor, CLIPModel

model = None
processor = None

def load_model():
    global model, processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_image(image):
    labels = ["a cat", "a dog", "a car", "a tree", "a human", "a pizza", "a keyboard", "a sunset", "a traffic light", "a bicycle", "a book", "a phone", "a chair", "a computer", "a flower", "a bird", "a plane", "a train", "a boat", "a watch", "a camera", "a guitar", "a hat", "a shoe", "a bag", "a bottle", "a glass", "a clock", "a lamp", "a mirror", "a painting", "a sculpture", "a toy", "a ball", "a drum", "a violin", "a trumpet", "a saxophone"]
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return labels[probs.argmax()], probs.max().item()

def open_image():
    try:
        print("\nChoose image source:")
        print("1. Enter file path manually")
        print("2. Enter image URL")
        choice = input("Enter 1 or 2 (or Ctrl+C to cancel): ").strip()

        if choice == "1":
            path = input("Enter the full image path (JPG/PNG): ").strip()
            image = Image.open(path)
        elif choice == "2":
            url = input("Paste image URL: ").strip()
            image = Image.open(requests.get(url, stream=True).raw)
        else:
            print("Invalid option.")
            return None

        if image.format not in ("JPEG", "PNG"):
            print("Unsupported format. Please use JPG or PNG.")
            return None

        return image

    except KeyboardInterrupt:
        print("\nReturning to image classifier menu...")
        return None
    except Exception as e:
        print("\nError loading image:")
        traceback.print_exc()
        return None

def image_classifier_main():
    clear_console = lambda: os.system('cls' if os.name == 'nt' else 'clear')
    load_model()
    
    while True:
        try:
            clear_console()
            print("======= Image Classifier =======")
            print("1. Detect object on image")
            print("2. Retrain (reload model)")
            print("3. Exit to main menu")
            choice = input("Enter choice: ").strip()

            if choice == "1":
                image = open_image()
                if image:
                    label, confidence = detect_image(image)
                    print(f"\nPrediction: {label} ({confidence:.2%} confidence)")
                    input("\nPress Enter to continue...")

            elif choice == "2":
                print("Reloading model...")
                load_model()
                print("Model loaded.")
                input("Press Enter to continue...")

            elif choice == "3":
                print("Returning to main menu...")
                break

            else:
                print("Invalid choice.")
                input("Press Enter to continue...")

        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            break
