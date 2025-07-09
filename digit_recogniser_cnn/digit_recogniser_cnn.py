import os
import io
import numpy as np
from PIL import Image
import requests

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'mnist_cnn.h5')

class DigitRecognizerCNN:
    model: models.Model

    def __init__(self):
        # Load MNIST data for training/evaluation
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.train_data = (
            x_train.reshape(-1,28,28,1).astype('float32')/255.0,
            to_categorical(y_train, 10)
        )
        self.test_data = (
            x_test.reshape(-1,28,28,1).astype('float32')/255.0,
            to_categorical(y_test, 10)
        )

        # Load or build model
        if os.path.exists(MODEL_FILE):
            self.model = models.load_model(MODEL_FILE)
            print(f"Loaded model from {MODEL_FILE}")
        else:
            self.model = self._build_model()
            print("Built new, untrained model")

    def _build_model(self) -> models.Model:
        m = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax'),
        ])
        m.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return m

    def train(self, epochs: int = 5, batch_size: int = 128):
        x_train, y_train = self.train_data
        x_test,  y_test  = self.test_data

        print("\n=== Training CNN on MNIST ===")
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=2
        )

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {acc:.2%}")

        self.model.save(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

    def predict(self, arr: np.ndarray) -> int:
        # Accepts 28×28 array or already‐batched data
        if arr.ndim == 2:
            img = arr.reshape(1,28,28,1).astype('float32')/255.0
        else:
            img = arr.astype('float32')/255.0
        p = self.model.predict(img, verbose=0)
        return int(np.argmax(p, axis=1)[0])

def load_image(path_or_url):
    """
    Load a local or URL image, convert to 28×28 grayscale numpy array.
    Supports .png, .jpg, .jpeg, .bmp
    """
    if path_or_url.lower().startswith(('http://', 'https://')):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)

    img = img.convert('L').resize((28,28))
    return np.array(img)

def main():
    dr = DigitRecognizerCNN()

    while True:
        print("\nCommands: train | detect | exit")
        cmd = input("Enter command: ").strip().lower()

        if cmd == 'exit':
            break

        elif cmd == 'train':
            dr.train()

        elif cmd == 'detect':
            if not dr.model:
                print("Model not loaded. Run `train` first.")
                continue

            path = input("Image path or URL (.png/.jpg): ").strip()
            if not path:
                print("No path provided.")
                continue

            try:
                arr = load_image(path)
            except Exception as e:
                print(f"Failed to load image: {e}")
                continue

            digit = dr.predict(arr)
            print(f"Predicted digit: {digit}")

        else:
            print("Unknown command. Use train, detect, or exit.")

    print("Exiting Digit Recognizer Module.")
