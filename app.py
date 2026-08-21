import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Config ---
TFLITE_PATH = "model.tflite"
LABEL_PATH = "label.txt"
IMG_SIZE = (150, 150)  # must match training input size

# --- Load labels ---
with open(LABEL_PATH, "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]

# --- Load TF-Lite model ---
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)  # MobileNetV2-specific normalization -> range [-1, 1]
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(image: Image.Image):
    if image is None:
        return {}
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    return {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a corn leaf photo"),
    outputs=gr.Label(num_top_classes=len(class_labels), label="Prediction"),
    title="Corn Leaf Disease Classifier",
    description=(
        "Upload a photo of a corn leaf to classify its condition. "
        "Model: MobileNetV2 (transfer learning, fine-tuned) — 95.22% test accuracy. "
        "Classes: Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy."
    ),
    examples=None,  # optionally add example image paths here, e.g. ["examples/rust1.jpg"]
)

if __name__ == "__main__":
    demo.launch()
