from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
import random
import tempfile

app = Flask(__name__)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
model = AutoModelForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")

def is_real_image(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    label = model.config.id2label[predicted_label]
    print
    return label == "Real"

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frames = sorted(random.sample(range(frame_count), 5))
    real_count = 0
    
    for frame_idx in random_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if is_real_image(image):
            real_count += 1
        else:
            cap.release()
            return False

    cap.release()
    return real_count == 5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file)
        if is_real_image(image):
            return jsonify({"label": "Real"})
        else:
            return jsonify({"label": "Fake"})
    
    elif file.filename.endswith(('.mp4', '.avi', '.mov')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file.read())
            temp_video_path = temp_video.name
        
        is_real = process_video(temp_video_path)
        temp_video.close()
        
        if is_real:
            return jsonify({"label": "Real"})
        else:
            return jsonify({"label": "Fake"})
    else:
        return jsonify({"error": "Unsupported file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
