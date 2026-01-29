from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import mediapipe as mp
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn

app = Flask(__name__)

# --------------------
# MODEL DEFINITIONS
# --------------------
class SignLSTM(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=256, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SignLSTM(num_classes=10).to(device)
model.load_state_dict(torch.load("base_pose_model.pth", map_location=device))
model.eval()

# LLM
tokenizer = T5Tokenizer.from_pretrained("t5-small")
llm_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Label map
label_map = {
    0: "hello",
    1: "thank you",
    2: "yes",
    3: "no"
}

# --------------------
# POSE EXTRACTION
# --------------------
mp_holistic = mp.solutions.holistic

def extract_pose(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened() and len(sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            pose = np.zeros(33 * 4)
            if results.pose_landmarks:
                pose = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]).flatten()

            sequence.append(pose)

    cap.release()
    return np.array(sequence)

# --------------------
# LLM REFINEMENT
# --------------------
def refine_with_llm(raw_text):
    prompt = f"Improve grammar and make sentence natural: {raw_text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------
# API ENDPOINT
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    video = request.files["video"]
    video_path = "uploaded.mp4"
    video.save(video_path)

    X_np = extract_pose(video_path)
    X = torch.tensor(X_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(1)

    raw_text = label_map.get(pred.item(), "unknown")
    refined_text = refine_with_llm(raw_text)

    return jsonify({
        "raw_text": raw_text,
        "refined_text": refined_text,
        "confidence": float(confidence.item())
    })

if __name__ == "__main__":
    app.run(debug=True)
