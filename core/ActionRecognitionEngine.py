import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

class ActionLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=3):
        super(ActionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ActionRecognitionEngine:
    def __init__(self, model_path=None):
        self.input_size = 34
        self.num_classes = 3
        self.classes = ["Idle", "Sprinting", "Kicking"]
        self.sequence_length = 30
        self.model = ActionLSTM(self.input_size, num_classes=self.num_classes)
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.model_path = str(base_dir / "training" / "lstm" / "checkpoints" / "action_lstm_best.pth")
        else:
            self.model_path = model_path
        self.loaded = False

    def load_model(self):
        try:
            if Path(self.model_path).exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                print(f"LSTM weights loaded from {self.model_path}")
            else:
                print(f"No LSTM weights found at {self.model_path}, using random weights.")
            self.model.eval()
            self.loaded = True
        except Exception as e:
            print(f"Could not load LSTM model: {e}")

    def preprocess_keypoints(self, keypoints_sequence):
        data = np.array(keypoints_sequence).reshape(self.sequence_length, -1)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    def predict_action(self, keypoints_buffer):
        if not self.loaded:
            self.load_model()

        if len(keypoints_buffer) < self.sequence_length:
            return "Buffering..."

        input_tensor = self.preprocess_keypoints(keypoints_buffer)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            
        return self.classes[predicted.item()]
