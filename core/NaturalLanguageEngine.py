from transformers import pipeline
import torch

class NaturalLanguageEngine:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.pipeline = None
        self.use_mock = True

    def load_model(self):
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline("text-generation", model=self.model_name, device=device)
            self.use_mock = False
            print("NLP Model loaded successfully.")
        except Exception:
            print("Falling back to mock responses.")
            self.use_mock = True

    def generate_response(self, prompt, max_length=50):
        if self.use_mock:
            return f"Mock response to: {prompt}"
            
        if not self.pipeline:
            self.load_model()
            
        if self.use_mock:
            return f"Mock response to: {prompt}"

        try:
            response = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception:
            return "Error generating response."

    def analyze_event(self, event_description):
        prompt = f"Analyze this sports event: {event_description}. Insight:"
        return self.generate_response(prompt)
