from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch
from src.models.base_model import BaseLanguageModel


class QwenModel(BaseLanguageModel):
    def __init__(self, device: str = None):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the Qwen model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    def generate_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response using the Qwen model"""
        # Apply the chat template with full conversation history
        text = self.tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the input
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate output
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=0.7
        )

        # Remove input tokens from the output
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):]

        # Decode the output
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response