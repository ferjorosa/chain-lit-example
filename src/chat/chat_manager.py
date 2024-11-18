from typing import List, Dict
from dataclasses import dataclass, field
from src.models.base_model import BaseLanguageModel


@dataclass
class ChatManager:
    """Manages conversation state and handles interaction with the language model"""
    model: BaseLanguageModel
    system_message: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize conversation with system message"""
        self.reset_conversation()

    def reset_conversation(self):
        """Reset conversation to initial state"""
        self.messages = [
            {"role": "system", "content": self.system_message}
        ]

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the model's response"""
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Generate response
        response = self.model.generate_response(self.messages)

        # Add response to history
        self.messages.append({"role": "assistant", "content": response})

        return response