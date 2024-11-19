from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLanguageModel(ABC):
    """Abstract base class for language models"""

    @abstractmethod
    def initialize_model(self):
        """Initialize the model and tokenizer"""
        pass

    @abstractmethod
    def generate_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response based on conversation history"""
        pass


