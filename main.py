
import chainlit
from chainlit.cli import run_chainlit
from src.models.qwen_model import QwenModel
from src.chat.chat_manager import ChatManager

# Initialize services
SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

model = QwenModel()
chat_manager = ChatManager(model, SYSTEM_MESSAGE)

@chainlit.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return chainlit.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@chainlit.on_chat_start
async def on_chat_start():
    """Initialize/reset the conversation when a new chat starts"""
    chat_manager.reset_conversation()


@chainlit.on_message
async def main(message: chainlit.Message):
    """Handle incoming chat messages"""
    # Process the message and get response
    response = chat_manager.process_message(message.content)

    # Send the response back
    await chainlit.Message(content=response).send()


if __name__ == "__main__":

    run_chainlit(__file__)