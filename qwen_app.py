import torch
import chainlit
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Load GPT-2 model and tokenizer from Hugging Face.
    Returns tokenizer, model, and config dictionary.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Create config dict to maintain compatibility with existing code
    config = model.config.to_dict()

    return tokenizer, model, config


# Load the model and tokenizer
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function for handling chat messages.
    """

    prompt = f"{message.content}"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    prepared_text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Encode the input text
    inputs = tokenizer(prepared_text, return_tensors="pt").to(device)

    # Generate tokens
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512
    )

    # Remove input tokens from the output
    output_ids = generated_ids[0][len(inputs["input_ids"][0]):]

    # Decode the generated text
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    await chainlit.Message(
        content=f"{response}",
    ).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)