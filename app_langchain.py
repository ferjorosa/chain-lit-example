import chainlit
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import StringPromptValue

# Function to initialize the model and tokenizer
def initialize_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

# Function to generate text
def generate_text(formatted_prompt: StringPromptValue, tokenizer, model, device: torch.device) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt.text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input
    model_inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7
    )

    # Remove input tokens from the output
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):]

    # Decode the output
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

# Device setup and model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer, model = initialize_model(model_name, device)

# Chainlit's on_message function
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Handles incoming chat messages, generates responses using the model pipeline, and sends the reply back.
    """
    # Prepare prompt template and parser
    template = "{question}"
    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    # Build pipeline
    chain = prompt | (lambda prompt_value: generate_text(prompt_value, tokenizer, model, device)) | parser

    # Process user input
    question = message.content
    result = chain.invoke({"question": question})

    # Send the generated response back
    await chainlit.Message(
        content=f"{result}"
    ).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)