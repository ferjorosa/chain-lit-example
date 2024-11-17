import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import StringPromptValue


def initialize_model(model_name: str, device: torch.device):
    """
    Initializes and returns the tokenizer and model for the given model name.

    Args:
        model_name (str): The name of the pretrained model.
        device (torch.device): The device to load the model onto.

    Returns:
        tokenizer: The tokenizer associated with the model.
        model: The model loaded onto the specified device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model


def generate_text(formatted_prompt: StringPromptValue, tokenizer, model, device: torch.device) -> str:
    """
    Generates text based on a formatted prompt using the specified model and tokenizer.

    Args:
        formatted_prompt (StringPromptValue): The formatted prompt containing the user query.
        tokenizer: The tokenizer for the model.
        model: The model used for text generation.
        device (torch.device): The device on which the model is running.

    Returns:
        str: The generated text response.
    """
    # Apply the chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt.text}  # Convert to string explicitly
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


# Main execution
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer, model = initialize_model(model_name, device)

    # Create the pipeline
    template = "{question}"
    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    chain = prompt | (lambda prompt_value: generate_text(prompt_value, tokenizer, model, device)) | parser

    # Example usage
    question = "What is electroencephalography?"
    result = chain.invoke({"question": question})
    print(result)
