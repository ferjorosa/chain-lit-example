from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply the chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize a single input
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate tokens
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Remove input tokens from the output
output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):]

# Decode the generated text
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print(response)
