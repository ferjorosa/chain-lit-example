import torch
import chainlit
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pipeline():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device=device,
    )
    return HuggingFacePipeline(pipeline=pipe)

hf = get_pipeline()

template = """
{question}
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))