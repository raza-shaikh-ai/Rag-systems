from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


MODEL_ID = "Xenova/TinyLlama-1.1B-Chat-v1.0"

# Load
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = ORTModelForCausalLM.from_pretrained(
    MODEL_ID,
    provider="CPUExecutionProvider"
)

# Simple prompt
prompt = "User: What is AI?\nAssistant:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)