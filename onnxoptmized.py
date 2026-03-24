from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

class ONNXChatModel:
    def __init__(self, model_id="Xenova/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = ORTModelForCausalLM.from_pretrained(
            model_id,
            provider="CPUExecutionProvider"
        )

    def invoke(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split("Answer:")[-1].strip()