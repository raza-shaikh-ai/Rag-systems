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
        # 🔥 FIX 1: Handle LangChain dict input
        if isinstance(prompt, dict):
            # try common keys
            prompt = prompt.get("question") or prompt.get("input") or str(prompt)

        # 🔥 FIX 2: Ensure string
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # 🔥 FIX 3: Prevent long input crash
        prompt = prompt[:1000]

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,   # safer for CPU/HF
            temperature=0.4
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🔥 FIX 4: safer split
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response.strip()