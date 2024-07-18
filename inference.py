from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class DialogueSystem:
    def __init__(self, checkpoint_path):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto"  # Automatically choose the best device (CPU/GPU)
        )
        self.history = []

    def generate_response(self, prompt):
        full_prompt = self._format_prompt(prompt)

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Adjust as needed
                temperature=0.7,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _format_prompt(self, prompt):
        full_prompt = ""
        for pair in self.history:
            full_prompt += f"Human: {pair[0]}\nAssistant: {pair[1]}\n"
        full_prompt += f"Human: {prompt}\nAssistant:"
        return full_prompt

    def converse(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = self.generate_response(user_input)
            print("Assistant:", response)
            self.history.append((user_input, response))

def main():
    checkpoint_path = "/content/drive/MyDrive/unsloth-test/checkpoint-60"
    dialogue_system = DialogueSystem(checkpoint_path)
    dialogue_system.converse()

if __name__ == "__main__":
    main()
