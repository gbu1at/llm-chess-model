import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "fine_tuned_gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        gen_tokens = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.7)

    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return gen_text


if __name__ == "__main__":
    prompt = "g3 b7 Bg2 Bb7 Bxb7"
    generated_text = generate_text(prompt)
    print("Сгенерированный текст:")
    print(generated_text)
