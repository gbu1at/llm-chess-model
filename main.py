from transformers import AutoModelForCausalLM, AutoTokenizer


print()
print("-----------------------")
print("START")
print("-----------------------")
print()


model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



prompt = "Hello!"


input_ids = tokenizer(prompt, return_tensors="pt").input_ids


gen_tokens = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)


gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

print(gen_text)


print()
print("-----------------------")
print("FINISH")
print("-----------------------")
print()