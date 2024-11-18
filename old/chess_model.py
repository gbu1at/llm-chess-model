from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from tokenizetion import tokenize_board_function

print("\n\nSTART\n\n")

chess_file_path = "../new/BASE/KASPAROV05"

with open(chess_file_path, 'r') as f:
    lines = f.readlines()



lines = lines[:1]


games = [line.strip() for line in lines if line.strip()]
data = []


for game in games:
    moves = game.split()
    for i in range(1, len(moves)):
        input_sequence = ' '.join(moves[:i])
        target_move = ' '.join(moves[:i + 1])
        data.append({"input": input_sequence, "target": target_move})

df = pd.DataFrame(data)


dataset = Dataset.from_pandas(df)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token


# def tokenize_function(examples):
#     inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
#     targets = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
#
#     print()
#     print()
#     print(inputs)
#     print()
#     print()
#     print()
#     print()
#     print(targets)
#     print()
#     print()
#
#     inputs["labels"] = targets["input_ids"]
#
#     return {key: val.squeeze() for key, val in inputs.items()}

tokenized_dataset = dataset.map(tokenize_board_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)


trainer.train()

model.save_pretrained('./chess_gpt2')
tokenizer.save_pretrained('./chess_gpt2')

print("\n\nFINISH\n\n")