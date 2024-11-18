import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {key: val.squeeze(0) for key, val in encoding.items()}


games = open("BASE/KASPAROV05").readlines()

for i in range(len(games)):
    games[i] = games[i].rstrip() + "."

texts = games

dataset = CustomDataset(texts)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

model_path = "/Users/apple/Desktop/cource/llm_chess_model/new/fine_tuned_gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
