from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Загрузите обученную модель и токенизатор
model = GPT2LMHeadModel.from_pretrained('models/chess_gpt_2')
tokenizer = GPT2Tokenizer.from_pretrained('models/chess_gpt_2')

# Убедитесь, что модель в режиме оценки
model.eval()

# Пример последовательности ходов, для которой нужно предсказать следующий ход
input_moves = "e4 e6"

# Токенизация входной последовательности
input_ids = tokenizer(input_moves, return_tensors='pt').input_ids

# Генерация предсказания
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=input_ids.size(1) + 1, num_return_sequences=1)

# Декодирование предсказанного хода
predicted_move = tokenizer.decode(outputs[0], skip_special_tokens=True).split()[-1]

print(f"Предсказанный следующий ход: {predicted_move}")
