import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Определение класса датасета для обучения модели
class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Кодирование сообщений с помощью токенизатора
        encoded_inputs = self.tokenizer.encode_plus(
            conversation["input"],
            conversation["output"],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_inputs["input_ids"].squeeze()
        attention_mask = encoded_inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# Определение класса модели чат-бота
class ChatBot(nn.Module):
    def __init__(self, model_name, pad_token_id):
        super(ChatBot, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        return outputs.loss

# Параметры обучения
model_name = "gpt2"
max_length = 128
batch_size = 8
num_epochs = 3
learning_rate = 1e-4

# Загрузка и предобработка данных обучения
conversations = [
    {"input": "Привет!", "output": "Привет, как я могу тебе помочь?"},
    {"input": "Какая погода сегодня?", "output": "Погода сегодня солнечная."},
    {"input": "Как тебя зовут?", "output": "Меня зовут Чат-Бот."},
    {"input": "Что ты умеешь делать?", "output": "Я могу отвечать на вопросы и поддерживать разговор."},
    # Добавьте больше диалоговых пар
]

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
dataset = ChatDataset(conversations, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Создание модели чат-бота
model = ChatBot(model_name, tokenizer.pad_token_id)

# Определение оптимизатора и функции потерь
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

# Сохранение обученной модели
output_dir = "path/to/save/model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Примеры работы модели
model.eval()

input_texts = [
    "Привет!",
    "Какая погода завтра?",
    # Добавьте больше входных фраз
]

for input_text in input_texts:
    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Generated Output: {generated_text}")
    print()

# Сохранение модели и оптимизатора
output_dir = "path/to/save/model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
torch.save(optimizer.state_dict(), output_dir + "/optimizer.pt")
