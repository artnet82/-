import telegram
from telegram.ext import Updater, MessageHandler, Filters
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка обученной модели и токенизатора
model_name = "path/to/your/model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Функция для генерации ответа на входное сообщение
def generate_response(message):
    encoded_input = tokenizer.encode_plus(
        message,
        add_special_tokens=True,
        return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Функция-обработчик входящих сообщений
def handle_message(update, context):
    message = update.message.text
    response = generate_response(message)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

# Основная функция для запуска бота
def main():
    # Здесь необходимо указать токен вашего бота Telegram
    token = "YOUR_TELEGRAM_BOT_TOKEN"

    # Создание объекта бота и обработчика сообщений
    bot = telegram.Bot(token=token)
    updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher

    # Добавление обработчика входящих сообщений
    message_handler = MessageHandler(Filters.text & (~Filters.command), handle_message)
    dispatcher.add_handler(message_handler)

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
