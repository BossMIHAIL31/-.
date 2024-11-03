from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image, UnidentifiedImageError
from telebot import TeleBot, types
import io

# Загрузка модели
model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name).to("cpu")
model.config.pad_token_id = model.config.eos_token_id 

# Промпт для определения типа кухни
prompt = """
Analyze the image and determine the type of cuisine associated with the restaurant or dining establishment.
Consider decor, colors, furniture style, table settings, and visible dishes.
Examples: Traditional Japanese decor and low tables may indicate Japanese cuisine; ornate patterns and dumplings may suggest Chinese cuisine.
Describe the cuisine type based on these elements.
"""

# Функция для определения типа кухни
def get_cuisine_type(image, prompt):
    try:

        image = image.convert("RGB").resize((384, 384))

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        print("Input IDs shape:", inputs["input_ids"].shape)
        print("Pixel values shape:", inputs["pixel_values"].shape)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            eos_token_id=model.config.eos_token_id
        )

        
        cuisine_description = processor.batch_decode(outputs, skip_special_tokens=True)

        return cuisine_description[0] if cuisine_description else "Не удалось определить кухню."
    except Exception as e:
        print(f"Ошибка в get_cuisine_type: {e}")
        return "Не удалось определить тип кухни."

# Настройка и запуск Telegram-бота
def start_bot():
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Замените на токен вашего бота от BotFather
    bot = TeleBot(TOKEN)

    # Обработка изображений от пользователя
    @bot.message_handler(content_types=['photo'])
    def handle_image(message: types.Message):
        try:
            # Получаем файл изображения
            file_info = bot.get_file(message.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            # Открываем изображение с помощью PIL
            image = Image.open(io.BytesIO(downloaded_file))

            print(f"Изображение загружено: {image.size}, формат: {image.format}")

             # Получаем описание типа кухни
            description = get_cuisine_type(image, prompt)
            bot.reply_to(message, description)

        except UnidentifiedImageError:
            bot.reply_to(message, "Не удалось распознать изображение. Убедитесь, что это действительное изображение.")
        except Exception as e:
            bot.reply_to(message, f"Ошибка при обработке изображения: {e}")

    print("Бот запущен. Отправьте изображение ресторана для анализа кухни.")
    bot.polling()

if __name__ == "__main__":
    start_bot()


