from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from torchvision import transforms
from PIL import Image
import os
from photo_process import PlantDiseaseRecognizer

DiseaseRecognizer = PlantDiseaseRecognizer(r'C:\Users\fedor\Desktop\Course_work\Bot\model_registry.json', device='cpu')

IMAGES_DIR = 'images'

def ensure_images_dir():
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Привет! Отправь мне картинку, и я скажу, что на ней. Для максимальной точночти необходимо отправить фото одного интерисующего листа, желательно на однородном фоне.')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    ensure_images_dir()
    filename = f"photo_{file.file_id}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)

    await file.download_to_drive(filepath)

    image = Image.open(filepath).convert('RGB')
    input_tensor = transform(image)

    result = await DiseaseRecognizer.recognize(input_tensor, topk=2)
    species = result["species_topk"][0]
    disease_topk = result["disease_topk"]
    nutrient_topk = result["nutrient_topk"]

    msg = f"🌿 *Растение*: `{species[0]}` (вероятность: {species[1]:.2%})\n"

    is_healthy_disease = disease_topk and disease_topk[0][0].lower() == "healthy"
    is_healthy_nutrient = nutrient_topk and nutrient_topk[0][0].lower() == "healthy"

    if disease_topk and not is_healthy_disease:
        msg += "\n🦠 *Возможные болезни:*\n"
        for name, prob in disease_topk:
            msg += f"• `{name}` — {prob:.2%}\n"
    else:
        msg += "\n🦠 *Болезни*: `не обнаружены`\n"

    if nutrient_topk and not is_healthy_nutrient:
        msg += "\n🥬 *Дефициты питательных веществ:*\n"
        for name, prob in nutrient_topk:
            msg += f"• `{name}` — {prob:.2%}\n"
    else:
        msg += "\n🥬 *Дефициты питательных веществ*: `не обнаружены`\n"

    await update.message.reply_text(msg, parse_mode="Markdown")

def main():
    TOKEN = ''
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    app = ApplicationBuilder().token(TOKEN).request(request).build()

    print('Бот запущен')

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()
    print('Бот остановлен')

if __name__ == '__main__':
    main()