from io import BytesIO
import json
import numpy as np
import requests
import csv
import threading
from flask import Flask, request

from torchvision import transforms
from PIL import Image
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

app = Flask(__name__)
PORT = 8443

@app.route('/webhook', methods=['POST'])
def webhook():
    update = request.get_json()
    print(update)
    # Do something with the update
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

# Replace with your own API token and model endpoint
API_TOKEN = "5556533102:AAEkjTGaUsx0HaqCZjoUYmAe-s0MkvToowU"
MODEL_ENDPOINT = "http://51.250.106.202:8002/invocations"
LABELS = []

def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data))
    #image = Image.open("sysmon.png")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformed_image = transform(image)
    # Convert the transformed image to a NumPy array
    image_array = np.array(transformed_image, dtype=np.double)

    # Convert the NumPy array to a JSON-serializable object
    json_object = image_array.tolist()
    return json_object


def process_image(image_data):
    headers = {
        "format": "pandas-split",
        "Content-Type": "application/json",
    }
    transformed_image = preprocess_image(image_data)
    data = {"inputs": [transformed_image]}
    response = requests.post(MODEL_ENDPOINT, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        # There was an error, so return an empty list
        return 'Извините, наша нейронка сейчас не отвечает :^( \n Наш девопс уже метнулся кабанчиком её поднимать'
    response_json = json.loads(response.content)
    prediction = response_json['predictions']
    prediction = softmax(prediction)
    prediction = prediction[0]
    label_probs = list(zip(LABELS[0], prediction))
    sorted_label_probs = sorted(label_probs, key=lambda x: x[1], reverse=True)
    top5_label_probs = sorted_label_probs[:5]
    response_string = 'Top 5 labels:\n'
    for label, prob in top5_label_probs:
        response_string += f'\'{label}\' with a probability of {prob * 100:.2f}%\n'
    return response_string


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def handle_image(update, context):
    image_data = update.message.photo[-1].get_file().download_as_bytearray()
    prediction = process_image(image_data)
    update.message.reply_text(prediction)

def handle_no_image(update, context):
    update.message.reply_text("I'm sorry, but the message you sent did not contain a picture.")

def process_response(response):
    response_json = json.loads(response)
    prediction = response_json['predictions']
    prediction = prediction[0]
    print(prediction)
    max_index = prediction.index(max(prediction))
    label_probs = list(zip(LABELS[0], prediction))
    sorted_label_probs = sorted(label_probs, key=lambda x: x[1], reverse=True)
    top5_label_probs = sorted_label_probs[:5]
    response_string = 'Top 5 labels:\n'
    for label, prob in top5_label_probs:
        response_string += f'\'{label}\' with a probability of {prob * 100:.2f}%\n'

    # s = f'\'{LABELS[0][max_index]}\' with a probability of {int(max(prediction)*100)}%'
    return response_string

def start(update, context):
    update.message.reply_text("Hi there! To use this bot, send it an image and it will return the prediction made by the machine learning model.")

def error(update, context):
    update.message.reply_text("Извините, наша нейронка сейчас не отвечает :^( \n Наш девопс уже метнулся кабанчиком её поднимать")
    """Log Errors caused by Updates."""
    print(f"Update {update} caused error {context.error}")

def start_thread(target):
    thread = threading.Thread(target=target)
    thread.start()

def main():
    with open('labels_list.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            LABELS.append(row)

    # Create the Updater and pass it your API token.
    updater = Updater(API_TOKEN, use_context=True)

    updater.start_webhook(listen="0.0.0.0",
                      port=int(PORT),
                      url_path=API_TOKEN)
    updater.bot.setWebhook(url=f'https://84.201.165.229/webhook')

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # On incoming messages, check if they contain an image and call handle_image if they do
    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    dp.add_handler(MessageHandler(~Filters.photo, handle_no_image))
    dp.add_handler(CommandHandler("start", start))
    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(PORT))
    main()