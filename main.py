from cv2 import FileStorage
import numpy as np
import torch
from PIL import Image
import io
import pytesseract
from flask import Flask, request, jsonify
import asyncio
from aiohttp import ClientSession


model_path = "model/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = model.autoshape()
loop = asyncio.get_event_loop()


async def async_main_requests(url, data):
    async with ClientSession() as session:
        async with session.post(url, json=data) as resp:
            print(resp.status)
            # print(await resp.json())


def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    return data


def predict(x):
    results = model(x)
    df = results.pandas().xyxy[0]
    total = len(df.index)
    return df, total


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print(request.files.get('roundId'))
        gameId = request.form.get('gameId')
        print("Game Id: ", gameId)
        roundId = request.files.get('roundId')
        file = request.files.get('card')
        # if file is None or file.filename == "":
        #     return jsonify({"error": "no file"})
        try:
            roundId_bytes = roundId.read()
            pillow_img = Image.open(io.BytesIO(roundId_bytes))
            value = transform_image(pillow_img)
            print(type(value))
            pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
            current_roundId = pytesseract.image_to_string(
                value, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            print(current_roundId.strip())
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            prediction, total = predict(tensor)

            try:
                print(prediction.to_json())
                # loop = asyncio.new_event_loop()
                loop.run_until_complete(async_main_requests(
                    "http://192.168.1.23:8001/game/" + str(gameId), {"roundId": current_roundId.strip(), "total": total, "card": prediction.to_json()}))
                # res = requests.post(
                #     "https://casino.odinflux.com/game/" + str(gameId), json={"roundId": current_roundId.strip(), "card": prediction.to_json()})
                # print("Response code: {}".format(res.status_code))
            except Exception as e:
                print(e)
                print("Error handhled in main website posting")
            return jsonify({"roundId": current_roundId.strip(), "card": prediction.to_json()})
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
