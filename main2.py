from asyncio import threads
from cv2 import FileStorage
import numpy as np
import torch
from PIL import Image
import io
import pytesseract
from flask import Flask, request, jsonify
import asyncio
# from aiohttp import ClientSession
import asyncio
import aiohttp
import threading
import requests
from sanic import Sanic
from sanic.response import json

model_path = "model/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = model.autoshape()
loop = asyncio.get_event_loop()


async def async_main_requests(url, data):
    async with ClientSession() as session:
        async with session.post(url, json=data) as resp:
            print(resp.status)
            print(await resp.json())


def transform_image(pillow_image):
    # print(type(pillow_image))
    # newSize = (640, 640)
    # image = pillow_image.resize(newSize)
    # # image.save("640x640.png")
    # pillow_image.save("pillow.png")
    data = np.asarray(pillow_image)
    print(data.shape)
    return data


async def thread_request(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            # res = requests.post(url, json=data)
            print(response.status)
    # print(res.text)



def predict(x):
    results = model(x)
    df = results.pandas().xyxy[0]
    total = len(df.index)
    return df, total


app = Sanic(__name__)


@app.route("/", methods=["GET", "POST"])
async def index(request):
    # await asyncio.sleep(0.1)
    if request.method == "POST":
        # print(request.files)
        # print(request.files.get('roundId'))
        gameId = request.form.get('gameId')
        print("Game Id: ", gameId)
        roundId = request.files.get('roundId')
        file = request.files.get('card')
        # if file is None or file.filename == "":
        #     return jsonify({"error": "no file"})
        try:
            # roundId_bytes = roundId.read()
            pillow_img = Image.open(io.BytesIO(roundId.body))
            # value = transform_image(pillow_img)
            # print(type(value))
            pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
            current_roundId = pytesseract.image_to_string(
                pillow_img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            # print(current_roundId.strip())
            # image_bytes =  file.read()
            pillow_img = Image.open(io.BytesIO(file.body))
            tensor = transform_image(pillow_img)
            prediction, total = predict(tensor)
            url = "https://casino-api2.odinflux.com/game/" + str(gameId)
            data = {"roundId": current_roundId.strip(), "total": total, "card": prediction.to_json()}
            print(data)
            # res = requests.post(url, json=data)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    # res = requests.post(url, json=data)
                    print(response.status)
            # try:
            #     th = threading.Thread(target=thread_request, args=(url, data))
            #     th.start()
            #     # print(prediction.to_json())
            #     # loop = asyncio.new_event_loop()
            #     # loop.run_until_complete(async_main_requests(
            #     #     "https://casino-api.odinflux.com/game/" + str(gameId), {"roundId": current_roundId.strip(), "total": total, "card": prediction.to_json()}))
            #     # res = requests.post(
            #     #     "https://casino.odinflux.com/game/" + str(gameId), json={"roundId": current_roundId.strip(), "card": prediction.to_json()})
            #     # print("Response code: {}".format(res.status_code))
            # except Exception as e:
            #     print(e)
            #     print("Error handhled in main website posting")
            # return jsonify({"roundId": current_roundId.strip(), "card": prediction.to_json()})
            # try:
            #     loop = asyncio.get_event_loop()
            #     loop.run_until_complete(async_main_requests(
            #         url, data))
            #     # res = requests.post(
            #     #     self.server_url, json=over_all_data)
            #     # print("Response code: {}".format(res.status_code))
            # except Exception as e:
            #     print(e)
            #     print("Error handhled in main website posting")
        except Exception as e:
            print(e)
            # return jsonify({"error": str(e)})

    return json({'hello': 'world'})


if __name__ == "__main__":
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8001, threads=100)
    app.run(host="0.0.0.0", port=8001, workers = 12)
