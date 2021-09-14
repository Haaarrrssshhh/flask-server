import json
from cv2 import data
import numpy as np
import time
import pyautogui
import cv2
import pytesseract
import requests
import torch.nn as nn
import os
import torch
import pandas as pd
import asyncio
from aiohttp import ClientSession
import aiohttp
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class TwoROIS:

    def __init__(self, model, eventId):
        self.model = model
       
        self.eventId = eventId
        # if len(args) >= 1:
        #     self.server_url = str(
        #         self.ip)[2:-1].replace(",", "").replace("'", "") + str(eventId)
        #     print("INSDIDE IF: ", self.server_url)
        # else:
        self.server_url = "https://casino.odinflux.com/game/" + \
                str(eventId)
        print("INSDIDE ELSE: ", self.server_url)

        # self.server_url = "http://192.168.1.27:8001/game/" + str(eventId)

        # self.local_ip = self.ip + str(eventId)
        print("IP Addres", self.server_url)

    def selecting_rois(self, frames):
        ROIs = cv2.selectROIs("Select Rois", frames)
        width1 = abs(ROIs[0][0] - (ROIs[0][2] + ROIs[0][0]))
        height1 = abs(ROIs[0][2] - (ROIs[0][3] + ROIs[0][2]))
        topLeftX1 = ROIs[0][0]
        topLeftY1 = ROIs[0][1]

        width2 = abs(ROIs[1][0] - (ROIs[1][2] + ROIs[1][0]))
        height2 = abs(ROIs[1][2] - (ROIs[1][3] + ROIs[1][2]))
        topLeftX2 = ROIs[1][0]
        topLeftY2 = ROIs[1][1]

        return topLeftX1, topLeftY1, height1, width1, topLeftX2, topLeftY2, height2, width2

    def first_capture(self):
        img = pyautogui.screenshot()
        img_np = np.array(img)
        (x1, y1, h1, w1, x2, y2, h2, w2) = self.selecting_rois(img_np)

        return (x1, y1, h1, w1, x2, y2, h2, w2)

    async def async_main_requests(self, url, data):
        async with ClientSession() as session:
            async with session.post(url, json=data) as resp:
                print(resp.status)
                print(await resp.json())

    async def async_admin_requests(self, url, _data):
        print("----URL---", url)
        print("----DATA----", _data)
        # print("----files----", files)

        async with ClientSession() as session:
            async with session.post(url, data=_data) as resp:
                print(
                    "ADMIN RESPONSE: -------------------------------------->", resp.status)
                print("ADMIN RESPONSE: -------------------------------------->", await resp.json())
            # print(await resp.json())

    def detection(self):

        card_list = []
        round_list = []
        prev_id = ''
        files = []

        (x1, y1, h1, w1, x2, y2, h2, w2) = self.first_capture()
        while True:
            img = pyautogui.screenshot()
            img_np = np.array(img)

            frame1 = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            frame1 = img_np[y1:y1+h1, x1:x1+w1]

            frame2 = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            frame2 = img_np[y2:y2+h2, x2:x2+w2]

            time.sleep(1)
            results = self.model(frame1)
            df = results.pandas().xyxy[0]
            card_list.append(frame1)
            round_list.append(frame2)
            if len(card_list) >= 10 and len(round_list) >= 10:
                # print("Emptying frame....")
                card_list.pop(0)
                round_list.pop(0)
                # print("The lenght is: {}".format(len(card_fam)))
                # print("The lenght is: {}".format(len(round_fam)))

            # pytesseract.pytesseract.tesseract_cmd = r'C:/Users/hzota/AppData/Local/Tesseract-OCR/tesseract.exe'
            pytesseract.pytesseract.tesseract_cmd = r'C:/Users/User/AppData/Local/Tesseract-OCR/tesseract.exe'
            current_round_id = pytesseract.image_to_string(
                frame2, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            try:
                if prev_id != current_round_id:
                    cv2.imwrite("card.png", card_list[-5])
                    cv2.imwrite("roundId.png", round_list[-5])
                    url = "https://casino.odinflux.com/admin/detection"
                    CARD_PATH = "card.png"
                    ROUND_PATH = "roundId.png"
                    card = open(CARD_PATH, 'rb')
                    round = open(ROUND_PATH, 'rb')
                    _data = aiohttp.FormData()
                    _data.add_field("images", card, content_type="image/jpeg")
                    _data.add_field("images", round, content_type="image/jpeg")
                    _data.add_field("prev_id", str(prev_id.strip()))
                    _data.add_field("event_id", str(self.eventId))

                    # image_data = aiohttp.FormData()
                    # image_data.add_field('images', open(CARD_PATH, 'rb'), filename="card.png", content_type='multipart/x-mixed-replace')
                    # image_data.add_field('images', open(ROUND_PATH, 'rb'), filename="rounfid.png",  content_type='multipart/x-mixed-replace')
                    # image_data.add_field('prev_id', prev_id.strip(), content_type='application/json')
                    # image_data.add_field('event_id', self.eventId)

                    # print(files)
                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(self.async_admin_requests(
                            url, _data))
                    except Exception as e:
                        print("Error handhled in admin posting")
                    # finally:
                    #     response = requests.post(
                    #         url, id_data, files=files)

                    prev_id = current_round_id
                    # print("changing round")
            except Exception as e:
                print(e)
            over_all_data = {"roundId": current_round_id.strip(),
                             "total": len(df.index),
                             "card": df.to_json(),
                             }
            # excel = pd.DataFrame.from_dict(over_all_data,  orient='index')
            # excel = pd.to_excel(excel)
            # print(excel)

            print(over_all_data)
            print("Event ID: {}".format(self.eventId))
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.async_main_requests(
                    self.server_url, over_all_data))
                # res = requests.post(
                #     self.server_url, json=over_all_data)
                # print("Response code: {}".format(res.status_code))
            except Exception as e:
                print(e)
                print("Error handhled in main website posting")
            # finally:
            #     res = requests.post(
            #         self.server_url, json=over_all_data)
            # print("Response code: {}".format(res.status_code))
