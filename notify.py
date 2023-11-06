import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
from os import getenv
from inference.inference_inria import LATEST_MODEL_PATH, LATEST_MODEL_NAME, InferenceInria, SAMPLE_PATH
import cv2
import numpy as np
from os.path import basename, isfile

load_dotenv()


SERVER_ADDRESS = getenv("SERVER_ADDRESS")
PORT = int(getenv("PORT"))
SENDER_EMAIL = getenv("SENDER_EMAIL")
SENDER_PASSWORD = getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = getenv("RECIPIENT_EMAIL")

SUBJECT = f"[Train] results {LATEST_MODEL_NAME}"

MODEL_VAL = LATEST_MODEL_PATH[:-3] + "_val.tsv"
BODY = f"Model: {LATEST_MODEL_NAME}, Sample: {basename(SAMPLE_PATH)}\n"


if isfile(MODEL_VAL):
    with open(MODEL_VAL, "r") as f:
        BODY += f.read()
else:
    BODY += "Val file not added."


class Notifier():
    def __init__(self) -> None:
        self.infer = InferenceInria(LATEST_MODEL_NAME, debug=False, save_out=False)

    def notify(self):
        # with open(SAMPLE_PATH, 'rb') as f:
        #     image_part_1 = MIMEImage(f.read())
        px_cls, px_prob = self.infer.infer_file(SAMPLE_PATH)
        buffer = cv2.imencode(".png", px_cls)[1]
        data_encode = np.array(buffer)
        image_part_2 = MIMEImage(data_encode.tobytes())
        image_part_2.add_header('Content-Disposition', f"attachment; filename={basename(SAMPLE_PATH)}.pxcls.png")

        buffer = cv2.imencode(".png", px_prob)[1]
        data_encode = np.array(buffer)
        image_part_1 = MIMEImage(data_encode.tobytes())
        image_part_1.add_header('Content-Disposition', f"attachment; filename={basename(SAMPLE_PATH)}.pxprob.png")

        message = MIMEMultipart()
        message['Subject'] = SUBJECT
        message['From'] = SENDER_EMAIL
        message['To'] = RECIPIENT_EMAIL
        html_part = MIMEText(BODY)
        message.attach(html_part)
        message.attach(image_part_1)
        message.attach(image_part_2)

        smtp = smtplib.SMTP(SERVER_ADDRESS, port=PORT)
        smtp.set_debuglevel(1)
        smtp.ehlo()  # send the extended hello to our server
        smtp.starttls()  # tell server we want to communicate with TLS encryption
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
        smtp.quit()


if __name__ == "__main__":
    notifier = Notifier()
    notifier.notify()
