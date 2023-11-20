from ultralytics import YOLO
from time import time
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import io
from datetime import datetime
import email_config
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio

# Load default configuration from JSON file
# "source": "rtsp://root:pass@192.168.1.190/axis-media/media.amp",
parameters_file = Path("parameters.json")
attach_image = True
attach_video = False

with open(parameters_file, "r") as f:
    parameters = json.load(f)

email_trigger_count = parameters["email_trigger_count"] - 1
mail_option = parameters["mail_option"]
video_record_in_sec = parameters["video_length"]
class_ids = parameters["class_ids"]
mail_pause_in_sec = parameters["mail_pause_in_sec"]
time_interval_for_enough_frame_sightings_in_sec = parameters["time_interval_for_enough_frame_sightings_in_sec"]


def send_mail(mail_time, class_name, media_attachment):

    msg = MIMEMultipart()
    msg['From'] = email_config.sender_email
    msg['Subject'] = email_config.get_email_subject(class_name, mail_time)
    msg.attach(MIMEText(email_config.get_email_body(class_name, mail_time), 'plain'))

    if media_attachment != None:
        msg.attach(media_attachment)

    with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
        server.starttls()
        server.login(email_config.sender_email, email_config.smtp_password)

        for recipient_email in email_config.recipient_emails:
            msg['To'] = recipient_email

            try:
                server.sendmail(email_config.sender_email, recipient_email, msg.as_string())
            except Exception as e:
                print(e)

        server.quit()



def prepare_mail_attachments(r_list, class_id):

    r = r_list[0]
    mail_time = datetime.now()
    class_name = r.names[class_id]

    if mail_option == "Mail with Image Attachment":
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        image_byte_array = io.BytesIO()
        im.save(image_byte_array, format='JPEG')
        media_attachment = MIMEImage(image_byte_array.getvalue(), name=f"{class_name}_detection.jpg")

    elif mail_option == "Mail with Video Attachment":
        total_times = [r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess'] for r in r_list]
        average_fps = len(total_times) / sum(total_times) * 1000

        video_byte_array = io.BytesIO()
        with imageio.get_writer(video_byte_array, format='mp4', fps=average_fps) as writer:
            for r in r_list:
                frame = r.plot()[..., ::-1]
                writer.append_data(frame)

        video_base = MIMEBase('application', 'octet-stream')
        video_base.set_payload(video_byte_array.getvalue())
        encoders.encode_base64(video_base)
        video_base.add_header('Content-Disposition', f"attachment; filename={class_name}_detection.mp4")
        media_attachment = video_base

    else:
        media_attachment = None

    send_mail(mail_time, class_name, media_attachment)
        


def send_email_wrapper(args):
    prepare_mail_attachments(*args)

def start_stream():

    results_list = {class_id: [] for class_id in class_ids}
    model = YOLO(parameters["yolo_model_path"])
    count = {class_id: 0 for class_id in class_ids}
    time_last_mail = {class_id: 0 for class_id in class_ids}
    time_first_detection = {class_id: 0 for class_id in class_ids}
    save_frames_for_video = {class_id: False for class_id in class_ids}

    results = model.predict(parameters["source"], stream=True, save=False, show=True, conf=parameters["confidence_level"])

    with ThreadPoolExecutor(max_workers=len(class_ids)) as executor:
        for r in results:
            for class_id in class_ids:
                if class_id in r.boxes.cls:
                    time_now = time()
                    if time_now - time_last_mail[class_id] >= mail_pause_in_sec:
                        if count[class_id] == 0:
                            time_first_detection[class_id] = time()

                        count[class_id] += 1

                        if time_now - time_first_detection[class_id] > time_interval_for_enough_frame_sightings_in_sec:
                            count[class_id] = 0

                        if count[class_id] >= email_trigger_count:
                            if mail_option == "Mail with Image Attachment" or mail_option == "Mail without Attachment":
                                executor.submit(send_email_wrapper, [[r], class_id])
                            elif mail_option == "Mail with Video Attachment":
                                save_frames_for_video[class_id] = True

                            time_last_mail[class_id] = time()
                            count[class_id] = 0
                    
                    if save_frames_for_video[class_id]:
                        if time_now - time_last_mail[class_id] <= video_record_in_sec:
                            results_list[class_id].append(r)
                        else:
                            frames_to_save = results_list[class_id].copy()
                            executor.submit(send_email_wrapper, [frames_to_save, class_id])
                            results_list[class_id].clear()  # Clear the list for the specific class_id
                            save_frames_for_video[class_id] = False




start_stream()



#mail_pause_sec !> mail_trigger_count