import json
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import ttk  # Import the ttk module
from tkinter import Checkbutton
from tkinter.ttk import Combobox
from tkinter import Label, Entry, Button, Listbox, Scrollbar, messagebox, Scale
import os
import requests
from requests.exceptions import RequestException
from urllib.parse import urlparse
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
import main_program
import threading
from PIL import Image, ImageTk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import io
import email_config
from concurrent.futures import ThreadPoolExecutor
import imageio
import numpy as np
import cv2
import multiprocessing
import time
import cv2
# "source": "rtsp://root:pass@192.168.1.190/axis-media/media.amp",

man_start_but_press = False
man_stop_but_press = False
manual_video_max_len = 30

config_file = Path("parameters.json")

with open(config_file, "r") as f:
    config = json.load(f)
parsed_url = urlparse(config["source"])
camera_url = f"http://{parsed_url.hostname}/axis-cgi/opticscontrol.cgi"  # Replace with the actual URL
auth = requests.auth.HTTPDigestAuth(parsed_url.username, parsed_url.password)

detected_class_ids_list = []
total_times_list = []
email_sent_flags = [] 

cap = None  # Initialize the capture object outside the loop

def send_mail(mail_time, class_name, media_attachment, manual):

    msg = MIMEMultipart()
    msg['From'] = email_config.sender_email
    msg['Subject'] = email_config.get_email_subject(class_name, mail_time, manual)
    msg.attach(MIMEText(email_config.get_email_body(class_name, mail_time, manual), 'plain'))


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



def prepare_mail_attachments(r_list, class_id, mail_option, manual):

    r = r_list[0]
    mail_time = datetime.now()
    if not manual:
        class_name = r.names[class_id]
    else:
        all_class_ids = list(set(int(item) for r in r_list for item in r.boxes.cls.tolist()))
        class_names = [r.names[integer] for integer in all_class_ids]
        class_name = '-'.join(class_names)

    if (not manual and mail_option == "Mail with Image Attachment") or (manual and mail_option_combobox.get() == "Mail with Image Attachment"):
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        image_byte_array = io.BytesIO()
        im.save(image_byte_array, format='JPEG')
        media_attachment = MIMEImage(image_byte_array.getvalue(), name=f"{class_name}_detection.jpg")

    elif (not manual and mail_option == "Mail with Video Attachment") or (manual and mail_option_combobox.get() == "Mail with Video Attachment"):
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
    
    send_mail(mail_time, class_name, media_attachment, manual)

def send_email_wrapper(args):
    prepare_mail_attachments(*args)


def main_function():

    global detected_class_ids_list
    global total_times_list
    global email_sent_flags
    global man_start_but_press
    global man_stop_but_press
  
    with open(config_file, "r") as f:
        parameters = json.load(f)

    email_trigger_count = parameters["email_trigger_count"] - 1
    mail_option = parameters["mail_option"]
    video_record_in_sec = parameters["video_length"]
    class_ids = parameters["class_ids"]
    mail_pause_in_sec = parameters["mail_pause_in_sec"]
    time_interval_for_enough_frame_sightings_in_sec = parameters["time_interval_for_enough_frame_sightings_in_sec"]
    record_plot = parameters["record_plot"]
    show_stream = parameters["show_stream"]

    results_list = {class_id: [] for class_id in class_ids}
    model = YOLO(parameters["yolo_model_path"])
    count = {class_id: 0 for class_id in class_ids}
    time_last_mail = {class_id: 0 for class_id in class_ids}
    time_first_detection = {class_id: 0 for class_id in class_ids}
    save_frames_for_video = {class_id: False for class_id in class_ids}
    save_manual_video = False
    manual_res_list = []

    if record_plot:
        if os.path.exists('plot_data.csv'):
            with open('plot_data.csv', 'w', newline=''):
                pass  # Empty block to truncate the file
    
    if not show_stream:
        canvas.delete("all")
  
    with ThreadPoolExecutor(max_workers=None) as executor:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                result = model.predict(frame, conf=parameters["confidence_level"], verbose=False)
                r = result[0]

                if show_stream: 
                    annotated_frame = r.plot()
                    update_display(annotated_frame)
                
                email_flag = 0
                for class_id in class_ids:
                    if class_id in r.boxes.cls:
                        time_now = time.time()
                        if time_now - time_last_mail[class_id] >= mail_pause_in_sec:
                            if count[class_id] == 0:
                                time_first_detection[class_id] = time.time()

                            count[class_id] += 1

                            if time_now - time_first_detection[class_id] > time_interval_for_enough_frame_sightings_in_sec:
                                count[class_id] = 0

                            if count[class_id] >= email_trigger_count:
                                if mail_option == "Mail with Image Attachment" or mail_option == "Mail without Attachment":
                                    executor.submit(send_email_wrapper, [[r], class_id, mail_option, False])
                                    email_flag = 1
                                elif mail_option == "Mail with Video Attachment":
                                    save_frames_for_video[class_id] = True

                                time_last_mail[class_id] = time.time()
                                count[class_id] = 0
                        
                        if save_frames_for_video[class_id]:
                            if time_now - time_last_mail[class_id] <= video_record_in_sec:
                                results_list[class_id].append(r)
                            else:
                                frames_to_save = results_list[class_id].copy()
                                executor.submit(send_email_wrapper, [frames_to_save, class_id, mail_option, False])
                                email_flag = 1
                                results_list[class_id].clear()  # Clear the list for the specific class_id
                                save_frames_for_video[class_id] = False

                if man_start_but_press:
                    man_start_but_press = False
                    if mail_option_combobox.get() == "Mail with Image Attachment" or mail_option_combobox.get() == "Mail without Attachment":
                        executor.submit(send_email_wrapper, [[r], None, mail_option, True])
                        email_flag = 2
                    elif mail_option_combobox.get() == "Mail with Video Attachment":
                        save_manual_video = True
                        manual_video_start_time = time.time()

                if save_manual_video:
                    manual_res_list.append(r)
                    if time.time() - manual_video_start_time > manual_video_max_len:
                        man_stop_but_press = False
                        save_manual_video = False
                        manual_frames_to_save = manual_res_list.copy()
                        executor.submit(send_email_wrapper, [manual_frames_to_save, None, mail_option, True])
                        email_flag = 2
                        manual_res_list.clear()
                        manual_stop_button.config(state=tk.DISABLED)
                        manual_start_button.config(state=tk.NORMAL)
                        messagebox.showerror("Error", f"Manual Video can maximum be {manual_video_max_len}")

                if man_stop_but_press:
                    man_stop_but_press = False
                    save_manual_video = False
                    manual_frames_to_save = manual_res_list.copy()
                    executor.submit(send_email_wrapper, [manual_frames_to_save, None, mail_option, True])
                    email_flag = 2
                    manual_res_list.clear()


                if record_plot:
                    email_sent_flags.append(email_flag)
                    detected_class_ids_list.append(np.array(r.boxes.cls))
                    total_times_list.append(time.time())
            
                    # the frequency of data saving into csv file is arbitrarly and could be tuned
                    if len(detected_class_ids_list) % 25 == 0:
                        # Save the array and total times to the same CSV file
                        with open('plot_data.csv', 'a', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            for detected_class_id, total_time, email_sent_flag in zip(detected_class_ids_list, total_times_list, email_sent_flags):
                                csvwriter.writerow(detected_class_id.tolist() + [total_time, email_sent_flag])
                        
                        # Reset the lists
                        detected_class_ids_list = []
                        total_times_list = []
                        email_sent_flags = []
    
            else:
                break
    canvas.delete("all")

    
def update_display(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image=img)

    canvas.config(width=img.width, height=img.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

#############################################################################################################################

def is_save_needed():
    # Check if any parameter has been changed
    try:
        error_label.config(text="")
        return (
            show_stream_var.get() != config["show_stream"] or
            record_plot_var.get() != config["record_plot"] or
            float(videolen_entry.get()) != config["video_length"] or
            float(confidence_entry.get()) != config["confidence_level"] or
            int(email_trigger_count_entry.get()) != config["email_trigger_count"] or
            float(mail_pause_entry.get()) != config["mail_pause_in_sec"] or
            float(time_interval_entry.get()) != config["time_interval_for_enough_frame_sightings_in_sec"] or
            class_id_entry.get() != ",".join(map(str, config["class_ids"])) or
            yolo_model_combobox.get() != os.path.relpath(config["yolo_model_path"], "yolo_models") or
            mail_option_combobox.get() != config["mail_option"] or
            source_entry.get() != config["source"]
        )
    except ValueError as e:
        error_label.config(text=str(e))
        return False

def toggle_save_button(event=None):

    save_button['state'] = 'normal' if is_save_needed() else 'disabled'

def start_program():

    global cap, camera_url, auth
    with open(config_file, "r") as f:
        parameters = json.load(f)
    video_source = parameters["source"]

    parsed_url = urlparse(video_source)
    camera_url = f"http://{parsed_url.hostname}/axis-cgi/opticscontrol.cgi"  # Replace with the actual URL
    auth = requests.auth.HTTPDigestAuth(parsed_url.username, parsed_url.password)

    if video_source.isdigit():
        # If video_source consists of digits, treat it as a webcam index
        cap = cv2.VideoCapture(int(video_source))
    else:
        # If video_source is a string, treat it as an RTSP link
        cap = cv2.VideoCapture(video_source)
 
  
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
 
    thread = threading.Thread(target=main_function, name='main_function')
    thread.start()

# Function to stop the main_program.py
def stop_program():
    global cap
    if cap is not None and cap.isOpened():
            cap.release()

    with open('plot_data.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for detected_class_id, total_time, email_sent_flag in zip(detected_class_ids_list, total_times_list, email_sent_flags):
            csvwriter.writerow(detected_class_id.tolist() + [total_time, email_sent_flag])

    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
  
def make_plot():
    plt.close('all')
    model = YOLO(config["yolo_model_path"])
    class_names_by_id = {int(class_id): class_name for class_id, class_name in model.names.items()}
    csv_file_path = 'plot_data.csv'
    data = []

    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Convert each value to float
            row = [float(value) for value in row]
            data.append(row)

    class_counts_over_time = []
    total_times = []  # Initialize with 0, as the first total_time is the time difference from the start
    email_flags = []

    for i in range(1, len(data)):
        total_times.append(data[i][-2])  # Accumulate total_time
        class_counts = Counter(data[i][:-2])  # Exclude the last two columns (email_flag and total_time)
        class_counts_over_time.append(class_counts)
        email_flags.append(data[i][-1])  # Last element is the email_flag

    # Extract class IDs and their counts
    class_ids = list(set(class_id for time_step_counts in class_counts_over_time for class_id in time_step_counts))
    class_counts_by_class = {class_id: [counts[class_id] for counts in class_counts_over_time] for class_id in class_ids}

    # Convert total_times to seconds
    start_time = datetime.fromtimestamp(total_times[0])
    total_times_seconds = [start_time + timedelta(seconds=time - total_times[0]) for time in total_times]

    # Plot the sum of each class over time using accumulated total_time as the x-axis
    plt.figure(figsize=(10, 6))

    for class_id, counts_over_time in class_counts_by_class.items():
        class_name = class_names_by_id.get(int(class_id), f'Class {int(class_id)}')  # Use class name if available, else use class ID
        plt.plot(total_times_seconds, counts_over_time, label=class_name)

    legend_text_1 = False
    legend_text_2 = False
    for time_point, email_flag in zip(total_times_seconds, email_flags):
        if email_flag == 1:
            label1 = 'Automatic Email Sent' if email_flag == 1 and not legend_text_1 else ''
            plt.axvline(x=time_point, color='black', linestyle='--', alpha=0.5, label=label1)
            legend_text_1 = True
        if email_flag == 2:
            label2 = 'Manual Email Sent' if email_flag == 2 and not legend_text_2 else ''
            plt.axvline(x=time_point, color='black', linestyle='--', label=label2)
            legend_text_2 = True
    
    plt.gca().xaxis_date()
    plt.gcf().autofmt_xdate()
    plt.title('Number of Appearances of each Detected Class over Time')
    plt.xlabel('Total Time (s)')
    plt.ylabel('Number of Appearances')
    plt.legend()
    plt.show()

def is_valid_class_ids(class_ids_str, class_id_list):
    try:
        class_ids = [int(class_id) for class_id in class_ids_str.split(",")]
        return all(0 <= cid < len(class_id_list) for cid in class_ids)
    except ValueError:
        return False

def save_config():
    try:

        videolen_val = float(videolen_entry.get())
        confidence_level_val = float(confidence_entry.get())
        email_trigger_count_val = int(email_trigger_count_entry.get())
        mail_pause_val = float(mail_pause_entry.get())
        time_interval_val = float(time_interval_entry.get())
        mail_option = mail_option_combobox.get()

        if not (0.0 <= confidence_level_val <= 1.0):
            raise ValueError("Confidence Level must be between 0.0 and 1.0.")
        
        if videolen_val < 0:
            raise ValueError("Video Length must be a non-negative float.")

        if email_trigger_count_val < 0:
            raise ValueError("Time Interval must be a non-negative float.")
        
        if mail_pause_val < 0:
            raise ValueError("Mail Pause must be a non-negative float.")
        
        if time_interval_val < 0:
            raise ValueError("Time Interval must be a non-negative float.")
        
        if videolen_val >= mail_pause_val and mail_option == "Mail with Video Attachment":
            raise ValueError("Video Length must be smaller than No Mail Sending Interval.")


        selected_model = yolo_model_combobox.get()
        model = YOLO(os.path.join("yolo_models", selected_model))
        config["yolo_model_path"] = os.path.join("yolo_models", selected_model)

        config["mail_option"] = mail_option

        config["source"] = source_entry.get()
        config["confidence_level"] = confidence_level_val
        config["video_length"] = videolen_val
        config["show_stream"] = show_stream_var.get()
        config["record_plot"] = record_plot_var.get()

        class_ids_str = class_id_entry.get()

        if not is_valid_class_ids(class_ids_str, model.names):
            raise ValueError(f"Invalid class ids. Must be comma-separated integers between 0 and {len(model.names) - 1}.")

        config["class_ids"] = [int(class_id) for class_id in class_ids_str.split(",")]

        config["email_trigger_count"] = email_trigger_count_val
        config["mail_pause_in_sec"] = mail_pause_val
        config["time_interval_for_enough_frame_sightings_in_sec"] = time_interval_val

        toggle_save_button()

        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

    except ValueError as e:
        messagebox.showerror("Error", str(e))

def attachment_changed(event=None):

    current_option = mail_option_combobox.get()

    email_trigger_count_label.config(text="Nr. of Frames with Detection for Video Recording Start")
    email_trigger_count_label.grid()
    email_trigger_count_entry.grid()
    time_interval_label.grid()
    time_interval_entry.grid()
    mail_pause_label.grid()
    mail_pause_entry.grid()
    videolen_label.grid()
    videolen_entry.grid()
    class_id_label.grid()
    class_id_entry.grid()
    manual_start_button.grid()
    manual_start_button.config(text="Start Video Recording Manually")
    manual_stop_button.grid()
    manual_stop_button.config(text="Send Mail with Video")



    if current_option == "No Mail":
        # Hide widgets
        email_trigger_count_label.grid_remove()
        email_trigger_count_entry.grid_remove()
        time_interval_label.grid_remove()
        time_interval_entry.grid_remove()
        mail_pause_label.grid_remove()
        mail_pause_entry.grid_remove()
        videolen_label.grid_remove()
        videolen_entry.grid_remove()
        class_id_label.grid_remove()
        class_id_entry.grid_remove()
        manual_start_button.grid_remove()
        manual_stop_button.grid_remove()
    
    elif current_option == "Mail without Attachment":
        email_trigger_count_label.config(text="Nr. of Frames with Detection for Mail Trigger")
        videolen_label.grid_remove()
        videolen_entry.grid_remove()
        manual_stop_button.grid_remove()
        manual_start_button.config(text="Send Mail Manually")
     
    elif current_option == "Mail with Image Attachment":
        email_trigger_count_label.config(text="Nr. of Frames with Detection for Image Snapshot")
        videolen_label.grid_remove()
        videolen_entry.grid_remove()
        manual_stop_button.grid_remove()
        manual_start_button.config(text="Send Mail with Image Manually")


    #mail_option_combobox.select_clear()
    toggle_save_button()

def update_class_id_list(event=None):
    selected_model = yolo_model_combobox.get()
    model = YOLO(os.path.join("yolo_models", selected_model))

    # Clear the listbox
    listbox.delete(0, tk.END)

    # Populate the listbox with class IDs for the selected YOLO model
    for class_id, class_name in model.names.items():
        listbox.insert(tk.END, f"{class_id}: {class_name}")

    #yolo_model_combobox.selection_clear()
    toggle_save_button()

def update_zoom_level(zoom_level):
    # Call the function to update the zoom level in axis_camera_control.py
    payload = {
        "apiVersion": "1",
        "context": "Axis library",
        "method": "setMagnification",
        "params": {
            "optics": [
                {
                    "opticsId": 0,
                    "magnification": zoom_level
                }
            ]
        }   
    }
    try:
        print(zoom_level)
        response = requests.post(camera_url, json=payload, auth=auth)
    except RequestException as e:
        print("Request Exception:", e)
   

def on_slider_release(event):
    update_zoom_level(zoom_slider.get())

def on_closing():
    # Function to handle the window closure event
    root.destroy()  # Destroy the Tkinter window
    plt.close('all')  # Close all matplotlib windows

def manual_start_button_pressed():
    global man_start_but_press
    man_start_but_press = True

    if mail_option_combobox.get() == "Mail with Video Attachment":
        manual_stop_button.config(state=tk.NORMAL)
        manual_start_button.config(state=tk.DISABLED)


def manual_stop_button_pressed():
    global man_stop_but_press
    man_stop_but_press = True

    manual_stop_button.config(state=tk.DISABLED)
    manual_start_button.config(state=tk.NORMAL)
# Load default configuration from JSON file

# Create a simple tkinter GUI
root = tk.Tk()
root.title("Alascom Axis Camera Object Detection")
style = ttk.Style()
style.theme_use("clam")

root.protocol("WM_DELETE_WINDOW", on_closing)

# Confidence Level
Label(root, text="Confidence Level:").grid(row=0, column=0, padx=10, pady=10)
confidence_entry = Entry(root)
confidence_entry.insert(0, config["confidence_level"])
confidence_entry.grid(row=0, column=1, padx=10, pady=10)
confidence_entry.bind("<KeyRelease>", toggle_save_button) 

# Source
Label(root, text="Videostream Source:").grid(row=1, column=0, padx=10, pady=10)
source_entry = Entry(root)
source_entry.insert(0, config["source"])
source_entry.grid(row=1, column=1, padx=10, pady=10)
source_entry.bind("<KeyRelease>", toggle_save_button)  

#Attachment Option
Label(root, text="Mail Options:").grid(row=2, column=0, padx=10, pady=10)
mail_option_combobox = Combobox(root, values=["No Mail", "Mail without Attachment", "Mail with Image Attachment", "Mail with Video Attachment"], state='readonly')
mail_option_combobox.set(config["mail_option"])
mail_option_combobox.grid(row=2, column=1, padx=10, pady=10)
mail_option_combobox.bind("<<ComboboxSelected>>", attachment_changed)

# Email Trigger Count
email_trigger_count_label = Label(root, text="Number of Frames with Detection for Mail Trigger:")
email_trigger_count_label.grid(row=3, column=0, padx=10, pady=10)
email_trigger_count_entry = Entry(root)
email_trigger_count_entry.insert(0, config["email_trigger_count"])
email_trigger_count_entry.grid(row=3, column=1, padx=10, pady=10)
email_trigger_count_entry.bind("<KeyRelease>", toggle_save_button)  

# Time Interval
time_interval_label = Label(root, text="Time To be Detected Interval (sec):")
time_interval_label.grid(row=4, column=0, padx=10, pady=10)
time_interval_entry = Entry(root)
time_interval_entry.insert(0, config["time_interval_for_enough_frame_sightings_in_sec"])
time_interval_entry.grid(row=4, column=1, padx=10, pady=10)
time_interval_entry.bind("<KeyRelease>", toggle_save_button)  

# Mail Pause
mail_pause_label = Label(root, text="No Mail Sending Interval (sec):")
mail_pause_label.grid(row=5, column=0, padx=10, pady=10)
mail_pause_entry = Entry(root)
mail_pause_entry.insert(0, config["mail_pause_in_sec"])
mail_pause_entry.grid(row=5, column=1, padx=10, pady=10)
mail_pause_entry.bind("<KeyRelease>", toggle_save_button)  

# Video Length
videolen_label = Label(root, text="Video Length (sec):")
videolen_label.grid(row=6, column=0, padx=10, pady=10)
videolen_entry = Entry(root)
videolen_entry.insert(0, config["video_length"])
videolen_entry.grid(row=6, column=1, padx=10, pady=10)
videolen_entry.bind("<KeyRelease>", toggle_save_button)  

# YOLO Model
Label(root, text="YOLO Model:").grid(row=0, column=2, padx=10, pady=10)
yolo_model_combobox = Combobox(root, values=[file for file in os.listdir("yolo_models") if file.endswith(".pt")], state='readonly')
yolo_model_combobox.set(os.path.relpath(config["yolo_model_path"], "yolo_models"))
yolo_model_combobox.grid(row=0, column=3, padx=10, pady=10)
yolo_model_combobox.bind("<<ComboboxSelected>>", update_class_id_list)

# Class ID and Name Listbox
listbox_label = Label(root, text="Object ID List:")
listbox_label.grid(row=1, column=2, padx=10, pady=10, rowspan=5)
listbox = Listbox(root, selectmode=tk.SINGLE, height=12)
listbox.grid(row=1, column=3, padx=10, pady=10, rowspan=5)

# Class IDs
class_id_label = Label(root, text="Object IDs to be emailed (comma-separated):")
class_id_label.grid(row=6, column=2, padx=10, pady=10)
class_id_entry = Entry(root)
class_id_entry.insert(0, ",".join(map(str, config["class_ids"])))
class_id_entry.grid(row=6, column=3, padx=10, pady=10)
class_id_entry.bind("<KeyRelease>", toggle_save_button)  

fixed_width_value = 400
fixed_height_value = 45

root.columnconfigure(0, minsize=fixed_width_value)
for row in range(root.grid_size()[1]):
    root.rowconfigure(row, min=fixed_height_value)


# Scrollbar for Listbox
scrollbar = Scrollbar(root, orient=tk.VERTICAL)
scrollbar.grid(row=0, column=4, padx=0, pady=10, rowspan=6)
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

# Save Button
save_button = Button(root, text="Save Parameters", command=save_config, state='disabled')
save_button.grid(row=9, column=3, columnspan=2, pady=20)

error_label = Label(root, text="", fg="red")
error_label.grid(row=10, column=0, columnspan=4)

# Start Button
start_button = Button(root, text="Start Detection", command=start_program, state=tk.NORMAL)
start_button.grid(row=9, column=2, pady=10, padx=10)

# Stop Button
stop_button = Button(root, text="Stop Detection", command=stop_program, state=tk.DISABLED)
stop_button.grid(row=9, column=2, pady=10,  padx=(280,10))

plot_button = Button(root, text="Show Plot", command=make_plot, state=tk.NORMAL)
plot_button.grid(row=8, column=2, pady=10,  padx=(280,10))

manual_start_button = Button(root, text="Start Video Recording Manually", command=manual_start_button_pressed, state=tk.NORMAL)
manual_start_button.grid(row=8, column=0, padx=10, pady=10)

manual_stop_button = Button(root, text="Send Mail with Video", command=manual_stop_button_pressed, state=tk.DISABLED)
manual_stop_button.grid(row=8, column=1, padx=10, pady=10)


show_stream_var = tk.BooleanVar(value=bool(config["show_stream"]))
show_stream_checkbox = Checkbutton(root, text="Show Video Stream", variable=show_stream_var)
show_stream_checkbox.grid(row=8, column=3, pady=10, padx=10)
show_stream_checkbox.config(command=toggle_save_button)

record_plot_var = tk.BooleanVar(value=bool(config["record_plot"]))
record_plot_checkbox = Checkbutton(root, text="Record New Plot", variable=record_plot_var)
record_plot_checkbox.grid(row=8, column=2, pady=10, padx=10)
record_plot_checkbox.config(command=toggle_save_button)

zoom_slider_label = Label(root, text="Zoom Level:")
zoom_slider_label.grid(row=9, column=0, padx=10, pady=10)

zoom_slider = Scale(root, from_=1, to=2, resolution=0.01, orient=tk.HORIZONTAL, length=200)
zoom_slider.set(1.0)  # Set the initial value
zoom_slider.grid(row=9, column=1, padx=10, pady=10)
zoom_slider.bind("<ButtonRelease-1>", on_slider_release)

video_frame = tk.Frame(root)
video_frame.grid(row=11, column=0, columnspan=5, pady=10)
canvas = tk.Canvas(video_frame)
canvas.grid(row=0, column=0)
#zoom_slider.config(command=lambda value: update_zoom_level(zoom_slider.get()))

update_zoom_level(1.0)
update_class_id_list()
attachment_changed()


root.mainloop()


# fix auth , camera_url and compatibility for zoom
# fix manual button
# verbose toggle