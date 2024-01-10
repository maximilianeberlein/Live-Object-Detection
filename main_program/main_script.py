from tkinter import Label, Entry, Button, Listbox, Scrollbar, messagebox, Scale
from concurrent.futures import ThreadPoolExecutor
from email.mime.multipart import MIMEMultipart
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
from smtplib import SMTP, SMTPException
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from urllib.parse import urlparse
from tkinter.ttk import Combobox
from tkinter import Checkbutton
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageTk
from ultralytics import YOLO
from email import encoders
from tkinter import ttk 
import tkinter as tk
import email_config
import numpy as np
import threading
import requests
import imageio
import json
import time
import cv2
import csv
import io
import os
import re


# Global variables
detected_class_ids_list, total_times_list, email_sent_flags = [], [], []
cap, camera_url, auth = None, None, None
man_start_but_press, man_stop_but_press = False, False
manual_video_max_len = 30                                                                                                       # maximum length of manual video recording in seconds before it is stopped

# File names
PARAMETERS_FILE = "parameters.json"
PLOT_DATA_FILE = 'plot_data.csv'

 ########################################################################################## Yolo Live Object Detection with Mail Alerts


def send_mail(mail_time, class_name, media_attachment, manual):
                                                                             
    mail_subject = email_config.get_email_subject(class_name, mail_time, manual)                                                # Define Mail Subject
    mail_text = MIMEText(email_config.get_email_body(class_name, mail_time, manual), 'plain')                                   # Define Mail Text

    with SMTP(email_config.smtp_server, email_config.smtp_port) as server:                                                      # Connect to SMTP server
        server.starttls()

        try:
            server.login(email_config.sender_email, email_config.smtp_password)                                                 # Login to SMTP server
            for recipient_email in load_parameters()["emails"] :                                                                # Send email to each of the recipients (load from parameters.json) in the list. If it doesnt work for one of the recipient addresses, this doesnt affect the other emails.
                
                msg = MIMEMultipart()
                msg['From'] = email_config.sender_email
                msg['To'] = recipient_email
                msg['Subject'] = mail_subject
                msg.attach(mail_text) 
                if media_attachment != None:                                                                                    # Attach media attachment if available
                    msg.attach(media_attachment)                                                                                     
                try:
                    server.sendmail(email_config.sender_email, recipient_email, msg.as_string())
                except SMTPException as e:
                    print(f"Failed to send email to {recipient_email}: {e}")
        except SMTPException as e:
            print(f"SMTP Exception: {e}")


def prepare_mail_attachments(r_list, class_id, mail_option, manual):
                            
    r = r_list[0]                                                                                                               # image attachment: r_list[0] contains the image frame /// video attachment: r_list contains all the video frames
    mail_time = datetime.now()                                                                      
    class_name = '-'.join(r.names[integer] for integer in set(int(item) for r in r_list                                         # Define the 'class name' based on whether it's a manual or automatic detection
                for item in r.boxes.cls.tolist())) if manual else r.names[class_id]
    
    if (not manual and mail_option == "Mail with Image Attachment") or (manual and mail_option_combobox.get() == "Mail with Image Attachment"):
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])                                                                       
        image_byte_array = io.BytesIO()
        im.save(image_byte_array, format='JPEG')
        media_attachment = MIMEImage(image_byte_array.getvalue(), name=f"{class_name}_detection.jpg")                           # Create image attachment
 
    elif (not manual and mail_option == "Mail with Video Attachment") or (manual and mail_option_combobox.get() == "Mail with Video Attachment"):
        total_times = [r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess'] for r in r_list]
        average_fps = len(total_times) / sum(total_times) * 1000
                                                                                                                        
        video_byte_array = io.BytesIO()                                                                                
        with imageio.get_writer(video_byte_array, format='mp4', fps=average_fps) as writer:
            for r in r_list:
                frame = r.plot()[..., ::-1]
                writer.append_data(frame)

        media_attachment = MIMEBase('application', 'octet-stream')
        media_attachment.set_payload(video_byte_array.getvalue())
        encoders.encode_base64(media_attachment)
        media_attachment.add_header('Content-Disposition', f"attachment; filename={class_name}_detection.mp4")                  # Create video attachment
    
    else:
        media_attachment = None                                                                                                 # No media attachment for other cases

    send_mail(mail_time, class_name, media_attachment, manual)


def send_email_wrapper(args):
    prepare_mail_attachments(*args)


def display_videoframe_on_canvas(frame):                                                                                        # display yolo frame output as image in tkinter canvas

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image=img)
    canvas.config(width=img.width, height=img.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo


def run_object_detection():                                                                                                     # This function continuously captures frames from the video source, performs object detection using YOLO, and handles email notifications and manual video recording based on the configured parameters.

    global detected_class_ids_list, total_times_list, email_sent_flags, man_start_but_press, man_stop_but_press
 
    parameters = load_parameters()                                                                                              # Load parameters from parameters.json
    email_trigger_count = parameters["email_trigger_count"] - 1
    mail_option = parameters["mail_option"]
    video_record_in_sec = parameters["video_length"]
    class_ids = parameters["class_ids"]
    mail_pause_in_sec = parameters["mail_pause_in_sec"]
    time_interval_for_enough_frame_sightings_in_sec = parameters["time_interval_for_enough_frame_sightings_in_sec"]
    record_plot = parameters["record_plot"]
    show_stream = parameters["show_stream"]

    results_list = {class_id: [] for class_id in class_ids}                                                                     # Initialize result lists, counters, and flags
    save_manual_video, manual_res_list = False, []
    count, time_last_mail, time_first_detection, save_frames_for_video = ({class_id: 0 for class_id in class_ids} for _ in range(4))
    model = YOLO(parameters["yolo_model_path"])

    if record_plot and os.path.exists(PLOT_DATA_FILE):                                                                          # Clear old plot data if recording a new plot
        with open(PLOT_DATA_FILE, 'w', newline=''):
            pass  

    if not show_stream:                                                                                                         # If the 'show stream' checkbox is empty, clear the canvas
        canvas.delete("all")
  
    with ThreadPoolExecutor(max_workers=None) as executor:                                                                      # Perform the object detection in multiple threads so that sending emails does not halt the program  
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                result = model.predict(frame, conf=parameters["confidence_level"], verbose=False)                               # Perform object detection using YOLO
                r = result[0]

                if show_stream:                                                                                                 # If the 'show stream' checkbox is ticked, display the video in the UI window
                    annotated_frame = r.plot()
                    display_videoframe_on_canvas(annotated_frame)
                
                email_flag = 0                                                                                                  # email_flag is the variable that tracks whether a mail has been sent in this loop iteration (0 = no mail sent, 1 = automated mail sent, 2 = manual mail sent)
                for class_id in class_ids:                                                                                      # Check only for the classes defined in 'Object IDs to be emailed (comma-separated)' in the UI
                    if class_id in r.boxes.cls:                                                                                 # Check if these chosen classes are detected by YOLO algorithm in the current frame
                        time_now = time.time()
                        if time_now - time_last_mail[class_id] >= mail_pause_in_sec:                                            # Check if enough time has passed since the last email
                            if count[class_id] == 0:
                                time_first_detection[class_id] = time.time()

                            count[class_id] += 1

                            if time_now - time_first_detection[class_id] > time_interval_for_enough_frame_sightings_in_sec:     # if not enough frames have been detected within the time interval, reset count to 0 (is helpful to disable mail alerts for short, false detections)
                                count[class_id] = 0

                            if count[class_id] >= email_trigger_count:                                                          # If enough frames have been detected before exceeding the time interval, trigger a mail send event
                                if mail_option == "Mail with Image Attachment" or mail_option == "Mail without Attachment":     
                                    executor.submit(send_email_wrapper, [[r], class_id, mail_option, False])                    # Send automated email without/image attachment in a new thread
                                    email_flag = 1
                                elif mail_option == "Mail with Video Attachment":
                                    save_frames_for_video[class_id] = True                                                      # Start recording video frames (automated)

                                time_last_mail[class_id] = time.time()                                                          # Update last email time and reset counter
                                count[class_id] = 0
                        
                        if save_frames_for_video[class_id]:                                                                     # Check if automated video recording should be started
                            if time_now - time_last_mail[class_id] <= video_record_in_sec:                                      # As long as the 'video length' defined in UI is not exceeded, save all frames into a list
                                results_list[class_id].append(r)
                            else:                                                                                               # If the 'video length' has been exceeded, send email with video attachment in a new thread
                                frames_to_save = results_list[class_id].copy()
                                executor.submit(send_email_wrapper, [frames_to_save, class_id, mail_option, False])
                                email_flag = 1
                                results_list[class_id].clear()  
                                save_frames_for_video[class_id] = False

                if man_start_but_press:                                                                                         # Check if the manual start button has been pressed
                    man_start_but_press = False
                    if mail_option_combobox.get() == "Mail with Image Attachment" or mail_option_combobox.get() == "Mail without Attachment":
                        executor.submit(send_email_wrapper, [[r], None, mail_option, True])                                     # Send manual email without/image attachment in a new thread
                        email_flag = 2
                    elif mail_option_combobox.get() == "Mail with Video Attachment":
                        save_manual_video = True                                                                                # Start recording video frames (manual)
                        manual_video_start_time = time.time()

                if save_manual_video:                                                                                           # Check if manual video recording should be started
                    manual_res_list.append(r)
                    if time.time() - manual_video_start_time > manual_video_max_len:                                            # If 'manual_video_max_len' (defined above in the code) is exceeded, stop video recording and send manual mail with video attachment
                        man_stop_but_press = False
                        save_manual_video = False
                        manual_frames_to_save = manual_res_list.copy()
                        executor.submit(send_email_wrapper, [manual_frames_to_save, None, mail_option, True])
                        email_flag = 2
                        manual_res_list.clear()
                        manual_stop_button.config(state=tk.DISABLED)
                        manual_start_button.config(state=tk.NORMAL)
                        messagebox.showerror("Error", f"Manual Video can maximum be {manual_video_max_len}")

                if man_stop_but_press:                                                                                          # If the manual stop button has been pressed, stop video recording and send manual mail with video attachment
                    man_stop_but_press = False
                    man_stop_but_press = False
                    save_manual_video = False
                    manual_frames_to_save = manual_res_list.copy()
                    executor.submit(send_email_wrapper, [manual_frames_to_save, None, mail_option, True])
                    email_flag = 2
                    manual_res_list.clear()


                if record_plot:                                                                                                 # If 'record new plot' checkbox is ticked, save the email_flags, detected classes and timestamps into a csv file                                                                   
                    email_sent_flags.append(email_flag)
                    detected_class_ids_list.append(np.array(r.boxes.cls))
                    total_times_list.append(time.time())
                    if len(detected_class_ids_list) % 25 == 0:                                                                  # The frequency of data saving (here 25) into csv file is arbitrarly and could be tuned
                        save_plot_data()
            else:
                break
    canvas.delete("all")
 
 ########################################################################################## Saving and Displaying Plot


def save_plot_data():                                                                                                           # This function appends detected class IDs, total times, and email sent flags to a CSV file. It is typically called at intervals during the object detection loop.

    global detected_class_ids_list, total_times_list, email_sent_flags

    with open(PLOT_DATA_FILE, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for detected_class_id, total_time, email_sent_flag in zip(detected_class_ids_list, total_times_list, email_sent_flags):
            csvwriter.writerow(detected_class_id.tolist() + [total_time, email_sent_flag])
    
    detected_class_ids_list, total_times_list, email_sent_flags = [], [], []                                                    # Reset the lists for the next interval


def show_plot_data():                                                                                                           # This function reads the object detection data from a CSV file and generates a plot showing the number of appearances of each detected class over time.


    plt.close('all')
    model = YOLO(config["yolo_model_path"])
    class_names_by_id = {int(class_id): class_name for class_id, class_name in model.names.items()}                             # Load YOLO model and retrieve class names
    data = []

    with open(PLOT_DATA_FILE, 'r') as csvfile:                                                                                  # Read data from the CSV file and convert values to floats
        csvreader = csv.reader(csvfile)
        data = [[float(value) for value in row] for row in csvreader] 
    
    class_counts_over_time = [Counter(row[:-2]) for row in data[1:]]                                                            # Extract total class counts, timestamps, and email_flags
    total_times = [row[-2] for row in data[1:]]
    email_flags = [row[-1] for row in data[1:]]

    class_ids = set(class_id for time_step_counts in class_counts_over_time for class_id in time_step_counts)                   # Extract class IDs and their counts
    class_counts_by_class = {class_id: [counts[class_id] for counts in class_counts_over_time] for class_id in class_ids}
    start_time = datetime.fromtimestamp(total_times[0])                                                                         # Convert total_times to seconds
    total_times_seconds = [start_time + timedelta(seconds=time - total_times[0]) for time in total_times]

    plt.figure(num='Live Object Detection History', figsize=(10, 6))

    for class_id, counts_over_time in class_counts_by_class.items():
        class_name = class_names_by_id.get(int(class_id), f'Class {int(class_id)}')                                             # Use class name if available, else use class ID for Plot Label
        plt.plot(total_times_seconds, counts_over_time, label=class_name)                                                       # Plot the sum of each class over time using accumulated total_time as the x-axis
 
    legend1_exists, legend2_exists = False, False
    for time_point, email_flag in zip(total_times_seconds, email_flags):                                                        # Add vertical lines for displaying sent emails (1 = automated, 2 = manual)
        if email_flag == 1:
            label = 'Automatic Email Sent' if email_flag == 1 and not legend1_exists else ''
            plt.axvline(x=time_point, color='black', linestyle='--', alpha=0.5, label=label)
            legend1_exists = True
        if email_flag == 2:
            label = 'Manual Email Sent' if email_flag == 2 and not legend2_exists else ''
            plt.axvline(x=time_point, color='black', linestyle='--', label=label)
            legend2_exists = True

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis_date()
    plt.gcf().autofmt_xdate()
    plt.title('Number of Appearances of each Detected Class over Time')
    plt.xlabel('Total Time (s)')
    plt.ylabel('Number of Appearances')
    plt.legend()
    plt.show()

 ########################################################################################## Parameter Saving Logic


def is_save_needed():                                                                                                           # Check if any parameter in the UI has been changed compared to the saved values in parameters.json                                                              

    try:
        if start_button['state'] == 'normal':
            error_label.config(text="")
            return (
                show_stream_var.get() != config["show_stream"] or
                record_plot_var.get() != config["record_plot"] or
                float(videolen_entry.get()) != config["video_length"] or
                float(confidence_entry.get()) != config["confidence_level"] or
                int(email_trigger_count_entry.get()) != config["email_trigger_count"] or
                float(mail_pause_entry.get()) != config["mail_pause_in_sec"] or
                float(time_interval_entry.get()) != config["time_interval_for_enough_frame_sightings_in_sec"] or
                [int(class_id.strip()) for class_id in class_id_entry.get().split(",")] != config["class_ids"] or
                [mail.strip() for mail in mail_list_entry.get().split(",")] != config["emails"] or
                yolo_model_combobox.get() != os.path.relpath(config["yolo_model_path"], "yolo_models") or
                mail_option_combobox.get() != config["mail_option"] or
                source_entry.get() != config["source"]
            )
        else:
            return False
    except ValueError as e:
        error_label.config(text=str(e))
        return False

def update_save_button_state(event=None):                                                                                       # This function checks if any parameter has been changed by calling the is_save_needed() function. If changes are detected, the save_button can be clicked. Otherwise,  the button is disabled and cannot be clicked.
    save_button['state'] = 'normal' if is_save_needed() else 'disabled'


def check_class_ids_vadility(class_ids_str, class_id_list):                                                                     # This function checks if the provided class IDs string can be converted into a list of integers. It also checks whether each integer is within the valid range for the given class_id_list.
    try:
        class_ids = [int(class_id) for class_id in class_ids_str.split(",")]
        return all(0 <= cid < len(class_id_list) for cid in class_ids)
    except ValueError:
        return False


def check_mail_list_vadility(mail_list):                                                                     # This function checks if the provided class IDs string can be converted into a list of integers. It also checks whether each integer is within the valid range for the given class_id_list.
    try:
        mails = [mail.strip() for mail in mail_list.split(",")]  # Strip whitespaces around each email
        email_pattern = re.compile(r"[^@]+@[^@]+\.[^@]+")  # Simple email pattern, you can adjust as needed
        return all(email_pattern.match(mail) for mail in mails)
    except ValueError:
        return False


def check_non_negative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be a non-negative float.")


def save_parameters():                                                                                                          # This function checks the validity of all parameters after the save button has been pressed. If valid, the updated parameters are saved to parameters.json. 

    try:
        check_non_negative(float(videolen_entry.get()), "Video Length")
        check_non_negative(int(email_trigger_count_entry.get()), "Email Trigger Count")
        check_non_negative(float(mail_pause_entry.get()), "Mail Pause")
        check_non_negative( float(time_interval_entry.get()), "Time Interval")

        if not (0.0 <= float(confidence_entry.get()) <= 1.0):
            raise ValueError("Confidence Level must be between 0.0 and 1.0.")
        if float(videolen_entry.get()) >= float(mail_pause_entry.get()) and mail_option_combobox.get() == "Mail with Video Attachment":
            raise ValueError("Video Length must be smaller than No Mail Sending Interval.")
        model_path = os.path.join("yolo_models", yolo_model_combobox.get())
        if not check_class_ids_vadility(class_id_entry.get(), YOLO(model_path).names):
            raise ValueError(f"Invalid class ids. Must be comma-separated integers between 0 and {len(YOLO(model_path).names) - 1}.")
        if not check_mail_list_vadility(mail_list_entry.get()):
            raise ValueError(f"Invalid mail address(es). Must be comma-separated and contain valid mail formats.")

        config.update({
            "yolo_model_path": model_path,
            "mail_option": mail_option_combobox.get(),
            "source": source_entry.get(),
            "confidence_level": float(confidence_entry.get()),
            "video_length": float(videolen_entry.get()),
            "show_stream": show_stream_var.get(),
            "record_plot": record_plot_var.get(),
            "class_ids": [int(class_id) for class_id in class_id_entry.get().split(",")],
            "emails": [mail.strip() for mail in mail_list_entry.get().split(",")],
            "email_trigger_count": int(email_trigger_count_entry.get()),
            "mail_pause_in_sec": float(mail_pause_entry.get()),
            "time_interval_for_enough_frame_sightings_in_sec": float(time_interval_entry.get())
        })

        with open(PARAMETERS_FILE, "w") as f:                                                                                   # Save parameters in parameters.json.
            json.dump(config, f, indent=4)

        update_save_button_state()                                                                                              # Update the state of the Save button based on changes (Here: Disable the save button again as all the new parameters have been saved)
        update_video_source_url(urlparse(config["source"]))                                                                     # If the 'source' parameter is changed to point to another camera stream, we might need to reauthenticate to have access to that camera stream.

    except ValueError as e:
        messagebox.showerror("Error", str(e))

 ########################################################################################## UI Triggered Functions


def update_video_source_url(parsed_url):                                                                                        # This function checks if the 'source' parameter is a valid URL hostname. If yes, it updates the global variables `camera_url` and `auth` for accessing the camera stream. It also manages the visibility ofthe zoom slider and its label, depending on whether the URL is a valid hostname.

    global camera_url, auth

    if parsed_url.hostname:                                                                                                     # If a hostname is present in the parsed URL, update camera_url and auth for camera stream access
        camera_url = f"http://{parsed_url.hostname}/axis-cgi/opticscontrol.cgi"
        auth = requests.auth.HTTPDigestAuth(parsed_url.username, parsed_url.password)
        zoom_slider.grid()
    else:
        camera_url, auth = None, None
        zoom_slider.grid_remove()


def update_mail_option(event=None):                                                                                             # This function dynamically configures the state and text of widgets related to mail options in the GUI based on the selected mail option from the mail_option_combobox.

    current_option = mail_option_combobox.get()

    entries = [videolen_entry, email_trigger_count_entry, time_interval_entry, mail_pause_entry, class_id_entry, mail_list_entry]
    labels =  [videolen_label, email_trigger_count_label, time_interval_label, mail_pause_label, class_id_label, mail_list_label]
    buttons = [manual_start_button, manual_stop_button]

    labels[1].config(text="Nr. of Frames with Detection for Video Recording Start")                                             # Reset default configuration for widgets
    buttons[0].config(text="Start Video Recording Manually")
    buttons[1].config(text="Send Mail with Video")
    for entry in entries:
        entry.config(state="normal")
    for label in labels:
        label.config(foreground="black")
    for button in buttons:
        button.grid()

    if current_option == "No Mail":                                                                                             # Customize widget configuration based on the selected mail option
        for entry in entries:
            entry.config(state="disabled")
        for label in labels:
            label.config(foreground="gray")
        for button in buttons:
            button.grid_remove()
    
    elif current_option == "Mail without Attachment":
        labels[1].config(text="Nr. of Frames with Detection for Mail Trigger")
        labels[0].config(foreground="gray")
        entries[0].config(state="disabled")
        buttons[0].config(text="Send Mail Manually")
        buttons[1].grid_remove()
     
    elif current_option == "Mail with Image Attachment":
        labels[1].config(text="Nr. of Frames with Detection for Image Snapshot")
        labels[0].config(foreground="gray")
        entries[0].config(state="disabled")
        buttons[0].config(text="Send Mail with Image Manually")
        buttons[1].grid_remove()

    update_save_button_state()                                                                                                  # Update the state of the Save button based on changes


def update_class_id_list(event=None):                                                                                           # This function retrieves the selected YOLO model from the yolo_model_combobox and updates the class ID list in the GUI based on the selected  model.
    
    selected_model = yolo_model_combobox.get()
    model = YOLO(os.path.join("yolo_models", selected_model))
    listbox.delete(0, tk.END)                                                                                                   # Clear the existing items in the listbox
    listbox.insert(tk.END, *[f"{class_id}: {class_name}" for class_id, class_name in model.names.items()])                      # Insert class ID and name pairs into the listbox

    update_save_button_state()                                                                                                  # Update the state of the Save button based on changes


def update_zoom_level(zoom_level):                                                                                              # This function constructs a payload with the desired zoom level and sends a POST request to the Axis camera's optics control endpoint. It uses the provided camera_url and auth for authentication. In case of a timeout or a general request exception, appropriate error messages are printed.

    payload = {
        "apiVersion": "1",
        "context": "Axis library",
        "method": "setMagnification",
        "params": {
            "optics": [
                {
                    "opticsId": 0,
                    "magnification": zoom_level                                                                                 # Update the zoom level of an Axis camera using the specified magnification value.
                }
            ]
        }   
    }
    try:
        response = requests.post(camera_url, json=payload, auth=auth, timeout=2)                                                # If a wrong URL was given, it will lead to a timeout, if there are other connection issues with the camera despite a correct URL, a Request Exception will be triggered.
    except requests.exceptions.Timeout as t:
        print("Timeout Error:", t)
    except requests.exceptions.RequestException as e:
        print("Request Exception:", e, "re")
   

def on_slider_release(event):                                                                                                   # Callback function triggered when the zoom slider is released.
    update_zoom_level(zoom_slider.get())


def manual_start_button_pressed():                                                                                              # Callback function triggered when the manual start button is pressed.
    
    global man_start_but_press
    man_start_but_press = True                                                                                                  # global variable used in the run_object_detection() function to know when a manual action should be triggered (send  mail without/image attachment or start recording video).
    if mail_option_combobox.get() == "Mail with Video Attachment":                                                              # If the mail option is "Mail with Video Attachment", it enables the manual stop button and disables the manual start button after the manual start button is pressed
        manual_stop_button.config(state=tk.NORMAL)
        manual_start_button.config(state=tk.DISABLED)


def manual_stop_button_pressed():
    
    global man_stop_but_press                                                                                                   # Callback function triggered when the manual stop button is pressed.
    man_stop_but_press = True                                                                                                   # global variable used in the run_object_detection() function to know when the manual video recording should be stopped and a mail with video attachment sent.
    manual_stop_button.config(state=tk.DISABLED)
    manual_start_button.config(state=tk.NORMAL)


def start_object_detection():                                                                                                   # Callback function that starts the object detection process and is triggered when the start button is pressed

    global cap
    cap = cv2.VideoCapture(int(load_parameters()["source"]) if load_parameters()["source"].isdigit() else load_parameters()["source"])  # Initialize the cv2 video capture using the source specified in the parameters.
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    manual_start_button.config(state=tk.NORMAL)
    save_button.config(state=tk.DISABLED)
    thread = threading.Thread(target=run_object_detection, name='run_object_detection')                                         # Start a new thread to run the live object detection process concurrently.
    thread.start()


def stop_object_detection():                                                                                                    # Callback function that stops the object detection process and is triggered when the stop button is pressed

    global cap
    cap.release() if cap is not None and cap.isOpened() else None                                                               # Release the cv2 video capture if it is still open and not None. 
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    manual_start_button.config(state=tk.DISABLED)

    update_save_button_state()
    save_plot_data()                                                                                                            # As we only save the datapoints all x seconds in run_object_detection(), there will quite probably be some datapoints left in 'detected_class_ids_list, total_times_list, email_sent_flags' when stopping the programming that need to be added to plot_data.csv.

 ########################################################################################## TKINTER GUI


def closing_window_handle():                                                                                                    # Function to handle the window closure event
    root.destroy()                                                                                                              # Destroy the Tkinter window
    plt.close('all')                                                                                                            # Close all open matplotlib windows


def load_parameters():                                                                                                          # Load parameters from the cparameters.json file.
    with open(PARAMETERS_FILE, "r") as f:
        return json.load(f)


root = tk.Tk()
root.title("Live Object Detection UI")
style = ttk.Style()
style.theme_use("clam")
root.protocol("WM_DELETE_WINDOW", closing_window_handle)
root.columnconfigure(0, minsize=400)
for row in range(root.grid_size()[1]):
    root.rowconfigure(row, min=45)
config = load_parameters()

# Confidence Level
Label(root, text="Confidence Level:").grid(row=0, column=0, padx=10, pady=10)
confidence_entry = Entry(root)
confidence_entry.insert(0, config["confidence_level"])
confidence_entry.grid(row=0, column=1, padx=10, pady=10)
confidence_entry.bind("<KeyRelease>", update_save_button_state) 

# Source
Label(root, text="Videostream Source:").grid(row=1, column=0, padx=10, pady=10)
source_entry = Entry(root)
source_entry.insert(0, config["source"])
source_entry.grid(row=1, column=1, padx=10, pady=10)
source_entry.bind("<KeyRelease>", update_save_button_state)  

#Attachment Option
Label(root, text="Mail Options:").grid(row=2, column=0, padx=10, pady=10)
mail_option_combobox = Combobox(root, values=["No Mail", "Mail without Attachment", "Mail with Image Attachment", "Mail with Video Attachment"], state='readonly')
mail_option_combobox.set(config["mail_option"])
mail_option_combobox.grid(row=2, column=1, padx=10, pady=10)
mail_option_combobox.bind("<<ComboboxSelected>>", update_mail_option)

# Email Trigger Count
email_trigger_count_label = Label(root, text="Number of Frames with Detection for Mail Trigger:")
email_trigger_count_label.grid(row=3, column=0, padx=10, pady=10)
email_trigger_count_entry = Entry(root)
email_trigger_count_entry.insert(0, config["email_trigger_count"])
email_trigger_count_entry.grid(row=3, column=1, padx=10, pady=10)
email_trigger_count_entry.bind("<KeyRelease>", update_save_button_state)  

# Time Interval
time_interval_label = Label(root, text="Time To be Detected Interval (sec):")
time_interval_label.grid(row=4, column=0, padx=10, pady=10)
time_interval_entry = Entry(root)
time_interval_entry.insert(0, config["time_interval_for_enough_frame_sightings_in_sec"])
time_interval_entry.grid(row=4, column=1, padx=10, pady=10)
time_interval_entry.bind("<KeyRelease>", update_save_button_state)  

# Mail Pause
mail_pause_label = Label(root, text="No Mail Sending Interval (sec):")
mail_pause_label.grid(row=5, column=0, padx=10, pady=10)
mail_pause_entry = Entry(root)
mail_pause_entry.insert(0, config["mail_pause_in_sec"])
mail_pause_entry.grid(row=5, column=1, padx=10, pady=10)
mail_pause_entry.bind("<KeyRelease>", update_save_button_state)  

# Video Length
videolen_label = Label(root, text="Video Length (sec):")
videolen_label.grid(row=6, column=0, padx=10, pady=10)
videolen_entry = Entry(root)
videolen_entry.insert(0, config["video_length"])
videolen_entry.grid(row=6, column=1, padx=10, pady=10)
videolen_entry.bind("<KeyRelease>", update_save_button_state)  

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
class_id_entry.bind("<KeyRelease>", update_save_button_state)  

# Class IDs
mail_list_label = Label(root, text="Recipient Emails List (comma-separated):")
mail_list_label.grid(row=7, column=2, padx=10, pady=10)
mail_list_entry = Entry(root)
mail_list_entry.insert(0, ",".join(map(str, config["emails"])))
mail_list_entry.grid(row=7, column=3, padx=10, pady=10)
mail_list_entry.bind("<KeyRelease>", update_save_button_state)  

# Scrollbar for Listbox
scrollbar = Scrollbar(root, orient=tk.VERTICAL)
scrollbar.grid(row=0, column=4, padx=0, pady=10, rowspan=6)
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

# Paramter Save Button
save_button = Button(root, text="Save Parameters", command=save_parameters, state='disabled')
save_button.grid(row=9, column=3, columnspan=2, pady=20)

# Program Start Button
start_button = Button(root, text=" START ", command=start_object_detection, state=tk.NORMAL)
start_button.grid(row=9, column=2, pady=10, padx=10)

# Program Stop Button
stop_button = Button(root, text=" STOP  ", command=stop_object_detection, state=tk.DISABLED)
stop_button.grid(row=9, column=2, pady=10,  padx=(180,10))

# Plot Button
plot_button = Button(root, text="Show Plot", command=show_plot_data, state=tk.NORMAL)
plot_button.grid(row=7, column=0, pady=10,  padx=10)

# Manual Mail Button Left
manual_start_button = Button(root, text="Start Video Recording Manually", command=manual_start_button_pressed, state=tk.DISABLED)
manual_start_button.grid(row=9, column=0, padx=10, pady=10)

# Manual Mail Button Right
manual_stop_button = Button(root, text="Send Mail with Video", command=manual_stop_button_pressed, state=tk.DISABLED)
manual_stop_button.grid(row=9, column=1, padx=10, pady=10)

# Show Videostream Checkbox
show_stream_var = tk.BooleanVar(value=bool(config["show_stream"]))
show_stream_checkbox = Checkbutton(root, text="Show Live Stream", variable=show_stream_var)
show_stream_checkbox.grid(row=7, column=1, pady=10, padx=10)
show_stream_checkbox.config(command=update_save_button_state)

# Record Plot Checkbox
record_plot_var = tk.BooleanVar(value=bool(config["record_plot"]))
record_plot_checkbox = Checkbutton(root, text="Record New Plot", variable=record_plot_var)
record_plot_checkbox.grid(row=7, column=0, pady=10, padx=(240,10))
record_plot_checkbox.config(command=update_save_button_state)

# Camera Zoom Slider
zoom_slider = Scale(root, from_=1, to=2, resolution=0.01, orient=tk.VERTICAL, length=200)
zoom_slider.set(1.0)  # Set the initial value
zoom_slider.grid(row=11, column=3, padx=10, pady=10)
zoom_slider.bind("<ButtonRelease-1>", on_slider_release)
parsed_url = urlparse(config["source"])
update_video_source_url(parsed_url)
if parsed_url.hostname:
    update_zoom_level(1.0)

# Error Message
error_label = Label(root, text="", fg="red")
error_label.grid(row=10, column=0, columnspan=4)

# Video Canvas
video_frame = tk.Frame(root)
video_frame.grid(row=11, column=0, columnspan=5, pady=10)
canvas = tk.Canvas(video_frame)
canvas.grid(row=0, column=0)

update_class_id_list()
update_mail_option()
root.mainloop()

 ########################################################################################## NOTES

# 'source' for the Axis Camera I was using: rtsp://root:pass@192.168.1.190/axis-media/media.amp
