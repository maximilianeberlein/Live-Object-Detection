import json
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import ttk  # Import the ttk module
from tkinter import Checkbutton
from tkinter.ttk import Combobox
from tkinter import Label, Entry, Button, Listbox, Scrollbar, messagebox, Scale
import os
import subprocess
import requests
from requests.exceptions import RequestException
from urllib.parse import urlparse
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
import main_program

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

    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    #plot_button.config(state=tk.DISABLED)

    # Check if the script is already running
    if hasattr(start_program, "process") and start_program.process.poll() is None:
        messagebox.showinfo("Info", "The script is already running.")
        return

    # Start the main_program.py
    main_program_path = os.path.join(os.path.dirname(__file__), "main_program.py")
    start_program.process = subprocess.Popen(["python3", main_program_path])


# Function to stop the main_program.py
def stop_program():

    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
   # plot_button.config(state=tk.NORMAL)
    # Check if the script is running
    if hasattr(start_program, "process") and start_program.process.poll() is None:
        main_program.stop_actions()
        start_program.process.terminate()
    else:
        messagebox.showinfo("Info", "The script is currently not running.")

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

    legend_text = False
    for time_point, email_flag in zip(total_times_seconds, email_flags):
        if email_flag == 1:
            label = 'Email Sent' if email_flag == 1 and not legend_text else ''
            plt.axvline(x=time_point, color='black', linestyle='--', alpha=0.5, label=label)
            legend_text = True
    
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
    
    elif current_option == "Mail without Attachment":
        email_trigger_count_label.config(text="Nr. of Frames with Detection for Mail Trigger")
        videolen_label.grid_remove()
        videolen_entry.grid_remove()
     
    elif current_option == "Mail with Image Attachment":
        email_trigger_count_label.config(text="Nr. of Frames with Detection for Image Snapshot")
        videolen_label.grid_remove()
        videolen_entry.grid_remove()

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
        response = requests.post(camera_url, json=payload, auth=auth)
    except RequestException as e:
        print("Request Exception:", e)
   

def on_slider_release(event):
    update_zoom_level(zoom_slider.get())

def on_closing():
    # Function to handle the window closure event
    root.destroy()  # Destroy the Tkinter window
    plt.close('all')  # Close all matplotlib windows

# Load default configuration from JSON file
config_file = Path("parameters.json")

with open(config_file, "r") as f:
    config = json.load(f)

parsed_url = urlparse(config["source"])
camera_url = f"http://{parsed_url.hostname}/axis-cgi/opticscontrol.cgi"  # Replace with the actual URL
auth = requests.auth.HTTPDigestAuth(parsed_url.username, parsed_url.password)

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
#zoom_slider.config(command=lambda value: update_zoom_level(zoom_slider.get()))

update_zoom_level(1.0)
update_class_id_list()
attachment_changed()


root.mainloop()
