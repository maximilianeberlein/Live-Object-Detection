import json
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import ttk  # Import the ttk module
from tkinter.ttk import Combobox
from tkinter import Label, Entry, Button, Listbox, Scrollbar, messagebox
import os
import subprocess

def is_save_needed():
    # Check if any parameter has been changed
    try:
        error_label.config(text="")
        return (
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
    # Check if the script is already running
    if hasattr(start_program, "process") and start_program.process.poll() is None:
        messagebox.showinfo("Info", "The script is already running.")
        return

    # Start the script
    script_path = os.path.join(os.path.dirname(__file__), "main_program.py")
    start_program.process = subprocess.Popen(["python3", script_path])

# Function to stop the main_program.py
def stop_program():

    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    # Check if the script is running
    if hasattr(start_program, "process") and start_program.process.poll() is None:
        start_program.process.terminate()
    else:
        messagebox.showinfo("Info", "The script is currently not running.")


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
        
        if videolen_val >= mail_pause_val:
            raise ValueError("Video Length must be smaller than No Mail Sending Interval.")


        selected_model = yolo_model_combobox.get()
        model = YOLO(os.path.join("yolo_models", selected_model))
        config["yolo_model_path"] = os.path.join("yolo_models", selected_model)

        mail_option = mail_option_combobox.get()
        config["mail_option"] = mail_option

        config["source"] = source_entry.get()
        config["confidence_level"] = confidence_level_val
        config["video_length"] = videolen_val

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

# Load default configuration from JSON file
config_file = Path("parameters.json")

with open(config_file, "r") as f:
    config = json.load(f)

# Create a simple tkinter GUI
root = tk.Tk()
root.title("Alascom Object Detection")
style = ttk.Style()
style.theme_use("clam")

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
save_button.grid(row=8, column=0, columnspan=2, pady=20)

error_label = Label(root, text="", fg="red")
error_label.grid(row=9, column=0, columnspan=4)

# Start Button
start_button = Button(root, text="Start Detection", command=start_program, state=tk.NORMAL)
start_button.grid(row=8, column=2, pady=10, padx=10)

# Stop Button
stop_button = Button(root, text="Stop Detection", command=stop_program, state=tk.DISABLED)
stop_button.grid(row=8, column=2, pady=10,  padx=(280,10))

update_class_id_list()
attachment_changed()


root.mainloop()
