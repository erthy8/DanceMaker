import tkinter as tk
import h5py
from tkinter import filedialog
from tkinter import font
from tkinter import ttk
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from ultralytics import YOLO
import pandas as pd
import pytube
import os
import re
from videodata import VideoPoser
from livedata import get_live_data
from postprocess import postProcessScore


def select_file_and_save(data):
    # Open a file dialog window to select a file location for saving
    initial_dir = os.path.join(os.path.expanduser(
        '~'), 'data', "final-new", 'videodata.hdf5')
    file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[
                                             ("HDF5 files", "*.h5"), ("All files", "*.*")])

    # Check if a file location was specified
    if file_path:
        # Save the data to the specified file location
        with h5py.File(file_path, 'w') as hf:
            # Create a dataset in the HDF5 file and write the data
            hf.create_dataset('data', data=data)
        print("Data saved to:", file_path)


def start_processing():
    label.pack_forget()
    entry.pack_forget()
    select_file_button.pack_forget()
    start_button.pack_forget()
    # Show loading screen
    base_options = python.BaseOptions(
        model_asset_path='models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    yolo = YOLO("YOLOv9/best.pt")
    processor = VideoPoser(detector, yolo)

    if "youtube" in entry.get():
        youtube_url = entry.get()
        video = pytube.YouTube(youtube_url)
        processor(video)
        live_processing()
    else:
        live_processing()  # this doesnt work
    root.destroy()


def process(label):
    # After processing is complete, switch to the "Are you ready to dance?" screen
    label.pack_forget()
    show_dance_message()


def live_processing():
    title = ""
    with open('logs.txt', 'r') as file:
        # Iterate over lines
        for line in file:
            title = line
            break  # Break after the first iteration
    df_dance = pd.read_hdf(
        f'./data/final-{title}/videodata.hdf5', key='videodata')
    get_live_data(title, df_dance)


def show_dance_message():
    # Create a label and button for dancing
    score = postProcessScore()
    ready_label = tk.Label(
        root, text=f"Score: {score}", bg="#2E2E2E", fg="white", font=comic_sans_font)
    ready_label.pack(pady=(window_height-70)//2)  # Center vertically
    ready_label.pack_configure(anchor="center")

    ready_button = ttk.Button(
        root, text="Ready", command=live_processing(), style="RoundedPink.TButton")
    ready_button.pack(pady=5)  # Add padding
    ready_button.pack_configure(anchor="center")


def is_valid_youtube_url(url):
    # Regular expression to match YouTube URL format
    youtube_regex = (
        r"(https?://)?(www\.)?"
        "(youtube|youtu|youtube-nocookie)\.(com|be)/"
        "(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )
    return re.match(youtube_regex, url)


def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)


# Create the main window
root = tk.Tk()
root.title("Dance-themed Input GUI")
root.configure(bg="#2E2E2E")  # Dark gray background

# Set window size and center the window
window_width = 500
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Define Comic Sans font
comic_sans_font = font.Font(family="Comic Sans MS", size=12)

# Create a label
label = tk.Label(root, text="Enter a YouTube URL or select an HDF5 file:",
                 bg="#2E2E2E", fg="white", font=comic_sans_font)
label.pack(pady=10)

# Create a string input field
entry = tk.Entry(root, font=comic_sans_font)
entry.pack(pady=5)

# Create a button to select HDF5 file
style = ttk.Style()
style.configure("RoundedPink.TButton", borderwidth=0)
style.map("RoundedPink.TButton", foreground=[
          ('pressed', "#2E2E2E"), ('active', "#2E2E2E")])
style.configure("RoundedGreen.TButton", borderwidth=0)
style.map("RoundedGreen.TButton", foreground=[
          ('pressed', "#2E2E2E"), ('active', "#2E2E2E")])
select_file_button = ttk.Button(
    root, text="Select HDF5 File", command=select_file_and_save, style="RoundedPink.TButton")
select_file_button.pack(pady=5)

start_button = ttk.Button(
    root, text="Start", command=start_processing, style="RoundedPink.TButton")
start_button.pack(pady=5)


# Run the Tkinter event loop
root.mainloop()

score = postProcessScore()


# MAKE A NEW GUI
root2 = tk.Tk()
root2.title("Dance-themed Input GUI")
root2.configure(bg="#2E2E2E")  # Dark gray background

# Set window size and center the window
window_width = 500
window_height = 300
screen_width = root2.winfo_screenwidth()
screen_height = root2.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root2.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Define Comic Sans font
comic_sans_font = font.Font(family="Comic Sans MS", size=12)

# Create a label
label = tk.Label(root2, text=f"Score = {round((score*100), 2)}",
                 bg="#2E2E2E", fg="white", font=comic_sans_font)
label.pack(pady=10)


# Run the Tkinter event loop
root2.mainloop()
