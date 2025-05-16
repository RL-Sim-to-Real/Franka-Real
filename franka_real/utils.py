import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json


def list_available_cameras():
    index = 0
    available_cameras = []
    while index < 10:  # limit to avoid hanging
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
        index += 1
    return available_cameras


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Viewer")
        
        self.available_cameras = list_available_cameras()
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras found!")
            self.root.destroy()
            return
        
        self.selected_camera = tk.IntVar(value=self.available_cameras[0])
        
        # Dropdown for camera selection
        self.camera_label = ttk.Label(root, text="Select Camera:", font=("Arial", 24))
        self.camera_label.pack(pady=5)
        
        self.camera_dropdown = ttk.Combobox(root, values=self.available_cameras, textvariable=self.selected_camera, font=("Arial", 16), width=20)
        self.camera_dropdown.option_add("*TCombobox*Listbox*Font", ("Arial", 16))
        self.camera_dropdown.pack(pady=5)
        
        # Button to start camera feed
        self.start_button = ttk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.config(width=20, padding=(10, 10))
        self.start_button.pack(pady=10)
        # Canvas for video feed
        self.video_canvas = tk.Canvas(root, width=640, height=480)
        self.video_canvas.pack(pady=10)

        self.recording = True
        self.out = None
        
        self.cap = None
        self.running = False

    def start_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.selected_camera.get())
        # Lower camera exposure
        self.running = True
        self.update_frame()

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = cv2.resize(frame, (64, 64))
                frame = cv2.resize(frame, (640, 480))
                if self.recording:
                    if self.out is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = cv2.imencode('.ppm', img)[1].tobytes()
                self.photo = tk.PhotoImage(data=img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.root.after(10, self.update_frame)

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.stop_camera()
    
    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_camera)
    root.mainloop()