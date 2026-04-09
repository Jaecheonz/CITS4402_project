import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import time

class ImageGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detection and Matching")
        self.master.geometry("950x580")

        # Variables
        self.original_image = None
        self.original_array = None
        self.current_file_path = ""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Main frame
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill="both", expand=True, padx=4, pady=4)
        # Border frame
        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.pack(fill="both", expand=True, padx=4, pady=4)

        # Image frame
        self.image_frame = tk.Frame(self.border)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        # Left image
        self.image_label = tk.Label(self.image_frame, bg="lightgray")
        self.image_label.grid(row=0, column=0, padx=20, pady=5)
        # Right image
        self.filtered_label = tk.Label(self.image_frame, bg="lightgray")
        self.filtered_label.grid(row=0, column=1, padx=20, pady=5)
        # Labels under images
        self.input_text = tk.Label(self.image_frame, text="Input Image")
        self.input_text.grid(row=1, column=0, pady=5)
        self.output_text = tk.Label(self.image_frame, text="Processed Image")
        self.output_text.grid(row=1, column=1, pady=5)

        # Controls frame
        self.controls_frame = tk.Frame(self.border)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10)
        # Message labels
        self.message_1 = tk.Label(self.controls_frame, text="No image loaded.")
        self.message_1.grid(row=0, column=0, columnspan=5, pady=(5, 5))
        self.message_2 = tk.Label(self.controls_frame, text="")
        self.message_2.grid(row=1, column=0, columnspan=5, pady=(0, 10))
        # Single Image button
        self.load_button = tk.Button(self.controls_frame, text="Single Image", command=self.load_image)
        self.load_button.grid(row=2, column=1, padx=20, pady=5)
        # Bulk Processing button
        self.bulk_button = tk.Button(self.controls_frame, text="Bulk Processing", command=self.bulk_processing)
        self.bulk_button.grid(row=2, column=3, padx=20, pady=5)

    def resize_image(self, pil_image):
        # Resize image to fit the GUI while keeping aspect ratio
        width, height = pil_image.size
        max_width = 430
        max_height = 260

        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return pil_image.resize((new_width, new_height))

    def get_skin_mask(self, image_array):
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 170, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    def detect_faces(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        skin_mask = self.get_skin_mask(image_array)

        raw_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        output = image_array.copy()
        valid_faces = []
        debug_info = []

        for (x, y, w, h) in raw_faces:
            face_skin = skin_mask[y:y+h, x:x+w]
            skin_ratio = np.count_nonzero(face_skin) / (w * h)
            debug_info.append((x, y, w, h, skin_ratio))

            if skin_ratio > 0.2:
                valid_faces.append((x, y, w, h))
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output, raw_faces, valid_faces, debug_info, skin_mask
    
    def load_image(self):
        # Open file dialog to choose an image
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.current_file_path = file_path

        # Load image using PIL
        self.original_image = Image.open(file_path).convert("RGB")
        self.original_array = np.array(self.original_image)

        # Display original image on left
        display_input = self.resize_image(self.original_image)
        input_photo = ImageTk.PhotoImage(display_input)
        self.image_label.configure(image=input_photo)
        self.image_label.image = input_photo

        # Face detection timing
        start_time = time.time()
        detected_image, raw_faces, valid_faces, debug_info, skin_mask = self.detect_faces(self.original_array)
        end_time = time.time()

        # Convert detected result back to PIL for display
        detected_pil = Image.fromarray(detected_image)
        display_output = self.resize_image(detected_pil)
        output_photo = ImageTk.PhotoImage(display_output)

        # Show processed image on right
        self.filtered_label.configure(image=output_photo)
        self.filtered_label.image = output_photo
        print("Raw faces:", len(raw_faces))
        for item in debug_info:
            x, y, w, h, skin_ratio = item
            print(f"Box {(x, y, w, h)} skin_ratio={skin_ratio:.3f}")
        # Status messages
        self.message_1.configure(text=f"{len(valid_faces)} face(s) detected.")
        self.message_2.configure(text=f"Processing time: {end_time - start_time:.3f} seconds")

    def bulk_processing(self):
        self.message_1.configure(text="Bulk processing not implemented yet.")
        self.message_2.configure(text="")

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()