import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import time
import os
import dlib

class ImageGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detection and Matching")
        self.master.geometry("950x580")

        # Variables
        self.original_image = None
        self.current_file_path = ""
        
        # Get the path to the models folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        # Paths for the OpenCV face detector files
        self.prototxt_path = os.path.join(models_dir, "opencv_face_detector.prototxt")
        self.model_path = os.path.join(models_dir, "opencv_face_detector.caffemodel")
        # Stop the program early if either model file is missing
        if not os.path.exists(self.prototxt_path):
            raise FileNotFoundError(f"OpenCV prototxt not found: {self.prototxt_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"OpenCV caffemodel not found: {self.model_path}")
        # Load the pretrained OpenCV DNN face detector
        self.face_net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        
        # Path for dlib 5-point facial landmark model
        self.landmark_model_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
        # Stop the program early if landmark model file is missing
        if not os.path.exists(self.landmark_model_path):
            raise FileNotFoundError(f"dlib landmark model not found: {self.landmark_model_path}")
        # Load pretrained dlib 5-point landmark predictor
        self.landmark_predictor = dlib.shape_predictor(self.landmark_model_path)
        
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
        # Convert RGB image to HSV colour space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

        # Lower and upper HSV thresholds for skin-coloured pixels
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 170, 255], dtype=np.uint8)
        
        # Create binary mask where skin-coloured pixels are white
        skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
        
        # Clean up small noise and fill small gaps
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    def detect_faces(self, image_array):
        # Create skin mask for the whole image
        skin_mask = self.get_skin_mask(image_array)
        # Copy image so rectangles can be drawn on it
        output_image = image_array.copy()
        
        # Lists for debugging and final accepted faces
        raw_faces = []
        valid_faces = []
        debug_info = []
        
        # Get original image size
        image_height, image_width = image_array.shape[:2]
        # Prepare image for OpenCV DNN face detector
        blob = cv2.dnn.blobFromImage(cv2.resize(image_array, (300, 300)), scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        # Run face detector
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # Loop through all detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Ignore weak detections
            if confidence < 0.5:
                continue
            
            # Convert normalized box coordinates back to original image coordinates
            box = detections[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
            start_x, start_y, end_x, end_y = box.astype("int")

            # Clamp coordinates so they stay inside the image
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(image_width - 1, end_x)
            end_y = min(image_height - 1, end_y)
            # Compute width and height of the detection box
            box_width = end_x - start_x
            box_height = end_y - start_y

            # Skip invalid boxes
            if box_width <= 0 or box_height <= 0:
                continue

            # Store raw detection before skin filtering
            raw_faces.append((start_x, start_y, box_width, box_height, confidence))
            # Measure how much of the detected region looks like skin
            face_skin_region = skin_mask[start_y:end_y, start_x:end_x]
            skin_ratio = np.count_nonzero(face_skin_region) / (box_width * box_height)
            # Save debug information
            debug_info.append((start_x, start_y, box_width, box_height, confidence, skin_ratio))
            # Use skin-colour segmentation as a secondary check to help reject unlikely face detections
            if skin_ratio > 0.05:
                valid_faces.append((start_x, start_y, box_width, box_height, confidence))
                cv2.rectangle(output_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return output_image, raw_faces, valid_faces, debug_info
    
    def detect_landmarks(self, image_array, valid_faces):
        # List to hold landmark results for each valid face
        landmarks_per_face = []
        
        # Loop through each valid detected face and run dlib landmark detection on it
        for (start_x, start_y, box_width, box_height, confidence) in valid_faces:
            end_x = start_x + box_width
            end_y = start_y + box_height

            # Create dlib rectangle from detected face box
            face_rect = dlib.rectangle(start_x, start_y, end_x, end_y)
            # Run landmark detection only inside this face region
            shape = self.landmark_predictor(image_array, face_rect)

            # dlib 5-point model:
            # points 0-1 = one eye corners
            # points 2-3 = other eye corners
            # point 4 = nose point
            points = []
            for i in range(5):
                points.append((shape.part(i).x, shape.part(i).y))

            # Compute eye centres from the two corner points for each eye
            right_eye_centre = (
                (points[0][0] + points[1][0]) // 2,
                (points[0][1] + points[1][1]) // 2
            )
            left_eye_centre = (
                (points[2][0] + points[3][0]) // 2,
                (points[2][1] + points[3][1]) // 2
            )

            # Original 5-point nose
            nose_tip = points[4]

            # Move nose slightly upward to correct the low placement
            nose_offset = int(box_height * 0.04)
            nose_tip = (nose_tip[0], nose_tip[1] - nose_offset)

            landmarks_per_face.append({
                "box": (start_x, start_y, box_width, box_height, confidence),
                "right_eye": right_eye_centre,
                "left_eye": left_eye_centre,
                "nose": nose_tip
            })

        return landmarks_per_face
    
    def draw_landmarks(self, image_array, landmarks_per_face):
        output_image = image_array.copy()

        # Process the landmarks for each face and draw them on the output image
        for face_data in landmarks_per_face:
            right_eye = face_data["right_eye"]
            left_eye = face_data["left_eye"]
            nose = face_data["nose"]

            # Draw circles for the 3 required landmarks
            cv2.circle(output_image, right_eye, 4, (255, 0, 0), -1)   # blue
            cv2.circle(output_image, left_eye, 4, (0, 255, 0), -1)    # green
            cv2.circle(output_image, nose, 4, (0, 0, 255), -1)        # red

        return output_image
    
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
        original_array = np.array(self.original_image)

        # Display original image on left
        display_input = self.resize_image(self.original_image)
        input_photo = ImageTk.PhotoImage(display_input)
        self.image_label.configure(image=input_photo)
        self.image_label.image = input_photo

        # Face detection + landmark timing
        start_time = time.time()
        detected_image, raw_faces, valid_faces, debug_info = self.detect_faces(original_array)
        landmarks_per_face = self.detect_landmarks(original_array, valid_faces)
        detected_image = self.draw_landmarks(detected_image, landmarks_per_face)
        end_time = time.time()

        # Convert detected result back to PIL for display
        detected_pil = Image.fromarray(detected_image)
        display_output = self.resize_image(detected_pil)
        output_photo = ImageTk.PhotoImage(display_output)

        # Show processed image on right
        self.filtered_label.configure(image=output_photo)
        self.filtered_label.image = output_photo
        
        # Status messages
        self.message_1.configure(text=f"{len(valid_faces)} face(s) detected.")
        self.message_2.configure(text=f"Processing time: {end_time - start_time:.3f} seconds")
        
        # Debug info
        image_name = os.path.basename(self.current_file_path)
        print(f"\nImage: {image_name}")
        print("Raw faces:", len(raw_faces))
        for item in debug_info:
            x, y, w, h, confidence, skin_ratio = item
            print(
                f"Box {(x, y, w, h)} "
                f"confidence={confidence:.3f} "
                f"skin_ratio={skin_ratio:.3f}"
            )
        print("Landmarks:")
        for face_data in landmarks_per_face:
            print(
                f"right_eye={face_data['right_eye']} "
                f"left_eye={face_data['left_eye']} "
                f"nose={face_data['nose']}"
            )
        print("-" * 50)

    def bulk_processing(self):
        self.message_1.configure(text="Bulk processing not implemented yet.")
        self.message_2.configure(text="")

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()