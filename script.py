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

            nose_x = nose_tip[0]
            nose_y = nose_tip[1]

            # Midpoint between the two eye centres
            eye_mid_x = (right_eye_centre[0] + left_eye_centre[0]) // 2

            # Distance between the eyes
            eye_distance = abs(left_eye_centre[0] - right_eye_centre[0])

            # Measure how tilted the face is using the eye vertical difference
            eye_vertical_diff = abs(left_eye_centre[1] - right_eye_centre[1])

            # Default values for more frontal faces
            nose_offset_ratio = 0.04
            side_shift_ratio = 0.20

            # If the face is noticeably tilted, reduce the heuristic strength
            if eye_vertical_diff > int(eye_distance * 0.12):
                nose_offset_ratio = 0.02
                side_shift_ratio = 0.30

            # Apply upward correction
            nose_offset = int(box_height * nose_offset_ratio)
            nose_y = nose_y - nose_offset

            # Limit left-right drift from the eye midpoint
            max_side_shift = int(eye_distance * side_shift_ratio)

            if nose_x < eye_mid_x - max_side_shift:
                nose_x = eye_mid_x - max_side_shift
            elif nose_x > eye_mid_x + max_side_shift:
                nose_x = eye_mid_x + max_side_shift

            nose_tip = (nose_x, nose_y)

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
    
    def align_and_crop_face(self, image_array, face_data):
        eye_a = face_data["right_eye"]
        eye_b = face_data["left_eye"]

        # Sort eyes by x-position so left-side eye is always first
        if eye_a[0] < eye_b[0]:
            left_eye_img = eye_a
            right_eye_img = eye_b
        else:
            left_eye_img = eye_b
            right_eye_img = eye_a

        left_eye_img = np.array(left_eye_img, dtype=np.float32)
        right_eye_img = np.array(right_eye_img, dtype=np.float32)

        # Eye direction
        dx = right_eye_img[0] - left_eye_img[0]
        dy = right_eye_img[1] - left_eye_img[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Eye distance scaling
        src_eye_distance = np.sqrt(dx * dx + dy * dy)
        dst_eye_distance = 85 - 40

        if src_eye_distance < 1:
            return np.zeros((125, 125, 3), dtype=np.uint8), None

        scale = dst_eye_distance / src_eye_distance

        # Midpoint between the eyes in source image
        eyes_center_src = (
            (left_eye_img[0] + right_eye_img[0]) / 2.0,
            (left_eye_img[1] + right_eye_img[1]) / 2.0
        )

        # Midpoint between target eye locations
        eyes_center_dst = (
            (40 + 85) / 2.0,
            40
        )

        # Rotation + scale matrix
        transform_matrix = cv2.getRotationMatrix2D(eyes_center_src, angle, scale)

        # Translate so eye midpoint lands in the target position
        transform_matrix[0, 2] += eyes_center_dst[0] - eyes_center_src[0]
        transform_matrix[1, 2] += eyes_center_dst[1] - eyes_center_src[1]

        # Warp to aligned 125 x 125 portrait
        aligned_face = cv2.warpAffine(image_array, transform_matrix, (125, 125))

        return aligned_face, transform_matrix

    def draw_landmarks_on_aligned_face(self, aligned_face, face_data, transform_matrix):
        output_face = aligned_face.copy()

        # Fixed eye landmark positions in the aligned image
        left_eye = (40, 40)
        right_eye = (85, 40)

        # Transform the original-image nose into aligned portrait coordinates
        original_nose = face_data["nose"]
        nose_x = int(
            transform_matrix[0, 0] * original_nose[0] +
            transform_matrix[0, 1] * original_nose[1] +
            transform_matrix[0, 2]
        )
        nose_y = int(
            transform_matrix[1, 0] * original_nose[0] +
            transform_matrix[1, 1] * original_nose[1] +
            transform_matrix[1, 2]
        )

        # Optional safety clamp so rare bad noses do not go too far off in the portrait
        nose_x = max(53, min(73, nose_x))
        nose_y = max(60, min(80, nose_y))

        nose = (nose_x, nose_y)

        cv2.circle(output_face, right_eye, 4, (255, 0, 0), -1)   # blue
        cv2.circle(output_face, left_eye, 4, (0, 255, 0), -1)    # green
        cv2.circle(output_face, nose, 4, (0, 0, 255), -1)        # red

        return output_face
    
    def place_faces_in_corners(self, image_array, aligned_faces):
        output_image = image_array.copy()
        image_height, image_width = output_image.shape[:2]

        # Corner positions for 125 x 125 aligned faces
        corner_positions = [
            (0, 0),                                   # top-left
            (image_width - 125, 0),                  # top-right
            (0, image_height - 125),                 # bottom-left
            (image_width - 125, image_height - 125)  # bottom-right
        ]

        for i in range(min(len(aligned_faces), 4)):
            x, y = corner_positions[i]
            output_image[y:y+125, x:x+125] = aligned_faces[i]
            cv2.rectangle(output_image, (x, y), (x + 124, y + 124), (255, 255, 255), 2)

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

        # Sort faces from left to right
        landmarks_per_face.sort(key=lambda face_data: face_data["box"][0])

        # Draw landmarks on the original image
        detected_image = self.draw_landmarks(detected_image, landmarks_per_face)

        # Align each detected face to 125 x 125 and draw landmarks on it
        aligned_faces = []
        for face_data in landmarks_per_face:
            aligned_face, transform_matrix = self.align_and_crop_face(original_array, face_data)
            aligned_face = self.draw_landmarks_on_aligned_face(aligned_face, face_data, transform_matrix)
            aligned_faces.append(aligned_face)

        # Paste aligned faces into the four corners of the output image
        detected_image = self.place_faces_in_corners(detected_image, aligned_faces)

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

    def process_single_image_to_file(self, file_path, output_dir):
            """Run the full detection pipeline on one image and save the result."""
            image = Image.open(file_path).convert("RGB")
            image_array = np.array(image)

            detected_image, _, valid_faces, debug_info = self.detect_faces(image_array)
            landmarks_per_face = self.detect_landmarks(image_array, valid_faces)

            landmarks_per_face.sort(key=lambda face_data: face_data["box"][0])

            detected_image = self.draw_landmarks(detected_image, landmarks_per_face)

            aligned_faces = []
            for face_data in landmarks_per_face:
                aligned_face, transform_matrix = self.align_and_crop_face(image_array, face_data)
                aligned_face = self.draw_landmarks_on_aligned_face(aligned_face, face_data, transform_matrix)
                aligned_faces.append(aligned_face)

            detected_image = self.place_faces_in_corners(detected_image, aligned_faces)

            # Debug output to console
            image_name = os.path.basename(file_path)
            print(f"\nImage: {image_name}")
            print("Raw detections debug:")
            for item in debug_info:
                x, y, w, h, confidence, skin_ratio = item
                print(
                    f"  Box {(x, y, w, h)} "
                    f"confidence={confidence:.3f} "
                    f"skin_ratio={skin_ratio:.3f}"
                )
            print("Landmarks:")
            for face_data in landmarks_per_face:
                print(
                    f"  right_eye={face_data['right_eye']} "
                    f"left_eye={face_data['left_eye']} "
                    f"nose={face_data['nose']}"
                )
            print("-" * 50)

            # Save result to output directory, preserving original filename
            output_path = os.path.join(output_dir, image_name)
            result_pil = Image.fromarray(detected_image)
            result_pil.save(output_path)

            return len(valid_faces)

    #Runs the full face detection process on one image but saves the results insteads of displaying it (for bulk process use only)
    def process_single_image_to_file(self, file_path, output_dir):
            image = Image.open(file_path).convert("RGB")
            image_array = np.array(image)

            detected_image, _, valid_faces, debug_info = self.detect_faces(image_array)
            landmarks_per_face = self.detect_landmarks(image_array, valid_faces)

            landmarks_per_face.sort(key=lambda face_data: face_data["box"][0])

            detected_image = self.draw_landmarks(detected_image, landmarks_per_face)

            aligned_faces = []
            for face_data in landmarks_per_face:
                aligned_face, transform_matrix = self.align_and_crop_face(image_array, face_data)
                aligned_face = self.draw_landmarks_on_aligned_face(aligned_face, face_data, transform_matrix)
                aligned_faces.append(aligned_face)

            detected_image = self.place_faces_in_corners(detected_image, aligned_faces)

            # Debug output to console
            image_name = os.path.basename(file_path)
            print(f"\nImage: {image_name}")
            print("Raw detections debug:")
            for item in debug_info:
                x, y, w, h, confidence, skin_ratio = item
                print(
                    f"  Box {(x, y, w, h)} "
                    f"confidence={confidence:.3f} "
                    f"skin_ratio={skin_ratio:.3f}"
                )
            print("Landmarks:")
            for face_data in landmarks_per_face:
                print(
                    f"  right_eye={face_data['right_eye']} "
                    f"left_eye={face_data['left_eye']} "
                    f"nose={face_data['nose']}"
                )
            print("-" * 50)

            # Save result to output directory, preserving original filename
            output_path = os.path.join(output_dir, image_name)
            result_pil = Image.fromarray(detected_image)
            result_pil.save(output_path)

            return len(valid_faces)

    def bulk_processing(self):
        # Let the user pick a folder of input images
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            return

        # Collect all supported image files in the selected folder (non-recursive)
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not image_files:
            self.message_1.configure(text="No image files found in the selected folder.")
            self.message_2.configure(text="")
            return

        # Create output subfolder inside the selected folder
        output_dir = os.path.join(folder_path, "processed")
        os.makedirs(output_dir, exist_ok=True)

        self.message_1.configure(text=f"Processing {len(image_files)} image(s)…")
        self.message_2.configure(text="Please wait.")
        self.master.update_idletasks()

        total_faces = 0
        errors = 0
        bulk_start = time.time()

        for idx, file_path in enumerate(image_files, start=1):
            # Update status for each image so the user can see progress
            self.message_1.configure(
                text=f"Processing image {idx} of {len(image_files)}: "
                        f"{os.path.basename(file_path)}"
            )
            self.master.update_idletasks()

            try:
                faces_found = self.process_single_image_to_file(file_path, output_dir)
                total_faces += faces_found
            except Exception as exc:
                errors += 1
                print(f"ERROR processing {file_path}: {exc}")

        bulk_end = time.time()
        elapsed = bulk_end - bulk_start

        # Show the last processed image in the GUI as a preview
        if image_files:
            last_file = image_files[-1]
            try:
                last_input = Image.open(last_file).convert("RGB")
                display_input = self.resize_image(last_input)
                input_photo = ImageTk.PhotoImage(display_input)
                self.image_label.configure(image=input_photo)
                self.image_label.image = input_photo

                last_output_path = os.path.join(output_dir, os.path.basename(last_file))
                if os.path.exists(last_output_path):
                    last_output = Image.open(last_output_path).convert("RGB")
                    display_output = self.resize_image(last_output)
                    output_photo = ImageTk.PhotoImage(display_output)
                    self.filtered_label.configure(image=output_photo)
                    self.filtered_label.image = output_photo
            except Exception:
                pass

        success_count = len(image_files) - errors
        self.message_1.configure(
            text=f"Done — {success_count}/{len(image_files)} image(s) processed, "
                    f"{total_faces} face(s) detected total."
                    + (f" ({errors} error(s))" if errors else "")
        )
        self.message_2.configure(
            text=f"Results saved to: …/{os.path.basename(folder_path)}/processed/   "
                    f"({elapsed:.1f}s)"
        )

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()