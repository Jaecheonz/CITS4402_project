import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import time
import os
import dlib
from sklearn.cluster import DBSCAN

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
        self.landmark_model_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
        # Stop the program early if landmark model file is missing
        if not os.path.exists(self.landmark_model_path):
            raise FileNotFoundError(f"dlib landmark model not found: {self.landmark_model_path}")
        # Load pretrained dlib 5-point landmark predictor
        self.landmark_predictor = dlib.shape_predictor(self.landmark_model_path)

        # Path for dlib face recognition model
        self.recognition_model_path = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.exists(self.recognition_model_path):
            raise FileNotFoundError(f"dlib recognition model not found: {self.recognition_model_path}")
        # Load pretrained dlib face recognition ResNet model
        self.face_recognizer = dlib.face_recognition_model_v1(self.recognition_model_path)
        
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
        landmarks_per_face = []

        for (start_x, start_y, box_width, box_height, confidence) in valid_faces:
            end_x = start_x + box_width
            end_y = start_y + box_height

            face_rect = dlib.rectangle(start_x, start_y, end_x, end_y)
            shape = self.landmark_predictor(image_array, face_rect)

            # 68-point model indices:
            # 36-41 = right eye, 42-47 = left eye, 30 = nose tip
            right_eye_pts = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            left_eye_pts  = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            nose_tip      = (shape.part(30).x, shape.part(30).y)

            right_eye_centre = (
                int(np.mean([p[0] for p in right_eye_pts])),
                int(np.mean([p[1] for p in right_eye_pts]))
            )
            left_eye_centre = (
                int(np.mean([p[0] for p in left_eye_pts])),
                int(np.mean([p[1] for p in left_eye_pts]))
            )

            # All 68 points for piecewise warp
            all_points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            landmarks_per_face.append({
                "box":       (start_x, start_y, box_width, box_height, confidence),
                "right_eye": right_eye_centre,
                "left_eye":  left_eye_centre,
                "nose":      nose_tip,
                "all_points": all_points
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
        all_points = face_data["all_points"]

        canonical = [
            (32,116),(31,98),(31,80),(33,63),(37,47),(44,33),(53,21),(64,13),(76,13),
            (87,21),(94,33),(99,47),(101,63),(101,80),(100,98),(99,116),(62,36),
            (62,28),(62,20),(62,13),(62,6),(50,30),(54,24),(60,21),(66,21),(71,24),
            (73,30),(68,36),(62,38),(56,36),(45,30),(49,24),(55,21),(62,21),(68,21),
            (74,24),(78,30),(37,42),(42,38),(48,38),(53,42),(48,45),(42,45),
            (71,42),(76,38),(82,38),(87,42),(82,45),(76,45),(62,55),(57,51),
            (52,49),(47,52),(52,55),(57,57),(62,57),(67,57),(72,55),(77,52),
            (72,49),(67,51),(52,52),(57,49),(62,49),(67,49),(72,52),(67,55),(62,55),(57,55)
        ]

        src_pts = np.array(all_points, dtype=np.float32)
        dst_pts = np.array(canonical,  dtype=np.float32)

        src_eye_distance = np.linalg.norm(src_pts[45] - src_pts[36])
        if src_eye_distance < 1:
            return np.zeros((125, 125, 3), dtype=np.uint8), None

        output_image = np.zeros((125, 125, 3), dtype=np.uint8)

        # Delaunay triangulation on destination points only
        rect = (0, 0, 125, 125)
        subdiv = cv2.Subdiv2D(rect)
        # Insert points and track insertion order
        for p in dst_pts:
            subdiv.insert((float(p[0]), float(p[1])))

        triangles = subdiv.getTriangleList().astype(np.float32)

        # Round dst_pts for fast lookup: coordinate -> index
        dst_lookup = {}
        for i, p in enumerate(dst_pts):
            key = (round(float(p[0]), 1), round(float(p[1]), 1))
            dst_lookup[key] = i

        for tri in triangles:
            pts = [
                (round(float(tri[0]), 1), round(float(tri[1]), 1)),
                (round(float(tri[2]), 1), round(float(tri[3]), 1)),
                (round(float(tri[4]), 1), round(float(tri[5]), 1)),
            ]

            # Skip triangles whose vertices aren't in our 68 points (boundary triangles)
            indices = []
            valid = True
            for pt in pts:
                if pt not in dst_lookup:
                    valid = False
                    break
                indices.append(dst_lookup[pt])
            if not valid:
                continue

            tri_src = np.array([all_points[i] for i in indices], dtype=np.float32)
            tri_dst = np.array([canonical[i]   for i in indices], dtype=np.float32)

            r = cv2.boundingRect(tri_dst)
            x, y, w, h = r
            if w <= 0 or h <= 0:
                continue

            # Offset to bounding box
            tri_src_offset = tri_src
            tri_dst_offset = tri_dst - np.array([x, y], dtype=np.float32)

            M = cv2.getAffineTransform(tri_dst_offset,
                                    tri_src_offset - np.array([x, y], dtype=np.float32) + np.array([x, y], dtype=np.float32))

            # Warp the source patch
            warped = cv2.warpAffine(image_array,
                                    cv2.getAffineTransform(tri_src, tri_dst),
                                    (125, 125))

            # Triangle mask in destination bounding box
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, tri_dst_offset.astype(np.int32), 255)

            roi     = output_image[y:y+h, x:x+w]
            src_roi = warped[y:y+h, x:x+w]
            roi[mask == 255] = src_roi[mask == 255]
            output_image[y:y+h, x:x+w] = roi

        return output_image, None
    
    def draw_landmarks_on_aligned_face(self, aligned_face, face_data, transform_matrix):
        output_face = aligned_face.copy()

        left_eye  = (37, 42)
        right_eye = (88, 42)
        nose      = (62, 75)

        cv2.circle(output_face, right_eye, 4, (255, 0, 0), -1)   # blue
        cv2.circle(output_face, left_eye,  4, (0, 255, 0), -1)   # green
        cv2.circle(output_face, nose,      4, (0, 0, 255), -1)   # red

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
            if transform_matrix is not None:
                aligned_face = self.draw_landmarks_on_aligned_face(aligned_face, face_data, transform_matrix)
                aligned_faces.append(aligned_face)

        # Paste aligned faces into the four corners of the output image
        detected_image = self.place_faces_in_corners(detected_image, aligned_faces)

        end_time = time.time()

        # Convert detected back to PIL for display
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

    #Runs the full detection process on an array instead of a singular image, and returns an array of images instead of a single one(for bulk process use only)
    def process_bulk_to_array(self, file_path):
            image = Image.open(file_path).convert("RGB")
            image_array = np.array(image)

            detected_image, _, valid_faces, debug_info = self.detect_faces(image_array)
            landmarks_per_face = self.detect_landmarks(image_array, valid_faces)

            landmarks_per_face.sort(key=lambda face_data: face_data["box"][0])

            detected_image = self.draw_landmarks(detected_image, landmarks_per_face)

            aligned_faces_clean = []   # for clustering and grid display
            aligned_faces_marked = []  # for saved result images
            for face_data in landmarks_per_face:
                aligned_face, transform_matrix = self.align_and_crop_face(image_array, face_data)
                if transform_matrix is not None:
                    aligned_faces_clean.append(aligned_face.copy())
                    marked = self.draw_landmarks_on_aligned_face(aligned_face, face_data, transform_matrix)
                    aligned_faces_marked.append(marked)

            detected_image = self.place_faces_in_corners(detected_image, aligned_faces_marked)

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

            return len(valid_faces), aligned_faces_clean
    


    def cluster_faces(self, face_crops):
        if not face_crops:
            return [], 0

        # Compute 128-dimensional dlib face embeddings for each aligned face crop.
        vectors = []
        for crop in face_crops:
            dlib_img = np.array(crop)
            h, w = dlib_img.shape[:2]

            # The crop is already an aligned 125 x 125 face, so use the full crop as the face region.
            rect = dlib.rectangle(0, 0, w - 1, h - 1)
            shape = self.landmark_predictor(dlib_img, rect)
            descriptor = self.face_recognizer.compute_face_descriptor(dlib_img, shape)
            vec = np.array(descriptor, dtype=np.float32)
            vectors.append(vec)
 
        X = np.array(vectors)
 
        # DBSCAN clustering

        # dlib embeddings work well in range 0.4–0.6; lower = stricter (more clusters)
        EPS = 0.55         # tune this to adjust cluster sensitivity
        MIN_SAMPLES = 1    #keep this value 1 so unique, singular faces only found once still get an identity/cluster of its own
        db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean").fit(X)
        raw_labels = db.labels_
 
        # Remap labels to 0-based consecutive integers (DBSCAN uses -1 for noise, with min_samples=1 there should be no noise, but handle it just in case by assigning each noise point its own unique cluster)
        label_map = {}
        labels = []
        next_noise_label = (max(raw_labels) + 1) if len(raw_labels) > 0 and max(raw_labels) >= 0 else 0

        for raw_label in raw_labels:
            if raw_label == -1:
                labels.append(next_noise_label)
                next_noise_label += 1
            else:
                labels.append(int(raw_label))

        # Remap labels to consecutive 0-based integers so filenames and grid rows are clean.
        unique_labels = sorted(set(labels))
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels]
        num_clusters = len(unique_labels)

        print(
            f"\nIdentity clustering using DBSCAN: {len(face_crops)} face(s) "
            f"→ {num_clusters} unique identity/identities "
            f"(eps={EPS}, min_samples={MIN_SAMPLES})"
        )
        for cid in range(num_clusters):
            members = [i for i, label in enumerate(labels) if label == cid]
            print(f"  Cluster {cid}: face indices {members}")

        return labels, num_clusters
    
    def build_identity_grid(self, face_crops, labels, num_clusters):
        FACE_SIZE  = 125   # edit to increase thumbnail size
        PADDING    = 6
        LABEL_H    = 20
        BG_COLOUR  = (40, 40, 40)
 
        # Group faces by cluster, preserving detection order within each cluster
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(face_crops[i])
 
        # Sort clusters so the cluster group with the highest count appears first
        sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
 
        max_cols = max(len(v) for v in clusters.values()) if clusters else 1
        num_rows = num_clusters
 
        grid_w = max_cols * (FACE_SIZE + PADDING) + PADDING
        grid_h = num_rows * (FACE_SIZE + PADDING + LABEL_H) + PADDING
 
        grid = np.full((grid_h, grid_w, 3), BG_COLOUR, dtype=np.uint8)
 
        # Distinct hues for cluster border colours (cycling if more than 12 clusters)
        distinct_colours = [
            (220, 50,  50),   # red
            (50,  120, 220),  # blue
            (50,  200, 50),   # green
            (220, 180, 50),   # yellow
            (180, 50,  220),  # purple
            (50,  210, 210),  # cyan
            (220, 120, 50),   # orange
            (220, 50,  150),  # pink
            (100, 220, 100),  # light green
            (50,  50,  200),  # dark blue
            (200, 100, 50),   # brown
            (150, 220, 50),   # lime
        ]
 
 
        for row_idx, (cid, faces) in enumerate(sorted_clusters):
            y_top = PADDING + row_idx * (FACE_SIZE + PADDING + LABEL_H)
 
            # Choose a border colour for this cluster
            border_colour = distinct_colours[row_idx % len(distinct_colours)]
 
            # Draw a thin coloured bar to separate rows (no text)
            label_bar = np.full((LABEL_H, grid_w - PADDING * 2, 3), border_colour, dtype=np.uint8)
            grid[y_top:y_top + LABEL_H, PADDING:grid_w - PADDING] = label_bar
 
            y_face = y_top + LABEL_H
            for col_idx, face in enumerate(faces):
                x_face = PADDING + col_idx * (FACE_SIZE + PADDING)
                # Place face thumbnail
                grid[y_face:y_face + FACE_SIZE, x_face:x_face + FACE_SIZE] = face
                
                # Draw coloured border around thumbnail
                cv2.rectangle(grid,
                              (x_face, y_face),
                              (x_face + FACE_SIZE - 1, y_face + FACE_SIZE - 1),
                              border_colour, 2)
 
        return grid


 
    def bulk_processing(self):
        # Lets the user pick a folder of input images
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            return

        # Collect all supported image files in the selected folder
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

        # Create output subfolder next to selected folder
        output_dir = os.path.join(os.path.dirname(folder_path), "Processed_Images")

        # If folder exists, clear all files inside it; otherwise create it
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                file_to_delete = os.path.join(output_dir, f)
                if os.path.isfile(file_to_delete):
                    os.remove(file_to_delete)
        else:
            os.makedirs(output_dir)

        self.message_1.configure(text=f"Processing {len(image_files)} image(s)…")
        self.message_2.configure(text="Please wait.")
        self.master.update_idletasks()

        total_faces = 0
        errors = 0
        all_face_crops = []
        bulk_start = time.time()

        # Update status for each image so the user can see progress (also for testing purposes)
        for idx, file_path in enumerate(image_files, start=1):
            self.message_1.configure(
                text=f"Processing image {idx} of {len(image_files)}: "
                        f"{os.path.basename(file_path)}"
            )
            self.master.update_idletasks()

            try:
                faces_found, crops = self.process_bulk_to_array(file_path)
                total_faces += faces_found
                all_face_crops.extend(crops)
            except Exception as exc:
                errors += 1
                print(f"ERROR processing {file_path}: {exc}")

        bulk_end = time.time()
        elapsed = bulk_end - bulk_start
        success_count = len(image_files) - errors

        self.message_1.configure(text="Clustering identities…")
        self.master.update_idletasks()

        labels, num_clusters = self.cluster_faces(all_face_crops)

        # Group face crops by cluster label
        cluster_face_counts = {}
        for face_idx, label in enumerate(labels):
            cluster_face_counts.setdefault(label, []).append(face_idx)

        # Map cluster label = identity number
        #This is the part where face crops from the array is assigned the correct label and saved individually.
        sorted_labels = sorted(cluster_face_counts.keys())
        for identity_num, label in enumerate(sorted_labels, start=1):
            for face_num, face_idx in enumerate(cluster_face_counts[label], start=1):
                crop = all_face_crops[face_idx]
                filename = f"Identity_{identity_num}_face_{face_num}.jpg"
                save_path = os.path.join(output_dir, filename)
                Image.fromarray(crop).save(save_path)

        #Building the identity grid image to be displayed in right panel for bulk process results
        if all_face_crops:
            grid_array = self.build_identity_grid(all_face_crops, labels, num_clusters)

            combined = grid_array

            # Display the first input image in the left panel
            try:
                first_input = Image.open(image_files[0]).convert("RGB")
                display_input = self.resize_image(first_input)
                input_photo = ImageTk.PhotoImage(display_input)
                self.image_label.configure(image=input_photo)
                self.image_label.image = input_photo
            except Exception:
                pass

            # Display the identity grid in the right panel
            grid_pil = Image.fromarray(combined)
            display_grid = self.resize_image(grid_pil)
            grid_photo = ImageTk.PhotoImage(display_grid)
            self.filtered_label.configure(image=grid_photo)
            self.filtered_label.image = grid_photo

        #Result messages
        summary_text = (
            f"Total {success_count} image(s) processed in {elapsed:.1f} seconds. "
            f"{total_faces} face(s) detected corresponding to "
            f"{num_clusters} unique identit{'y' if num_clusters == 1 else 'ies'}."
            + (f" ({errors} error(s))" if errors else "")
        )
        self.message_1.configure(text=summary_text)
        self.message_2.configure(
            text=f"Saved to: …/Processed_Images/   ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()