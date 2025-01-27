from flask import Flask, render_template, request, send_file, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import os
import zipfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Mediapipe setup
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Background removal function
def remove_background(image_path, output_path, bg_color=(255, 255, 255)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = segmentation.process(image_rgb)
    mask = results.segmentation_mask
    mask = (mask * 255).astype(np.uint8)
    refined_mask = cv2.GaussianBlur(mask, (13, 13), 0)
    condition = refined_mask > 128

    bg_image = np.zeros_like(image, dtype=np.uint8)
    bg_image[:] = bg_color
    alpha = refined_mask / 255.0
    foreground = image * alpha[..., None]
    background = bg_image * (1 - alpha[..., None])
    output_image = cv2.add(foreground.astype(np.uint8), background.astype(np.uint8))
    cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return "No files uploaded", 400

    files = request.files.getlist('files[]')
    if len(files) > 10:
        return "You can upload a maximum of 10 images", 400

    processed_images = []

    for file in files:
        if file.filename == '':
            continue

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(PROCESSED_FOLDER, 'processed_' + filename)

        file.save(input_path)
        remove_background(input_path, output_path)
        processed_images.append(output_path)

    # Create ZIP file of processed images
    zip_path = os.path.join(PROCESSED_FOLDER, "processed_images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for image in processed_images:
            zipf.write(image, os.path.basename(image))

    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
