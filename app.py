from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import torch
import numpy as np
from models import get_processed_inputs, inpaint, processor, model, device  # Import functions and models

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('image')
        if not file:
            return "No file uploaded", 400

        # Open and process the uploaded image
        raw_image = Image.open(file).convert("RGB").resize((512, 512))
        input_points = [[[150, 170], [300, 250]]]  # Example input points; you may update for user input

        # Generate mask using SAM
        mask = get_processed_inputs(raw_image, input_points)

        # Prepare prompt inputs
        prompt = request.form.get('prompt', 'a german shepherd dog')
        negative_prompt = request.form.get('negative_prompt', 'artifacts, low quality, distortion')

        # Inpainting with specified prompt
        inpainted_image = inpaint(raw_image, mask, prompt, negative_prompt)

        # Save the inpainted image temporarily to send to the user
        inpainted_image_path = 'static/inpainted_image.png'
        inpainted_image.save(inpainted_image_path)

        return send_file(inpainted_image_path, mimetype='image/png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
