
# AI Photo Editing with Inpainting

This project is a web application that allows users to swap out the background of a subject in an image using SAM (Segment Anything Model) for segmentation and Stable Diffusion for inpainting. Users can upload an image, specify a text prompt, and generate new backgrounds for their photos.

## Features

- **Background Removal**: Automatically detect and remove the background of a subject using the Segment Anything Model (SAM).
- **AI-Powered Inpainting**: Replace the background with an AI-generated image using Stable Diffusion XL.
- **Interactive Web Interface**: Easily upload images and generate new backgrounds through an intuitive web interface.

## Setup

To get started with this project, follow these steps:

### Prerequisites

Ensure you have the following installed on your machine:

- Python 3.8+
- CUDA-enabled GPU (if available) for faster processing
- Git
- pip

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/coding4vinayak/ai-photo-editing-inpainting.git
   cd ai-photo-editing-inpainting
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install the Dependencies**

   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**

   Ensure you have the required models downloaded:

   ```python
   from transformers import SamModel, SamProcessor
   from diffusers import AutoPipelineForInpainting

   # Load models
   model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
   processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
   pipeline = AutoPipelineForInpainting.from_pretrained(
       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
       torch_dtype=torch.float16
   ).to("cuda")
   ```

## Usage

To run the application, execute the following command:

```bash
python app.py
```

Once the server is running, you can access the web app through the provided link, typically `http://127.0.0.1:7860` or a public URL for remote access.

### How to Use the Web App

1. **Upload an Image**: Click on the upload button and select an image to edit.
2. **Define Points**: Manually select points to indicate the subject you want to keep.
3. **Enter a Text Prompt**: Describe the new background you want to generate.
4. **Generate**: Click on the generate button to see the AI-powered background replacement.

## Project Structure

- `app.py`: The main Flask application script that sets up the web server.
- `model.py`: Contains the SAM and Stable Diffusion model loading and processing functions.
- `static/`: Contains static assets like CSS and JavaScript files.
- `templates/`: HTML templates for the Flask application.
- `requirements.txt`: List of Python dependencies.

## Key Components

### Segment Anything Model (SAM)

SAM is used to generate segmentation masks of the subject, allowing precise separation of the foreground from the background.

### Stable Diffusion XL

Stable Diffusion is employed to generate realistic and contextually accurate backgrounds based on the provided text prompt.

## Troubleshooting

- **Memory Issues**: Ensure that your GPU has enough memory; otherwise, consider reducing the image size or using CPU fallback.
- **Model Loading Errors**: Make sure you have correctly installed the models and dependencies as listed.

## Future Improvements

- Adding more advanced controls for mask selection.
- Enhancing the UI for a better user experience.
- Introducing more inpainting models for different art styles.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Facebook SAM Model](https://github.com/facebookresearch/segment-anything)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

---

Happy editing!
```
