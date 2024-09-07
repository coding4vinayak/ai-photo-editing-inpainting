from transformers import SamModel, SamProcessor
from diffusers import AutoPipelineForInpainting
import torch

# Set the device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model and processor
checkpoint = "facebook/sam-vit-base"
model = SamModel.from_pretrained(checkpoint).to(device)
processor = SamProcessor.from_pretrained(checkpoint)

def get_processed_inputs(image, input_points):
    inputs = processor(images=image, input_points=input_points, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    best_mask = masks[0][0][outputs.iou_scores.argmax()]
    return ~best_mask.cpu().numpy()

# Load the inpainting pipeline
pipeline_checkpoint = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
pipeline = AutoPipelineForInpainting.from_pretrained(
    pipeline_checkpoint,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

def inpaint(raw_image, input_mask, prompt, negative_prompt=None, seed=74294536, cfgs=7):
    mask_image = Image.fromarray(input_mask)
    rand_gen = torch.manual_seed(seed)
    image = pipeline(
        image=raw_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        mask_image=mask_image,
        guidance_scale=cfgs,
        generator=rand_gen
    ).images[0]
    return image
