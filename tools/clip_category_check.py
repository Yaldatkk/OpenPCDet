import torch
import clip
from PIL import Image

# Load the model and preprocess the image
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocess function to the correct device
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess the image and move it to the appropriate device
image = preprocess(Image.open("/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/training/image_2/000105.png")).unsqueeze(0).to(device)

# Define the category to check
categories = ["pan", "cyclist", "cyclist", "car"]
text_inputs = torch.cat([clip.tokenize(c).to(device) for c in categories])  # Move the text inputs to the same device

# Forward pass through the model
with torch.no_grad():
    image_features = model.encode_image(image)  # Encode the image
    text_features = model.encode_text(text_inputs)  # Encode the text prompts

# Compute similarities between image and text prompts
similarities = (image_features @ text_features.T).squeeze(0)

# Print the category with the highest score
best_category = categories[similarities.argmax().item()]
print(f"The image is most likely: {best_category}")

