import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os

# Custom Dataset for loading images from .npz files
class CustomDataset(Dataset):
    def __init__(self, npz_files, preprocess):
        self.npz_files = npz_files
        self.preprocess = preprocess

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        
        # Extract images
        rpn_img = data['rpn_box']
        projected_img = data['projected_box']
        
        # Convert images to PIL format
        rpn_img_pil = Image.fromarray((rpn_img * 255).astype(np.uint8))
        projected_img_pil = Image.fromarray((projected_img * 255).astype(np.uint8))
        
        # Preprocess images
        rpn_img_pil = self.preprocess(rpn_img_pil)
        projected_img_pil = self.preprocess(projected_img_pil)
        
        # Extract text label from file name
        yolo_label = os.path.basename(npz_file).split('_')[2]  # Extract the label from the file name
        text = clip.tokenize(f"a photo of a {yolo_label}")
        
        return rpn_img_pil, projected_img_pil, text

# Fine-tune CLIP
def fine_tune_clip(npz_files, device="cuda", num_epochs=10, batch_size=32, lr=1e-5):
    # Load CLIP Model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare Dataset and DataLoader
    dataset = CustomDataset(npz_files, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            images_rpn, images_projected, texts = batch
            
            # Remove the extra dimension from texts
            texts = texts.squeeze(1)  # Shape should now be (batch_size, sequence_length)

            # Ensure texts tensor is correctly shaped
            print(f"Shape of texts: {texts.shape}")

            images_rpn = images_rpn.to(device)
            images_projected = images_projected.to(device)
            texts = texts.to(device)

            # Forward pass for RPN images
            image_features_rpn = model.encode_image(images_rpn)
            text_features = model.encode_text(texts)
            image_features_rpn = image_features_rpn / image_features_rpn.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logits_per_image_rpn = image_features_rpn @ text_features.T
            logits_per_text_rpn = text_features @ image_features_rpn.T
            ground_truth = torch.arange(len(images_rpn), device=device)
            loss_rpn = (torch.nn.functional.cross_entropy(logits_per_image_rpn, ground_truth) +
                        torch.nn.functional.cross_entropy(logits_per_text_rpn, ground_truth)) / 2

            # Forward pass for Projected images
            image_features_projected = model.encode_image(images_projected)
            image_features_projected = image_features_projected / image_features_projected.norm(dim=1, keepdim=True)
            logits_per_image_projected = image_features_projected @ text_features.T
            logits_per_text_projected = text_features @ image_features_projected.T
            loss_projected = (torch.nn.functional.cross_entropy(logits_per_image_projected, ground_truth) +
                              torch.nn.functional.cross_entropy(logits_per_text_projected, ground_truth)) / 2

            # Total loss
            total_loss += (loss_rpn + loss_projected).item()

            # Backward pass
            optimizer.zero_grad()
            (loss_rpn + loss_projected).backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_clip.pt")

# Example usage
if __name__ == "__main__":
    # Path to Cropped_Images directory
    npz_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Cropped_Images/"
    
    # Get all .npz file paths from the directory
    npz_files = [os.path.join(npz_dir, fname) for fname in os.listdir(npz_dir) if fname.endswith('.npz')]
    
    # Check if any .npz files are found
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in directory {npz_dir}.")
    
    # Fine-tune CLIP with the images and generated text prompts
    fine_tune_clip(npz_files)

