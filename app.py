# ==========================================
# app.py - Streamlit MAE Reconstruction App
# ==========================================
from huggingface_hub import hf_hub_download
import os
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io

# ---------------------------------------------------------
# COPY YOUR CLASSES HERE (PatchifyAndMask, TransformerBlock, 
# ViT_Encoder, ViT_Decoder, unpatchify) from the Kaggle notebook.
# For brevity, paste them exactly as they are in Kaggle here!
# ---------------------------------------------------------

# (PASTE CLASSES HERE)

class PatchifyAndMask(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        
        # Step 9: Define Patch Size and Image Size
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Step 10: Calculate Dimensions
        # Total patches = (224 / 16) * (224 / 16) = 14 * 14 = 196
        self.num_patches = (img_size // patch_size) ** 2 
        
        # Patch dimension = 3 channels * 16 height * 16 width = 768
        self.patch_dim = in_channels * patch_size * patch_size 

    # Step 11: Implement `patchify` Method
    def patchify(self, imgs):
        """
        Takes an image tensor of shape [Batch, Channels, Height, Width]
        and converts it to a sequence of patches [Batch, Num_Patches, Patch_Dim]
        """
        p = self.patch_size
        # h = w = 224 // 16 = 14
        h = w = self.img_size // p 
        
        # 1. Reshape the image to separate the patches
        # From [Batch, 3, 224, 224] -> [Batch, 3, 14, 16, 14, 16]
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        
        # 2. Swap axes to group spatial dimensions (h, w) and patch pixels (p, p, 3)
        # We use torch.einsum for clean swapping. 'nchpwq->nhwpqc'
        # N=Batch, C=Channels, H=h, P=patch_h, W=w, Q=patch_w
        x = torch.einsum('nchpwq->nhwpqc', x)
        
        # 3. Flatten the spatial dimensions (14*14=196) and patch pixels (16*16*3=768)
        # Final shape: [Batch, 196, 768]
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    # Step 12: Implement `random_masking` Method
    def random_masking(self, x, mask_ratio=0.75):
        """
        Takes the sequence of patches and hides 75% of them.
        x shape: [Batch, 196, 768]
        """
        N, L, D = x.shape  # N = Batch, L = 196 (Length), D = 768 (Dim)
        
        # Calculate exactly how many patches to keep (25%)
        # 196 * 0.25 = 49 patches
        len_keep = int(L * (1 - mask_ratio)) 
        
        # 1. Generate random noise for all 196 patches
        noise = torch.rand(N, L, device=x.device) 
        
        # 2. Sort the noise to get shuffled indices (smallest to largest)
        # ids_shuffle tells us the new random order of patches
        ids_shuffle = torch.argsort(noise, dim=1) 
        
        # 3. Create 'ids_restore' by sorting the shuffled indices again.
        # We MUST save this to put patches back in their correct grid position later!
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 4. Keep only the first 49 indices from our shuffled list
        ids_keep = ids_shuffle[:, :len_keep]
        
        # 5. Gather the actual patch pixel data for those 49 indices
        # We have to expand ids_keep to match the 768 dimensions to copy the data properly
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 6. Generate a binary mask tensor (1 for masked/hidden, 0 for visible)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 # Set first 49 to 0 (visible)
        
        # Unshuffle the mask so it matches the original image layout
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Return the 49 visible patches, the binary mask, and the restore indices
        return x_kept, mask, ids_restore
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Step 17: Multi-Head Self-Attention (MSA)
        # batch_first=True means our tensor is [Batch, Sequence, Features]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Step 18: MLP Block
        # In a standard ViT, the hidden layer in the MLP is 4x the embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), # GELU is the standard activation function for Transformers
            nn.Linear(embed_dim * 4, embed_dim)
        )

    # Step 19: Combine them into a Block
    def forward(self, x):
        # 1. Attention part with Residual Connection
        # self.attn returns a tuple (output, weights). We only need output [0]
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output # Residual connection
        
        # 2. MLP part with Residual Connection
        x = x + self.mlp(self.norm2(x)) # Residual connection
        return x

class ViT_Encoder(nn.Module):
    # Step 13: Define Config (ViT-Base: dim 768, 12 layers, 12 heads)
    def __init__(self, patch_dim=768, embed_dim=768, depth=12, num_heads=12, num_patches=196):
        super().__init__()
        
        # Step 14: Patch Embedding Layer
        # Projects raw pixel values (768) into the Transformer's hidden dimension (768)
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # Step 15: Positional Embeddings
        # A learnable tensor for 196 positions. Shape: [1, 196, 768]
        # We use nn.Parameter so the model updates these during training
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Step 20: Stack 12 Transformer Blocks
        # nn.ModuleList is like a Python list but registers the modules inside PyTorch
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    # Step 21: Encoder Forward Pass
    def forward(self, x, patcher_module):
        # x input shape: [Batch, 196, 768] (All raw patches from Step 11)
        
        # 1. Project patches to embedding dimension
        x = self.patch_embed(x)
        
        # 2. Add Positional Embeddings to ALL 196 patches
        x = x + self.pos_embed
        
        # 3. NOW we apply random masking!
        # We pass the embedded patches to Phase 2 module to hide 75%.
        # It returns exactly 49 visible embedded patches.
        x_visible, mask, ids_restore = patcher_module.random_masking(x, mask_ratio=0.75)
        
        # 4. Pass ONLY the 49 visible patches through the 12 Transformer blocks
        for block in self.blocks:
            x_visible = block(x_visible)
            
        x_visible = self.norm(x_visible)
        
        # Return encoded features + the metadata needed for the Decoder
        return x_visible, mask, ids_restore


class ViT_Decoder(nn.Module):
    # Step 22: Define Config (ViT-Small: dim 384, 12 layers, 6 heads)
    def __init__(self, encoder_dim=768, decoder_dim=384, depth=12, num_heads=6, num_patches=196, patch_dim=768):
        super().__init__()
        
        # Step 23: Project to Decoder Dimension
        # The encoder outputs size 768, but decoder only needs 384
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        
        # Step 24: Initialize Mask Tokens
        # This is a learnable "blank canvas" representing the missing patches.
        # Shape: [1 batch, 1 patch, 384 dimension]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Step 27: Decoder Positional Embeddings
        # The decoder needs to know where every patch belongs (all 196 of them)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        
        # Step 28: Stack 12 Smaller Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=decoder_dim, num_heads=num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(decoder_dim)
        
        # Step 29: Prediction Head
        # Finally, project the 384-dim hidden state back to 768 (which is 16x16x3 pixels)
        self.pred = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x, ids_restore):
        # 'x' is the visible patches from the encoder [Batch, 49, 768]
        # 'ids_restore' is the map to un-shuffle the patches [Batch, 196]
        
        # 1. Project from 768 to 384
        x = self.decoder_embed(x) # Shape becomes [Batch, 49, 384]
        
        # 2. Figure out how many mask tokens we need (Total 196 - Visible 49 = 147)
        B = x.shape[0] # Batch size
        num_visible = x.shape[1] # 49
        num_total = ids_restore.shape[1] # 196
        num_masks = num_total - num_visible # 147
        
        # 3. Create 147 mask tokens for every image in the batch
        # Repeat the single mask token to shape [Batch, 147, 384]
        mask_tokens = self.mask_token.repeat(B, num_masks, 1)
        
        # Step 25: Concatenate Tokens
        # Join the 49 visible tokens and 147 mask tokens -> Shape [Batch, 196, 384]
        # BUT they are not in the correct grid order yet!
        x_full = torch.cat([x, mask_tokens], dim=1) 
        
        # Step 26: Restore Spatial Order
        # We need to expand ids_restore to match the 384 dimension so we can gather
        ids_restore_expanded = ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        # Un-shuffle! Now the patches are back in their original 14x14 grid order
        x_unshuffled = torch.gather(x_full, dim=1, index=ids_restore_expanded)
        
        # Step 27: Add Positional Embeddings to all 196 unshuffled patches
        x_unshuffled = x_unshuffled + self.decoder_pos_embed
        
        # Step 28: Apply the 12 Transformer Blocks
        for block in self.blocks:
            x_unshuffled = block(x_unshuffled)
            
        x_unshuffled = self.norm(x_unshuffled)
        
        # Step 29: Predict the raw pixels!
        # Project from [Batch, 196, 384] to [Batch, 196, 768]
        predictions = self.pred(x_unshuffled)
        
        return predictions


def unpatchify(x, patch_size=16, img_size=224):
    """
    Turns [Batch, 196, 768] back into [Batch, 3, 224, 224]
    This is the exact mathematical reverse of our patchify function!
    """
    p = patch_size
    h = w = img_size // p # 14
    
    # 1. Reshape to separate patch dimensions
    # Shape becomes [Batch, 14, 14, 16, 16, 3]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    
    # 2. Swap axes to group channels and spatial dimensions correctly
    # 'nhwpqc->nchpwq'
    x = torch.einsum('nhwpqc->nchpwq', x)
    
    # 3. Flatten the spatial dimensions to get final image
    # Shape becomes [Batch, 3, 224, 224]
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs

# ---------------------------------------------


# We modify the Wrapper class slightly to accept the slider's mask_ratio
class MaskedAutoencoderApp(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher = PatchifyAndMask(img_size=224, patch_size=16, in_channels=3)
        self.encoder = ViT_Encoder(patch_dim=768, embed_dim=768, depth=12, num_heads=12)
        self.decoder = ViT_Decoder(encoder_dim=768, decoder_dim=384, depth=12, num_heads=6)

    def forward(self, imgs, mask_ratio):
        target_patches = self.patcher.patchify(imgs)
        
        # Manually handling the encoder steps to pass the custom mask_ratio
        x = self.encoder.patch_embed(target_patches)
        x = x + self.encoder.pos_embed
        x_visible, mask, ids_restore = self.patcher.random_masking(x, mask_ratio=mask_ratio)
        
        for block in self.encoder.blocks:
            x_visible = block(x_visible)
        x_visible = self.encoder.norm(x_visible)
        
        predictions = self.decoder(x_visible, ids_restore)
        return predictions, target_patches, mask

# ---------------------------------------------------------
# STREAMLIT UI CODE
# ---------------------------------------------------------

st.set_page_config(page_title="MAE Image Reconstruction", layout="wide")
st.title("Generative AI: Masked Autoencoder (MAE)")
st.write("Upload an image, select how much to hide, and watch the AI reconstruct it!")

# Load Model
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="abrar-hero/mae-image-reconstruction",
        filename="mae_model.pth",
        cache_dir="./models"
    )

    model = MaskedAutoencoderApp()
    # model.load_state_dict(torch.load("mae_model1.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure 'mae_model.pth' is in the same folder. Error: {e}")

# Sidebar UI
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
mask_ratio = st.sidebar.slider("Masking Ratio (%)", min_value=10, max_value=90, value=75, step=5) / 100.0

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension [1, 3, 224, 224]
    
    with torch.no_grad():
        preds, targets, mask = model(input_tensor, mask_ratio=mask_ratio)
        
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, targets.shape[2])
        
        # 1. Masked Input
        masked_input = targets * (1 - mask_expanded)
        masked_img_np = unpatchify(masked_input).squeeze().numpy()
        
        # 2. Reconstruction
        final_recon = preds * mask_expanded + targets * (1 - mask_expanded)
        recon_img_np = unpatchify(final_recon).squeeze().numpy()
        
        # 3. Original
        orig_img_np = unpatchify(targets).squeeze().numpy()

    # Display Images in Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"Masked ({int(mask_ratio*100)}% Hidden)")
        st.image(np.clip(np.transpose(masked_img_np, (1, 2, 0)), 0, 1), use_container_width=True)
        
    with col2:
        st.subheader("AI Reconstruction")
        st.image(np.clip(np.transpose(recon_img_np, (1, 2, 0)), 0, 1), use_container_width=True)
        
    with col3:
        st.subheader("Original")
        st.image(np.clip(np.transpose(orig_img_np, (1, 2, 0)), 0, 1), use_container_width=True)