import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# -------------------------------------------
# 1. Define the model architecture
# -------------------------------------------
# Import SwinTransformer from your repository (adjust path if needed)
from Swin_Transformer_TF.swintransformer.model import SwinTransformer
from tensorflow.keras import layers


def load_and_crop_image(path, crop_size=256):
    """
    Loads an image from disk, decodes it, converts to float [0,1],
    and then crops a central patch of size (crop_size, crop_size).
    """
    img = tf.io.read_file(path)
    
    # Decode based on file extension
    if path.lower().endswith('.png'):
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Get the image dimensions
    shape = tf.shape(img)
    height, width = shape[0], shape[1]
    
    # Compute offsets for central crop
    offset_height = (height - crop_size) // 2
    offset_width = (width - crop_size) // 2
    
    # Crop the image
    cropped_img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, crop_size, crop_size)
    
    return cropped_img

def create_swin_denoiser(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initialize Swin components
    swin_backbone = SwinTransformer(
        model_name='swin_base_224',
        include_top=False,
        pretrained=False,
        num_classes=0
    )
    
    # Build features
    x = swin_backbone.patch_embed(inputs)
    if swin_backbone.ape:
        x = x + swin_backbone.absolute_pos_embed
    x = swin_backbone.pos_drop(x)
    x = swin_backbone.basic_layers(x)  # Expected shape: (batch, 49, 1024)
    
    # Reshape to spatial dimensions (7x7 grid)
    x = layers.Reshape((7, 7, 1024))(x)
    
    # Decoder with transpose convolutions to upscale back to 224x224
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)  # 14x14
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)  # 28x28
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # 56x56
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)   # 112x112
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)  # 224x224
    
    return tf.keras.Model(inputs, outputs, name='swin_denoiser')

# -------------------------------------------
# 2. Helper functions to load images & SIDD pairs
# -------------------------------------------
def load_image(path, patch_size=224):
    """Load an image, decode, normalize to [0,1] and resize."""
    img = tf.io.read_file(path)
    if path.lower().endswith('.png'):
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [patch_size, patch_size])
    return img

def load_sidd_pairs(sidd_data_dir, patch_size=224):
    """
    Scan each subfolder of the SIDD dataset directory (e.g., SIDD_Medium_Raw/Data)
    and collect paths to the ground-truth (GT) and noisy images.
    """
    noisy_paths = []
    clean_paths = []
    
    # Each subfolder should contain both a "GT_SRGB" and a "NOISY_SRGB" image.
    for subfolder in os.listdir(sidd_data_dir):
        subfolder_path = os.path.join(sidd_data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            gt_file = None
            noisy_file = None
            for f in files:
                if 'GT_SRGB' in f.upper():
                    gt_file = os.path.join(subfolder_path, f)
                elif 'NOISY_SRGB' in f.upper():
                    noisy_file = os.path.join(subfolder_path, f)
            if gt_file and noisy_file:
                clean_paths.append(gt_file)
                noisy_paths.append(noisy_file)
    
    return noisy_paths, clean_paths

# -------------------------------------------
# 3. Evaluation: Compute PSNR and SSIM on SIDD
# -------------------------------------------
def evaluate_model_on_sidd(model, sidd_data_dir, patch_size=224):
    """
    1. Loads (noisy, clean) image pairs from SIDD.
    2. Denoises the noisy images using the loaded model.
    3. Computes and prints average PSNR and SSIM.
    """
    noisy_paths, clean_paths = load_sidd_pairs(sidd_data_dir, patch_size)
    
    psnr_values = []
    ssim_values = []
    
    for noisy_path, clean_path in tqdm(zip(noisy_paths, clean_paths), total=len(noisy_paths), desc="Evaluating"):
        # Load images
        noisy_image = load_and_crop_image(noisy_path, patch_size)
        clean_image = load_and_crop_image(clean_path, patch_size)
        
        # Add batch dimension for inference
        noisy_batch = tf.expand_dims(noisy_image, axis=0)
        
        # Run model inference
        denoised_batch = model(noisy_batch, training=False)
        denoised_image = tf.squeeze(denoised_batch, axis=0)
        
        # Compute PSNR and SSIM (images expected in [0,1])
        psnr = tf.image.psnr(denoised_image, clean_image, max_val=1.0)
        print(f"psnr:{psnr:.2f} dB")
        ssim = tf.image.ssim(denoised_image, clean_image, max_val=1.0)
        print(f"ssim:{psnr:.4f} ")
        psnr_values.append(psnr.numpy())
        ssim_values.append(ssim.numpy())
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print("\n--- Evaluation Results on SIDD ---")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

# -------------------------------------------
# 4. Main script: Load trained model and evaluate on SIDD
# -------------------------------------------
if __name__ == "__main__":
    # Path to your trained model weights file
    weights_path = "/Users/vidooshisharma/Downloads/best_model_weights.h5"
    
    # Path to the SIDD dataset directory (adjust this to your SIDD_Medium_Raw/Data folder)
    sidd_data_dir = "/Users/vidooshisharma/Downloads/SIDD_Medium_Raw/Data"
    
    # Rebuild the model architecture and load weights
    model = create_swin_denoiser(input_shape=(224, 224, 3))
    model.load_weights(weights_path)
    print("Trained model loaded successfully.")
    
    # Evaluate the model on the SIDD dataset
    evaluate_model_on_sidd(model, sidd_data_dir, patch_size=224)


