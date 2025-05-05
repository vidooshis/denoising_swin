
# # At the top of optimized.py
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# from Swin_Transformer_TF.swintransformer.model import SwinTransformer
# # Test model
# model = SwinTransformer(model_name='swin_base_224', include_top=False, pretrained='/Users/vidooshisharma/Downloads/swin_base_224/swin_base_224.ckpt')
# test_input = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
# output = model(test_input)
# print("Output shape:", output.shape)  # Should print a concrete tensor

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.backend as K
import glob, os
import numpy as np
from PIL import Image
import optuna
from tqdm import tqdm

import sys
#sys.path.append("/Users/vidooshisharma/Desktop/Neighbor2Neighbor/Swin-Transformer-TF")

# Import the Swin Transformer model from the repository.
# Corrected import path and added the missing in_chans parameter.
from Swin_Transformer_TF.swintransformer.model import SwinTransformer


# -------------------------------
# 1. Dataset: Create a tf.data pipeline for denoising images.
# -------------------------------
def load_image(path, patch_size=224):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [patch_size, patch_size])
    return img

def load_dataset(data_dir, patch_size=224, batch_size=64):
    image_paths = glob.glob(os.path.join(data_dir, '**/*.JPEG'), recursive=True)
    image_paths += glob.glob(os.path.join(data_dir, '**/*.png'), recursive=True)
   
    # Explicitly set dtype to tf.string to avoid issues with an empty list.
    image_paths = tf.constant(image_paths, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda path: load_image(path, patch_size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# -------------------------------
# 2. Model: Build a denoising network using Swin Transformer as backbone.
# -------------------------------
def create_swin_denoiser(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initialize Swin components
    swin_backbone = SwinTransformer(
        model_name='swin_base_224',
        include_top=False,
        pretrained=False,
        num_classes=0
    )
    
    # Manually build features without final pooling
    x = swin_backbone.patch_embed(inputs)
    if swin_backbone.ape:
        x = x + swin_backbone.absolute_pos_embed
    x = swin_backbone.pos_drop(x)
    x = swin_backbone.basic_layers(x)  # Shape: (batch, 49, 1024)
    
    # Reshape to spatial dimensions (7x7 grid)
    x = layers.Reshape((7, 7, 1024))(x)
    
    # Decoder with proper upsampling
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)  # 14x14
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)  # 28x28
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # 56x56
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)   # 112x112
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', 
                                   activation='sigmoid')(x)  # 224x224
    
    return tf.keras.Model(inputs, outputs, name='swin_denoiser')
# -------------------------------
# 3. Perceptual Loss: Define a loss combining MSE and VGG-based perceptual term.
# -------------------------------
class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        # Use MSE without batch reduction
        self.mse = tf.keras.losses.MeanSquaredError(reduction="none")  
        
        # Load VGG19
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights=None  # Disable auto-download
        )
        vgg.load_weights(os.path.expanduser("~/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"))
        
        # Feature extractor
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=vgg.get_layer("block3_conv3").output
        )
        self.feature_extractor.trainable = False

    def call(self, y_true, y_pred):
        # MSE loss per sample [batch_size, 224, 224, 3] → [batch_size]
        mse_loss = tf.reduce_mean(self.mse(y_true, y_pred), axis=[1,2])

        # Preprocess images for VGG
        y_true_vgg = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
        y_pred_vgg = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

        # Extract features [batch_size, h, w, c]
        features_true = self.feature_extractor(y_true_vgg)
        features_pred = self.feature_extractor(y_pred_vgg)

        # Perceptual loss per sample [batch_size]
        perceptual_loss = tf.reduce_mean(
            tf.square(features_true - features_pred), 
            axis=[1,2,3]
        )

        # Combine losses
        return mse_loss + 0.1 * perceptual_loss
# -------------------------------
# 4. Noise Estimator: A small conv net that predicts a noise level.
# -------------------------------
def create_noise_estimator(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(1, 3, padding='same', activation='relu')(x)
    return tf.keras.Model(inputs, x, name='noise_estimator')
# -------------------------------
# 5. Training Step using tf.GradientTape
# -------------------------------
@tf.function
def train_step(model, noise_estimator, loss_fn, optimizer, images):
    sigma_pred = tf.stop_gradient(noise_estimator(images))
    noise = tf.random.normal(tf.shape(images)) * sigma_pred
    noisy_images = images + noise

    with tf.GradientTape() as tape:
        denoised = model(noisy_images, training=True)
        loss = loss_fn(images, denoised)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(model, noise_estimator, train_dataset, val_dataset, optimizer, loss_fn, epochs=50, patience=5):
    avg_loss = tf.keras.metrics.Mean()
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        for batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            loss = train_step(model, noise_estimator, loss_fn, optimizer, batch)
            avg_loss.update_state(loss)
        train_loss = avg_loss.result().numpy()
        avg_loss.reset_states()

        # Validate after each epoch
        val_loss = validate_model(model, noise_estimator, val_dataset, loss_fn)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.get_weights()  # Save best weights
            patience_counter = 0
            # Save checkpoint
            model.save_weights("best_model_weights.h5")
            print("Validation loss improved. Saving model checkpoint.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

        # Early stopping if no improvement for 'patience' epochs
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Restore best model weights
    if best_model_weights is not None:
        model.set_weights(best_model_weights)
    return model



def validate_model(model, noise_estimator, dataset, loss_fn):
    """Calculate average loss on validation dataset"""
    avg_loss = tf.keras.metrics.Mean()
    
    for batch in dataset:
        sigma_pred = tf.stop_gradient(noise_estimator(batch))
        noise = tf.random.normal(tf.shape(batch)) * sigma_pred
        noisy_batch = batch + noise
        
        denoised = model(noisy_batch, training=False)
        loss_value = loss_fn(batch, denoised)
        avg_loss.update_state(loss_value)
    
    return avg_loss.result().numpy()
# -------------------------------
# 6. Optuna Objective Function
# -------------------------------
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32)
    
    # Models
    model = create_swin_denoiser()
    noise_estimator = create_noise_estimator()
    
    # Datasets
    train_data = load_dataset(
        '/Users/vidooshisharma/Desktop/Neighbor2Neighbor/dataset/new_dataset/',
        patch_size=224,
        batch_size=batch_size
    )
    val_data = load_dataset(
        '/Users/vidooshisharma/Desktop/Neighbor2Neighbor/validation/', 
        patch_size=224,
        batch_size=batch_size
    )
    
    # Training with early stopping (here, patience is set to 3 epochs)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    trained_model = train_model(
        model,
        noise_estimator,
        train_data,
        val_data,
        optimizer,
        PerceptualLoss(),
        epochs=20,       # You can set a maximum number of epochs
        patience=3       # Early stopping patience
    )
    
    # Validation
    return validate_model(trained_model, noise_estimator, val_data, PerceptualLoss())


# -------------------------------
# 7. Run Optuna Study
# -------------------------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print("Best trial:")
trial = study.best_trial
print(f"  Loss: {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# -------------------------------
# 8. Retrain Using Best Hyperparameters and Save the Model
# -------------------------------
# Retrieve best trial hyperparameters
# Retrieve best trial hyperparameters
best_params = study.best_trial.params

# Rebuild models with best parameters and train longer using early stopping
model = create_swin_denoiser()
noise_estimator = create_noise_estimator()

train_data = load_dataset('/Users/vidooshisharma/Desktop/Neighbor2Neighbor/dataset/new_dataset', patch_size=224, batch_size=best_params['batch_size'])
val_data = load_dataset('/Users/vidooshisharma/Desktop/Neighbor2Neighbor/validation/BSD300/test', patch_size=224, batch_size=best_params['batch_size'])
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=best_params['lr'])

# For full training, use more epochs with early stopping (e.g., max 50 epochs with patience=5)
trained_model = train_model(model, noise_estimator, train_data, val_data, optimizer, PerceptualLoss(), epochs=50, patience=5)

# Save the best model (it is already saved during training as "best_model.h5")
trained_model.save_weights('swin_denoiser_best_weights.h5')
