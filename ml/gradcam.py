import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def find_last_conv_layer(model):
    """
    Dynamically find the last convolutional layer in the model by iterating
    in reverse order through the layers.
    """
    # For Sequential models or simple functional models
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        
        # If it's a nested model (like MobileNetV2 base)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
                    
    # Fallback to check by layer names or types
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() or 'activation' in layer.name.lower():
            return layer.name
            
    raise ValueError("Could not find any convolutional layer in the model.")

def get_layer_by_name(model, name):
    """Recursively search for a layer by name, including inside nested models."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if sub_layer.name == name:
                    return sub_layer
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes the Grad-CAM heatmap for a given image and model.
    """
    # Find the layer recursively
    target_layer = get_layer_by_name(model, last_conv_layer_name)
    if target_layer is None:
        raise ValueError(f"No such layer: {last_conv_layer_name}")

    # Check if the first layer is a nested base model (like in transfer learning)
    has_nested = hasattr(model.layers[0], 'layers')

    if has_nested:
        base_model = model.layers[0]
        grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )
    else:
        # Find index of target layer in the flat model
        target_idx = -1
        for idx, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                target_idx = idx
                break
        
        if target_idx == -1:
            raise ValueError(f"Target layer {last_conv_layer_name} not found in model layers.")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=target_layer.output
        )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        if has_nested:
            last_conv_layer_output, base_output = grad_model(img_array)
            # Pass through the rest of the Sequential model layers
            x = base_output
            for layer in model.layers[1:]:
                x = layer(x)
            preds = x
        else:
            last_conv_layer_output = grad_model(img_array)
            # Pass through the remaining layers after the target conv layer
            x = last_conv_layer_output
            for layer in model.layers[target_idx + 1:]:
                x = layer(x)
            preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # We apply ReLU to keep only positive activation features
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    """
    Superimposes the Grad-CAM heatmap onto the original image and saves it.
    """
    # Load the original image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Original image not found at {img_path}")
        
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to original image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Save the superimposed image
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path
