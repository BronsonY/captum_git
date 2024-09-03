import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, NoiseTunnel
import numpy as np
import matplotlib.patches as patches

# Define the class labels (modify as per your model's classes)
class_labels = {
    0: 'background',
    1: 'handwritten',  # Update with actual labels
    # Add more labels as needed
}

# Load and prepare the model
model = fasterrcnn_resnet50_fpn(weights=None)  # Updated 'pretrained' to 'weights'
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the model checkpoint
checkpoint = torch.load('final_model.pth', map_location=torch.device('cpu'), weights_only = True)
model.load_state_dict(checkpoint)
model.eval()

def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_img = transform(image)
    input_img = input_img.unsqueeze(0)  # Add a batch dimension
    return image, input_img

def custom_forward(input_tensor, index):
    """Custom forward function to extract output for a specific detection."""
    outputs = model(input_tensor)
    if len(outputs[0]['scores']) > index:
        return outputs[0]['scores'][index].unsqueeze(0)
    else:
        return torch.tensor([0.0])

def plot_heatmap(original_image, attributions, alpha_value=0.7, cmap='jet', title='Attributions Heatmap'):
    """Function to plot the heatmap."""
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(attributions, cmap=cmap, alpha=alpha_value)
    plt.axis('off')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

def plot_bounding_boxes_with_attributions(image, attributions, outputs):
    """Plot bounding boxes with attribution scores."""
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    
    for i, box in enumerate(outputs[0]['boxes']):
        score = outputs[0]['scores'][i].item()
        attribution_score = attributions[i]
        color = 'red' if score > 0.5 else 'blue'
        linewidth = 2 + 3 * attribution_score  # Adjust linewidth based on attribution
        
        # Create a rectangle patch for bounding box
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.title('Bounding Boxes with Attributions')
    plt.savefig('bounding_boxes_with_attributions.png')
    plt.show()

# Specify the path to your image file
image_path = 'form.jpeg'

# Preprocess the image to get the PIL image and the input tensor
original_image, input_tensor = preprocess_image(image_path)

# Define the baseline, often zeros or some other meaningful baseline for the domain
baseline = torch.zeros_like(input_tensor)

# Initialize the total attributions tensor
total_attributions = torch.zeros_like(input_tensor)

# Compute attributions for each detection and accumulate them
outputs = model(input_tensor)
for i in range(len(outputs[0]['scores'])):
    integrated_gradients = IntegratedGradients(lambda x: custom_forward(x, i))
    attributions = integrated_gradients.attribute(input_tensor, baseline, n_steps=5)
    total_attributions += attributions
    
# Convert the accumulated attributions to numpy and aggregate over channels
attributions_np = total_attributions.squeeze().detach().numpy()  # Added detach()
attributions_sum = np.sum(np.abs(attributions_np), axis=0)

# Normalize the attributions to [0, 1] for better visualization
attributions_norm = (attributions_sum - attributions_sum.min()) / (attributions_sum.max() - attributions_sum.min())

# Plot the attributions heatmap
plot_heatmap(original_image, attributions_norm)

# Apply thresholding to focus on areas with higher attributions
threshold = 0.5
attributions_thresholded = np.where(attributions_norm > threshold, attributions_norm, 0)

# Plot the thresholded attributions heatmap
plot_heatmap(original_image, attributions_thresholded, title='Thresholded Attributions Heatmap')

# Normalize attributions to scale between 0 and 1 for each detected object
attribution_scores = [attributions_norm[int((box[1] + box[3]) / 2), int((box[0] + box[2]) / 2)] for box in outputs[0]['boxes']]

# Plot bounding boxes with attributions
plot_bounding_boxes_with_attributions(original_image, attribution_scores, outputs)

# Apply SmoothGrad for smoother attributions
noise_tunnel = NoiseTunnel(IntegratedGradients(lambda x: custom_forward(x, 0)))

# Calculate SmoothGrad attributions
smooth_attributions = noise_tunnel.attribute(input_tensor, nt_type='smoothgrad', n_samples=10, stdevs=0.1)

# Convert smooth attributions to numpy for visualization
smooth_attributions_np = smooth_attributions.squeeze().detach().numpy()  # Added detach()
smooth_attributions_sum = np.sum(np.abs(smooth_attributions_np), axis=0)

# Normalize the smooth attributions
smooth_attributions_norm = (smooth_attributions_sum - smooth_attributions_sum.min()) / (smooth_attributions_sum.max() - smooth_attributions_sum.min())

# Plot the SmoothGrad attributions heatmap
plot_heatmap(original_image, smooth_attributions_norm, title='SmoothGrad Attributions Heatmap')
