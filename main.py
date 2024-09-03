import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import numpy as np

# Define the class labels
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

checkpoint = torch.load('final_model.pth', map_location=torch.device('cpu'),weights_only = True)
model.load_state_dict(checkpoint)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_img = transform(image)
    input_img = input_img.unsqueeze(0)  # Add a batch dimension
    return image, input_img

# Define a custom forward function that extracts the desired output for a specific detection
def custom_forward(input_tensor, index):
        outputs = model(input_tensor)
        if len(outputs[0]['scores']) > index:
            return outputs[0]['scores'][index].unsqueeze(0)
        else:
            return torch.tensor([0.0])

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
attributions_np = total_attributions.squeeze().detach().numpy()
attributions_sum = np.sum(np.abs(attributions_np), axis=0)

# Normalize the attributions to [0, 1] for better visualization
attributions_norm = (attributions_sum - attributions_sum.min()) / (attributions_sum.max() - attributions_sum.min())


# Create a heatmap of attributions
plt.figure(figsize=(8, 8))
plt.imshow(original_image)
plt.imshow(attributions_norm, cmap='hot', alpha=0.5)
plt.axis('off')
plt.title('Attributions Heatmap')

# Save the output image
output_path = 'attributions_heatmap.png'
plt.savefig(output_path)

# Display the output image
plt.show()
