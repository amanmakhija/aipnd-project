import torch
from torchvision import models, transforms
from PIL import Image
import json
import argparse

def load_model_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Load a pre-trained model
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Load the state dictionary
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def preprocess_image(image_path):
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path).convert("RGB")
    
    # Define the transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply the transformation
    img_tensor = preprocess(image)
    
    return img_tensor

def make_prediction(image_path, model, top_k=5, device="cpu"):
    # Predict the class (or classes) of an image using a trained deep learning model
    
    # Set the model to evaluation mode
    model.eval()
    
    # Process the image
    img_tensor = preprocess_image(image_path)
    
    # Add a batch dimension and move the tensor to the specified device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Make the prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Calculate the probabilities and indices of the top-k predictions
    probabilities, indices = torch.topk(torch.nn.functional.softmax(output[0], dim=0), top_k)
    
    # Move the tensors to the CPU and convert to numpy
    probabilities = probabilities.cpu().numpy()
    indices = indices.cpu().numpy()
    
    # Invert the class_to_idx mapping
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Map indices to classes
    classes = [idx_to_class[idx.item()] for idx in indices]
    
    return probabilities, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower names from an image using a trained deep learning model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_checkpoint', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    args = parser.parse_args()

    # Load the model
    model = load_model_checkpoint(args.model_checkpoint)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    # Load the category names
    with open(args.category_names, 'r') as file:
        category_to_name = json.load(file)

    # Make predictions
    probabilities, classes = make_prediction(args.image_path, model, args.top_k, device)

    # Map classes to names
    class_names = [category_to_name[class_] for class_ in classes]

    # Print the results
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probabilities[i]}")


main()