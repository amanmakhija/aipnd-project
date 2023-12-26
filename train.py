import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import argparse

def load_data(data_directory):
    # Data transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets using ImageFolder
    train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_directory + '/valid', transform=valid_test_transforms)
    test_data = datasets.ImageFolder(data_directory + '/test', transform=valid_test_transforms)

    # DataLoaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=32)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader, train_data.class_to_idx

def create_model(architecture, hidden_units):
    # Load a pre-trained model
    model = getattr(models, architecture)(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the classifier
    model.classifier = classifier
    
    return model

def train(model, trainloader, validloader, criterion, optimizer, epochs, device='cuda'):
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 20

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        validation_loss += criterion(outputs, labels).item()
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

def save_model_checkpoint(model, train_data, save_directory, architecture, hidden_units, epochs, optimizer):
    checkpoint = {
        'architecture': architecture,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }

    torch.save(checkpoint, save_directory)

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a flower image dataset')
    parser.add_argument('data_directory', type=str, help='Path to the image dataset directory')
    parser.add_argument('--save_directory', type=str, default='checkpoint.pth', help='Directory to save the model checkpoint')
    parser.add_argument('--architecture', type=str, default='vgg16', help='Pre-trained model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training the model')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

    # Load data
    trainloader, validloader, _, train_data = load_data(args.data_directory)

    # Build the model
    model = create_model(args.architecture, args.hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train(model, trainloader, validloader, criterion, optimizer, args.epochs, device)

    # Save the model checkpoint
    save_model_checkpoint(model, train_data, args.save_directory, args.architecture, args.hidden_units, args.epochs, optimizer)


main()
