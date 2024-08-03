import torch
import torch.nn as nn
import torchvision.transforms as transforms
import glob
from PIL import Image
import pathlib

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
train_paths = [
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\30SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\50SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\70SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\AccessDenied",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Bumper",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\CloseRoad",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\LeftSign",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\OneWayRoad",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Parking",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\PedestrianCrossWalk",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\RightSign",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Roundabout",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Stop",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Uneven",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Train\Yield"
]
test_paths = [
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\30SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\50SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\70SpeedLimit",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\AccessDenied",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Bumper",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\CloseRoad",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\LeftSign",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\OneWayRoad",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Parking",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\PedestrianCrossWalk",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\RightSign",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Roundabout",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Stop",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Uneven",
    r"C:\Users\ABDULLAH\Desktop\Project\NewDataset\NewDataset\Test\Yield"
]

# Define classes based on the directory names
classes = [pathlib.Path(path).stem for path in train_paths]

# CNN Model definition
class ConvNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=15):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.flatten = nn.Flatten()
        
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(1024)
        
        self.l1 = nn.Linear(1024*4*4, 512)
        self.l2 = nn.Linear(512, 128)
        self.batchnorm4 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, output_dim)
        
        self.best_val_acc = 0.0  # Initialize best validation accuracy
        
    def forward(self, input):
        conv = self.conv1(input)
        conv = self.conv2(conv)
        batchnorm = self.relu(self.batchnorm1(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv3(maxpool)
        conv = self.conv4(conv)
        batchnorm = self.relu(self.batchnorm2(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv5(maxpool)
        conv = self.conv6(conv)
        batchnorm = self.relu(self.batchnorm3(conv))
        maxpool = self.maxpool(batchnorm)
        
        flatten = self.flatten(maxpool)
        
        dense_l1 = self.l1(flatten)
        dropout = self.dropout3(dense_l1)
        dense_l2 = self.l2(dropout)
        batchnorm = self.batchnorm4(dense_l2)
        dropout = self.dropout2(batchnorm)
        output = self.l3(dropout)
        
        return output

# Load the trained model checkpoint
checkpoint = torch.load('bestestt.model')

# Instantiate the model
model = ConvNet(input_dim=3, output_dim=15)
# Load weights from the checkpoint
model.load_state_dict(checkpoint)

# Move model to GPU if available
model.to(device)

# Transforms for image preprocessing
transformer = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prediction function
def prediction(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    image_tensor = image_tensor.to(device)  # Move tensor to GPU if available

    with torch.no_grad():
        input = torch.autograd.Variable(image_tensor)
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        pred_class = classes[index]
    
    return pred_class

# Perform predictions on test images and store results in a dictionary
pred_dict = {}

for class_path in test_paths:
    images_path = glob.glob(class_path + '/*.png')
    for img_path in images_path:
        filename = img_path.split('\\')[-1]  # Extract the filename from the full path
        pred_class = prediction(img_path, transformer)
        pred_dict[filename] = pred_class

# Print the predictions
print(pred_dict)
