from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torch
import pathlib
import os
import torch.nn as nn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Checking device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transforms for image preprocessing
transformer = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define paths for your dataset
train_paths = [
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\30SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\50SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\70SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\AccessDenied",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Bumper",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\CloseRoad",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\LeftSign",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\OneWayRoad",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Parking",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\PedestrianCrossWalk",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\RightSign",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Roundabout",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Stop",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Uneven",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Train\Yield"
]
test_paths = [
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\30SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\50SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test70SpeedLimit",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\AccessDenied",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Bumper",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\CloseRoad",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\LeftSign",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\TestOneWayRoad",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Parking",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\PedestrianCrossWalk",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\RightSign",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Roundabout",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Stop",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Uneven",
    r"C:\Users\momin\OneDrive\Desktop\NewPro\ProjectPro\NewDataset\NewDataset\Test\Yield"
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
model = ConvNet(input_dim=3, output_dim=15)
checkpoint_path = r'C:\Users\momin\OneDrive\Desktop\ProjectDB\NewProject2\NewProject\bestestt.model'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)

# Move model to GPU if available
model.to(device)

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
        if index < len(classes):
            pred_class = classes[index]
        else:
            pred_class = "Picture not matched"

    return pred_class

# Ensure the uploads directory exists
if not os.path.exists("./uploads"):
    os.makedirs("./uploads")

# Route for the root URL
@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Welcome to the object detection API!'})

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file part in the request'})

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded file locally
        file_path = os.path.join("./uploads", "photo.jpg")
        file.save(file_path)

        # Perform prediction
        try:
            prediction_result = prediction(file_path, transformer)
            return jsonify({'prediction': prediction_result})
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Prediction error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
