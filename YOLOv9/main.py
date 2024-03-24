from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov9c.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='dataset/data.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()
