from ultralytics import YOLO

def main():
    # Load the YOLOv8n model (git add .you can change "yolov8l.pt" to your model file or custom model path)
    model = YOLO("yolov8l.pt")
    # Train the model on your dataset.
    # Using a raw string for the dataset path to properly handle backslashes.
    model.train(
        data=r"D:/MDX UNI/Assignments/Shut_The_Box/dataset_yolov8/data.yaml", #Path of you data.yaml file
        epochs=50,
        imgsz=640
    )

if __name__ == "__main__":
    main()