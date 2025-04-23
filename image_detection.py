import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

def main():
    # Initialize a hidden Tkinter root window for file dialog.
    root = tk.Tk()
    root.withdraw()
    
    # Ask the user to select an image file.
    image_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    # If no image is selected, exit.
    if not image_path:
        print("No image selected. Exiting.")
        return
    else:
        print("Selected image:", image_path)

    # Load your trained YOLO model. Update the path as needed.
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Load the image using OpenCV.
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        return

    # Run inference on the image (set show=False to allow manual annotation).
    results = model(image, show=False)
    
    # Obtain the annotated image using the model's built-in plot() function.
    annotated_image = results[0].plot()

    # Calculate the total outcome from the detected dice.
    total_value = 0
    boxes = results[0].boxes
    if boxes is not None:
        dets = boxes.data
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        for det in dets:
            # Get the class index from the detection.
            cls = int(det[5])
            # Assume that the class label represents a dice value ("1", "2", etc.).
            label = results[0].names[cls]
            try:
                value = int(label)
            except ValueError:
                value = 0
            total_value += value

    # Overlay the total outcome on the annotated image.
    cv2.putText(annotated_image, f"Total Outcome: {total_value}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the annotated image.
    cv2.imshow("YOLOv8 Dice Detection - Image Test", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
