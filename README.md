# Shut_The_Box

## Features

- **Dice Detection:**  
  Uses YOLOv8 (variants such as `yolov8m.pt`,`yolov8n.pt`, and `yolov8l.pt`) to detect dice in both images and real-time webcam feeds. This modal is trained by using `yolov8l.pt`.

- **Arithmetic-Based Game Logic:**  
  The game computes the total outcome from detected dice using addition. Then it will ask the user to save the move by pressing C on the keyboard. If addition does not result in a valid candidate move (i.e. no available tile or combination matches the computed total), the system automatically falls back to subtraction and multiplication to suggest valid moves. If no valid moves left then it will ask the user to restart the game by pressing R on the keyboard.

- **Game State Management:**  
  The project keeps track of which tiles (typically numbered 1 through 9) are available ("closed"). Once a candidate move is confirmed by the user, the corresponding tile(s) are removed from the available set.

- **User-Friendly Interaction:**  
  - **Webcam Mode:** Real-time detection with key commands to confirm moves, restart gameplay, or quit.
  - **Image Mode:** An interactive file dialog lets you choose an image for detection.


*Note: Adjust file/directory names and paths as needed.*

## Installation

### Prerequisites

- **Python:** Version 3.8 or later  
- **pip:** For installing Python packages  
- **GPU:** A CUDA-capable GPU is recommended for faster training/inference (CPU mode works too)

### Installing Dependencies

The project relies on several Python libraries. The required packages are listed in `requirements.txt`:


To install the dependencies, run the following command in your terminal:
bash

```
pip install -r requirements.txt

```  

# Start the training with
```  
train.py

```   

# Running Dice Detection (Webcam) 
```  
detection.py

```  
# Running Dice Detection (Image)
```  
image_detection.py

```

## Project status
This project is still under development and can be used to train with different modal.


## License
For open source projects, say how it is licensed.
