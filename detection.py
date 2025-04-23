import cv2
import torch
from ultralytics import YOLO
from itertools import combinations

def find_combinations(tiles, target):
    """
    Returns all combinations of tiles whose sum equals the target.
    """
    result = []
    for r in range(1, len(tiles) + 1):
        for combo in combinations(tiles, r):
            if sum(combo) == target:
                result.append(combo)
    return result

def draw_text_with_background(image, text_lines, start_point, font_scale=0.7, thickness=2,
                              font=cv2.FONT_HERSHEY_SIMPLEX, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draws multi-line text with a semi-transparent background.
    """
    x, y = start_point
    margin = 5
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
    max_width = max(w for (w, h) in sizes)
    total_height = sum(h for (w, h) in sizes) + margin * (len(text_lines) - 1)
    
    overlay = image.copy()
    cv2.rectangle(overlay, (x - margin, y - margin), (x + max_width + margin, y + total_height + margin), bg_color, -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    y_offset = 0
    for line, (w, h) in zip(text_lines, sizes):
        cv2.putText(image, line, (x, y + y_offset + h), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset += h + margin
    return image

def main():
    # Load the YOLO model.
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Open webcam on device index 1.
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set up initial closed tiles (1 through 9).
    closed_tiles = list(range(1, 10))
    candidate_move = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run YOLO inference.
        results = model(frame, show=False)
        annotated_frame = results[0].plot()

        total_value = 0
        dice_values = []
        
        boxes = results[0].boxes
        if boxes is not None:
            dets = boxes.data.cpu().numpy() if isinstance(boxes.data, torch.Tensor) else boxes.data
            for det in dets:
                cls = int(det[5])
                label = results[0].names[cls]
                try:
                    value = int(label)
                except ValueError:
                    value = 0
                total_value += value
                dice_values.append(value)
        
        text_lines = [f"Total Outcome (Addition): {total_value}"]
        candidate_move = None  # reset candidate move each frame

        if total_value == 0:
            text_lines.append("Not enough dice for addition.")
        else:
            # Try Addition
            if total_value in closed_tiles:
                candidate_move = (total_value,)
                text_lines.append(f"Move (Addition): {total_value}")
            else:
                combos = find_combinations(closed_tiles, total_value)
                if combos:
                    candidate_move = combos[0]
                    text_lines.append("Move (Addition): " + "+".join(map(str, candidate_move)))
                else:
                    text_lines.append(f"Addition: {total_value} -> No valid move.")
            
            # Fallback to Subtraction if Addition fails.
            if candidate_move is None and len(dice_values) >= 2:
                subtraction_target = abs(dice_values[0] - dice_values[1])
                if subtraction_target in closed_tiles:
                    candidate_move = (subtraction_target,)
                    text_lines.append(f"Move (Subtraction): {subtraction_target}")
                else:
                    combos_sub = find_combinations(closed_tiles, subtraction_target)
                    if combos_sub:
                        candidate_move = combos_sub[0]
                        text_lines.append("Move (Subtraction): " + "+".join(map(str, candidate_move)))
                    else:
                        text_lines.append(f"Subtraction: {subtraction_target} -> No valid move.")
            
            # Fallback to Multiplication if both Addition and Subtraction fail.
            if candidate_move is None and len(dice_values) >= 2:
                multiplication_target = dice_values[0] * dice_values[1]
                if multiplication_target in closed_tiles:
                    candidate_move = (multiplication_target,)
                    text_lines.append(f"Move (Multiplication): {multiplication_target}")
                else:
                    combos_mul = find_combinations(closed_tiles, multiplication_target)
                    if combos_mul:
                        candidate_move = combos_mul[0]
                        text_lines.append("Move (Multiplication): " + "+".join(map(str, candidate_move)))
                    else:
                        text_lines.append(f"Multiplication: {multiplication_target} -> No valid move.")
            
            if candidate_move is not None:
                text_lines.append("Press 'c' to confirm move.")
            else:
                text_lines.append("Press 'r' to restart game.")

        annotated_frame = draw_text_with_background(annotated_frame, text_lines, (10, 10))
        cv2.imshow("YOLOv8 Dice Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and candidate_move:
            # Confirm move: Remove candidate move tiles.
            for tile in candidate_move:
                if tile in closed_tiles:
                    closed_tiles.remove(tile)
            print(f"Move confirmed: {candidate_move}. Remaining: {closed_tiles}")
            candidate_move = None
        elif key == ord('r'):
            print("Restarting game...")
            closed_tiles = list(range(1, 10))
            candidate_move = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()