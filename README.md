# PyFilter

PyFilter is an interactive Augmented Reality (AR) filter application using Python, OpenCV, and MediaPipe. It allows you to apply real-time visual effects to a selected region of your webcam feed using hand gestures or mouse clicks.

## Features

- **Hand Gesture Control:** Use your hand(s) to select a region on the webcam feed and apply filters by pinching or clicking on on-screen buttons.
- **Filter Selection:** Choose from 5 different filters:
  1. Posterize
  2. Grayscale
  3. Negative
  4. Threshold (Black & White)
  5. Face Landmarks Overlay
- **Smooth Transitions:** Filter transitions and region selection are smoothed for a better user experience.
- **UI Buttons:** Select filters using on-screen buttons with your mouse or hand gestures.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy

Install dependencies with:

```bash
pip install opencv-python mediapipe numpy
```

## Controls

- **Mouse:**
  - Click on the on-screen numbered buttons (1-5) at the bottom to change filters.
- **Keyboard:**
  - Press keys `1` to `5` to switch filters directly.
  - Press `q` to quit the application.
- **Hand Gestures:**
  - Pinch gesture (thumb and finger tips close together) over a button to select a filter.
  - Use thumb and index finger (or both hands) to define the region for filter application.

## Usage

Run the application:

```bash
python filter.py
```

- Use your webcam to show your hand(s).
- Pinch with your thumb and finger(s) to select and move the filter region.
- Click or pinch on the on-screen buttons (1-5) to change filters.
- Press `q` to quit.

## File Structure

- `filter.py` - Main application file.
- `README.md` - Project documentation.

## Notes

- Make sure your webcam is connected and accessible.
- The application window may need focus for mouse events to work.
- For best results, use in a well-lit environment.

## License

MIT License
