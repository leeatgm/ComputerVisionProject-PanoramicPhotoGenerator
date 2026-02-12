# 🖼️ Panoramic Photo Generator

This project generates panoramic images from input videos using computer vision techniques. It extracts relevant frames from a video, aligns them, and stitches them together to create a single panoramic image.

Built with Python and OpenCV.

---

## 📌 Project Overview

The system:

- Takes a panning or 360° video as input  
- Extracts frames based on a configurable stride  
- Detects and matches features between frames  
- Aligns and stitches frames into a panorama  
- Outputs the final panoramic image  

The quality of the result depends on proper stride and threshold configuration for each video.

---

## 🚀 How to Run

There are **two entry points** to run the program:

### 1️⃣ With User Interface  

Run:

`ui.py`

This opens the graphical interface for generating panoramas interactively.

---

### 2️⃣ Without User Interface  

Run:

`main.py`

This runs the program directly without UI.

---

## ⚙️ Configuration

To modify parameters such as:

- `stride`
- Matching thresholds
- Other tuning parameters

Edit them directly inside:

`main.py`

⚠️ **Important:**  
Each video requires a different stride value to produce the best result.  
Always adjust the stride before running the program.

---

## ❗ Important Notes

- Do **not** run the program from terminal or command line.  
  It may incorrectly show that the `cv2` module is not installed.
- Run the project using an IDE instead.
- **PyCharm is recommended.**

---

## 📦 Requirements

- Python  
- OpenCV (`cv2`)  
- Other required imaging or GUI libraries used in the project
- 
