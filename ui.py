import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2

from main import extract_frames, stitch_images
from PostProcess import post_process

class PanoramaUI(tk.Tk):
    def __init__(self):
        super().__init__()                       # initialize main window
        self.title("Panoramic Photo Generator")  # window title
        self.geometry("400x200")                 # window size

        # video file selection section
        tk.Label(self, text="Video file:").pack(anchor="w", padx=10, pady=(10, 0))
        video_row = tk.Frame(self)
        video_row.pack(fill="x", padx=10)
        self.video_path = tk.StringVar()         # store video path
        tk.Entry(video_row, textvariable=self.video_path).pack(side="left", fill="x", expand=True)
        tk.Button(video_row, text="Browse", command=self.browse_video).pack(side="right")

        # output directory section
        tk.Label(self, text="Output dir:").pack(anchor="w", padx=10, pady=(10, 0))
        output_row = tk.Frame(self)
        output_row.pack(fill="x", padx=10)
        self.output_dir = tk.StringVar()         # store output folder
        tk.Entry(output_row, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        tk.Button(output_row, text="Browse", command=self.browse_output).pack(side="right")

        # run button
        tk.Button(self, text="Generate Image", command=self.run_pipeline).pack(pady=20)

    # choose video file
    def browse_video(self):
        # open a dialog for choosing mp4 video and filter only mp4 extension
        path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)

    # choose output folder
    def browse_output(self):
        # open folder selection dialog
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    # get and check paths
    def run_pipeline(self):
        vid = self.video_path.get().strip()                     #get 360 video path
        out = self.output_dir.get().strip()                     # get output path
        if not os.path.isfile(vid) or not os.path.isdir(out):
            messagebox.showerror("Error", "Check video file and output folder.")
            return

        # prepare frames folder
        frames_dir = os.path.join(out, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        self.update()  # refresh UI

        # extract frames with  default parameter
        extract_frames(vid, frames_dir)

        # load frames for stitching and filter for jpg only
        frame_files = [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(".jpg")
        ]
        images = [cv2.imread(f) for f in frame_files]   # read each file into memory

        # stitch and post‑process
        panorama = stitch_images(images)
        if panorama is None:
            messagebox.showerror("Error", "Stitching failed.")
            return

        # save outputs
        raw_path = os.path.join(out, "panorama.jpg")
        proc_path = os.path.join(out, "panorama_processed.jpg")
        cv2.imwrite(raw_path, panorama)                             # write raw panorama
        post_process(panorama, proc_path)                           # apply post-processing

        # notify user with full file paths
        messagebox.showinfo("Done", f"Saved:\n{raw_path}\n{proc_path}")

if __name__ == "__main__":
    app = PanoramaUI()
    app.mainloop()
