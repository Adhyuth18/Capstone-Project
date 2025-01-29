import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt

class SignatureProcessor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signature TCC & FCC")
        self.geometry("400x200")

        self.labelFrame = tk.LabelFrame(self, text="Select Signature Images")
        self.labelFrame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.button = tk.Button(self.labelFrame, text="Browse", command=self.load_images)
        self.button.pack(pady=20)
        
        self.images = []
    
    def load_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_paths:
            return
        
        self.images = [self.process_image(file_path) for file_path in file_paths]
        self.display_all()
    
    def process_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", f"Could not open image: {file_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Create False Color Composite (FCC) - Apply a false color mapping
        false_color = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        
        return (image, gray, false_color)
    
    def display_all(self):
        if not self.images:
            return
        
        fig, axes = plt.subplots(len(self.images), 3, figsize=(12, 4 * len(self.images)))
        
        for i, (image, gray, false_color) in enumerate(self.images):
            axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title("Original (TCC)")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(gray, cmap="gray")
            axes[i, 1].set_title("Grayscale")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(cv2.cvtColor(false_color, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title("False Color Composite (FCC)")
            axes[i, 2].axis("off")
        
        plt.show()

if __name__ == '__main__':
    app = SignatureProcessor()
    app.mainloop()
