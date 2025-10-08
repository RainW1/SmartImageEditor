import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for proper display (no font issues for English)
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display


def load_and_preprocess(image_path):
    """Load the image and preprocess it (convert to grayscale + binarization; morphological ops work best on binary images)"""
    # Read the image (OpenCV uses BGR format by default)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}\nPlease check if the path is correct.")
    
    # Convert to grayscale (morphological operations typically run on single-channel images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarization: Convert grayscale image to black-white binary image
    # Threshold = 127: Pixels > 127 → 255 (white, foreground); Pixels < 127 → 0 (black, background)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return img, gray, binary


def create_kernel(kernel_size=(3, 3), kernel_type="rect"):
    """Create a structuring element (Kernel): the "tool" for morphological operations"""
    if kernel_type == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # Rectangular kernel
    elif kernel_type == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)  # Elliptical kernel
    elif kernel_type == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)  # Cross-shaped kernel
    else:
        raise ValueError("Kernel type only supports 'rect', 'ellipse', or 'cross'")
    
    return kernel


def morphological_operations(binary_img, kernel):
    """Implement 7 common morphological operations"""
    # 1. Erosion: Shrinks foreground objects
    erosion = cv2.erode(binary_img, kernel, iterations=1)
    # 2. Dilation: Expands foreground objects
    dilation = cv2.dilate(binary_img, kernel, iterations=1)
    # 3. Opening: Erosion → Dilation (removes background noise)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    # 4. Closing: Dilation → Erosion (fills foreground holes)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # 5. Morphological Gradient: Dilation - Erosion (extracts edges)
    gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel)
    # 6. Top Hat: Original - Opening (highlights bright small regions)
    top_hat = cv2.morphologyEx(binary_img, cv2.MORPH_TOPHAT, kernel)
    # 7. Black Hat: Closing - Original (highlights dark small regions)
    black_hat = cv2.morphologyEx(binary_img, cv2.MORPH_BLACKHAT, kernel)
    
    return {
     
        "Erosion": erosion,
        "Dilation": dilation,
        "Opening": opening,
        "Closing": closing,
        "Morphological Gradient": gradient,
        "Top Hat": top_hat,
        "Black Hat": black_hat
    }


def show_results(original_img, gray_img, operations_results):
    """Display all results in a 3x3 grid"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # 1. Original color image (convert BGR → RGB for matplotlib)
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Color Image")
    axes[0].axis("off")
    
    # 2. Grayscale image
    axes[1].imshow(gray_img, cmap="gray")
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")
    
    # 3-9. Morphological operation results
    for i, (title, img) in enumerate(operations_results.items(), start=2):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    # --------------------------
    # MODIFIED: Your image path
    # --------------------------
    image_path = r"photo.jpeg"
    kernel_size = (5, 5)  # Adjustable (e.g., (3,3) for weaker effect, (7,7) for stronger)
    kernel_type = "rect"  # Optional: "rect", "ellipse", "cross"
    
    try:
        # Step 1: Load and preprocess image
        original_img, gray_img, binary_img = load_and_preprocess(image_path)
        
        # Step 2: Create structuring element
        kernel = create_kernel(kernel_size, kernel_type)
        print(f"Using Kernel: Type={kernel_type}, Size={kernel_size}")
        print("Kernel Matrix:\n", kernel)
        
        # Step 3: Run morphological operations
        results = morphological_operations(binary_img, kernel)
        
        # Step 4: Show results
        show_results(original_img, gray_img, results)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()