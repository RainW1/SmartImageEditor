import cv2
import glob
from filters import LinearFilters, NonLinearFilters, EdgeDetection  # ✅ Simple import!

def test_filters_save():
    # Find images
    images = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
    
    if not images:
        print("No images found!")
        return
    
    image_path = images[0]
    print(f"Using: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image!")
        return
    
    print(f"Loaded: {img.shape}")
    print("Applying filters...")
    
    # Create instances
    linear = LinearFilters()
    nonlinear = NonLinearFilters()
    edge = EdgeDetection()
    
    # Apply filters
    cv2.imwrite('output_mean.jpg', linear.mean_filter(img, 5))
    print("1. Mean filter done")
    
    cv2.imwrite('output_gaussian.jpg', linear.gaussian_filter(img, 5, 1.5))
    print("2. Gaussian filter done")
    
    cv2.imwrite('output_sharpen.jpg', linear.sharpen_filter(img))
    print("3. Sharpen filter done")
    
    cv2.imwrite('output_median.jpg', nonlinear.median_filter(img, 5))
    print("4. Median filter done")
    
    cv2.imwrite('output_sobel.jpg', edge.sobel_edge(img))
    print("5. Sobel edge done")
    
    cv2.imwrite('output_prewitt.jpg', edge.prewitt_edge(img))
    print("6. Prewitt edge done")
    
    cv2.imwrite('output_laplacian.jpg', edge.laplacian_edge(img))
    print("7. Laplacian edge done")
    
    print("\nALL DONE! Check output_*.jpg files")

if __name__ == "__main__":
    test_filters_save()