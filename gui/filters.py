import cv2
import numpy as np


class LinearFilters:
    """Class for applying linear filters to images"""
    
    @staticmethod
    def mean_filter(image, kernel_size=3):
        """Apply Mean Filter (Box Filter)"""
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and positive")
        return cv2.blur(image, (kernel_size, kernel_size))
    
    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=1.0):
        """Apply Gaussian Filter"""
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and positive")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    
    @staticmethod
    def sharpen_filter(image):
        """Apply Sharpening Filter"""
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, sharpen_kernel)


class NonLinearFilters:
    """Class for applying non-linear filters to images"""
    
    @staticmethod
    def median_filter(image, kernel_size=3):
        """Apply Median Filter"""
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and positive")
        return cv2.medianBlur(image, kernel_size)


class EdgeDetection:
    """Class for edge detection operations"""
    
    @staticmethod
    def sobel_edge(image):
        """Apply Sobel Edge Detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
        return cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def prewitt_edge(image):
        """Apply Prewitt Edge Detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_magnitude = np.uint8(np.clip(prewitt_magnitude, 0, 255))
        return cv2.cvtColor(prewitt_magnitude, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def laplacian_edge(image):
        """Apply Laplacian Edge Detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)


class GeometricTransforms:
    """Class for geometric transformations"""
    
    @staticmethod
    def crop_image(image, x, y, width, height):
        """
        Crop image to specified region
        Args:
            image: Input image
            x, y: Top-left corner coordinates
            width, height: Crop dimensions
        """
        h, w = image.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        x2 = max(0, min(x + width, w))
        y2 = max(0, min(y + height, h))
        return image[y:y2, x:x2]
    
    @staticmethod
    def resize_image(image, width=None, height=None, scale=None, interpolation=cv2.INTER_LINEAR):
        """
        Resize image
        Args:
            image: Input image
            width: Target width (if None, calculated from height or scale)
            height: Target height (if None, calculated from width or scale)
            scale: Scale factor (used if width and height are None)
            interpolation: Interpolation method (INTER_LINEAR, INTER_CUBIC, INTER_NEAREST)
        """
        h, w = image.shape[:2]
        
        if scale is not None:
            width = int(w * scale)
            height = int(h * scale)
        elif width is not None and height is None:
            height = int(h * (width / w))
        elif height is not None and width is None:
            width = int(w * (height / h))
        elif width is None and height is None:
            raise ValueError("Must specify width, height, or scale")
        
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    @staticmethod
    def rotate_image(image, angle, scale=1.0, keep_size=True):
        """
        Rotate image by specified angle
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            scale: Scale factor
            keep_size: If True, keeps original image size; if False, expands to fit
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        if not keep_size:
            # Calculate new size to fit entire rotated image
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            return cv2.warpAffine(image, M, (new_w, new_h))
        else:
            return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def flip_image(image, flip_code):
        """
        Flip image
        Args:
            image: Input image
            flip_code: 0 = vertical flip, 1 = horizontal flip, -1 = both
        """
        return cv2.flip(image, flip_code)
    
    @staticmethod
    def rotate_90(image, k=1):
        """
        Rotate image by 90 degrees
        Args:
            image: Input image
            k: Number of 90-degree rotations (1=90°, 2=180°, 3=270°)
        """
        return np.rot90(image, k)

class MorphologicalFilters:
    """Class for morphological operations on images"""
    
    @staticmethod
    def create_kernel(kernel_size=(5, 5), kernel_type="rect"):
        """
        Create structuring element (kernel) for morphological operations
        
        Args:
            kernel_size: Size of kernel (width, height)
            kernel_type: Type of kernel - "rect", "ellipse", or "cross"
        
        Returns:
            Kernel/structuring element
        """
        if kernel_type == "rect":
            return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        elif kernel_type == "ellipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        elif kernel_type == "cross":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        else:
            raise ValueError("Kernel type must be 'rect', 'ellipse', or 'cross'")
    
    @staticmethod
    def erosion(image, kernel_size=(5, 5), kernel_type="rect", iterations=1):
        """
        Erosion: Shrinks foreground objects, removes small noise
        Best for: Removing small white noise, separating objects
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.erode(image, kernel, iterations=iterations)
    
    @staticmethod
    def dilation(image, kernel_size=(5, 5), kernel_type="rect", iterations=1):
        """
        Dilation: Expands foreground objects, fills small gaps
        Best for: Connecting broken lines, filling small holes
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.dilate(image, kernel, iterations=iterations)
    
    @staticmethod
    def opening(image, kernel_size=(5, 5), kernel_type="rect"):
        """
        Opening: Erosion followed by Dilation
        Best for: Removing background noise while preserving shape
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def closing(image, kernel_size=(5, 5), kernel_type="rect"):
        """
        Closing: Dilation followed by Erosion
        Best for: Filling holes in foreground objects
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def morphological_gradient(image, kernel_size=(5, 5), kernel_type="rect"):
        """
        Morphological Gradient: Dilation - Erosion
        Best for: Edge detection, extracting object boundaries
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    @staticmethod
    def top_hat(image, kernel_size=(5, 5), kernel_type="rect"):
        """
        Top Hat: Original - Opening
        Best for: Extracting small bright details from dark background
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    @staticmethod
    def black_hat(image, kernel_size=(5, 5), kernel_type="rect"):
        """
        Black Hat: Closing - Original
        Best for: Extracting small dark details from bright background
        """
        kernel = MorphologicalFilters.create_kernel(kernel_size, kernel_type)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    @staticmethod
    def preprocess_for_morphology(image):
        """
        Convert image to binary for morphological operations
        Returns both grayscale and binary versions
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return gray, binary