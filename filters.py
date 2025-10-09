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