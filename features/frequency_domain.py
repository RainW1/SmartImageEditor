# File: features/frequency_domain.py
# Frequency domain analysis untuk GUI

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class FrequencyDomainAnalysis:
    """Class for frequency domain operations and analysis"""
    
    @staticmethod
    def fourier_transform(image):
        """
        Perform 2D Fourier Transform on image
        
        Args:
            image: Input image (grayscale)
        
        Returns:
            fft_result: Complex FFT result
            magnitude_spectrum: Magnitude spectrum for visualization
            phase_spectrum: Phase spectrum
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Perform 2D FFT
        fft_result = np.fft.fft2(gray)
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_result)
        
        # Calculate magnitude spectrum (for visualization)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1)  # Log scale
        
        # Calculate phase spectrum
        phase_spectrum = np.angle(fft_shifted)
        
        return fft_shifted, magnitude_spectrum, phase_spectrum
    
    @staticmethod
    def inverse_fourier_transform(fft_shifted):
        """
        Perform inverse FFT to reconstruct image
        
        Args:
            fft_shifted: Shifted FFT result
        
        Returns:
            Reconstructed image
        """
        # Inverse shift
        fft_result = np.fft.ifftshift(fft_shifted)
        
        # Inverse FFT
        image_back = np.fft.ifft2(fft_result)
        image_back = np.abs(image_back)
        
        return image_back.astype(np.uint8)
    
    @staticmethod
    def create_filter_mask(shape, filter_type="lowpass", cutoff=30):
        """
        Create frequency domain filter mask
        
        Args:
            shape: Image shape (height, width)
            filter_type: "lowpass" or "highpass"
            cutoff: Cutoff frequency (radius in pixels)
        
        Returns:
            Filter mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:rows, :cols]
        
        # Calculate distance from center
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Create mask
        if filter_type == "lowpass":
            mask = (distance <= cutoff).astype(float)
        else:  # highpass
            mask = (distance > cutoff).astype(float)
        
        return mask
    
    @staticmethod
    def apply_frequency_filter(image, filter_type="lowpass", cutoff=30):
        """
        Apply frequency domain filter
        
        Args:
            image: Input image
            filter_type: "lowpass" (blur) or "highpass" (sharpen/edges)
            cutoff: Cutoff frequency
        
        Returns:
            Filtered image
        """
        # Get FFT
        fft_shifted, _, _ = FrequencyDomainAnalysis.fourier_transform(image)
        
        # Create filter mask
        mask = FrequencyDomainAnalysis.create_filter_mask(
            image.shape[:2], filter_type, cutoff
        )
        
        # Apply filter
        fft_filtered = fft_shifted * mask
        
        # Inverse FFT
        filtered_image = FrequencyDomainAnalysis.inverse_fourier_transform(fft_filtered)
        
        return filtered_image, fft_filtered, mask
    
    @staticmethod
    def visualize_frequency_analysis(image, title="Frequency Domain Analysis"):
        """
        Create comprehensive frequency domain visualization
        
        Args:
            image: Input image
            title: Window title
        
        Returns:
            matplotlib Figure object
        """
        # Perform FFT
        fft_shifted, magnitude_spectrum, phase_spectrum = \
            FrequencyDomainAnalysis.fourier_transform(image)
        
        # Convert to grayscale for display
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8))
        
        # Original image
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(gray, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Magnitude spectrum
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='hot')
        ax2.set_title('Magnitude Spectrum (Log Scale)')
        ax2.axis('off')
        
        # Phase spectrum
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(phase_spectrum, cmap='hsv')
        ax3.set_title('Phase Spectrum')
        ax3.axis('off')
        
        # 1D frequency profile (horizontal center line)
        ax4 = fig.add_subplot(2, 2, 4)
        center_row = magnitude_spectrum[magnitude_spectrum.shape[0]//2, :]
        ax4.plot(center_row)
        ax4.set_title('Frequency Profile (Horizontal)')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Magnitude')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def visualize_filter_comparison(image, filter_type="lowpass", cutoff=30):
        """
        Compare original, filtered, and filter mask
        
        Args:
            image: Input image
            filter_type: "lowpass" or "highpass"
            cutoff: Cutoff frequency
        
        Returns:
            matplotlib Figure object
        """
        # Apply filter
        filtered, fft_filtered, mask = FrequencyDomainAnalysis.apply_frequency_filter(
            image, filter_type, cutoff
        )
        
        # Get magnitude spectrums
        _, original_spectrum, _ = FrequencyDomainAnalysis.fourier_transform(image)
        filtered_spectrum = 20 * np.log(np.abs(fft_filtered) + 1)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create figure
        fig = Figure(figsize=(12, 8))
        
        # Original image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(gray, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Original spectrum
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(original_spectrum, cmap='hot')
        ax2.set_title('Original Spectrum')
        ax2.axis('off')
        
        # Filter mask
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(mask, cmap='gray')
        ax3.set_title(f'{filter_type.title()} Filter\n(Cutoff={cutoff})')
        ax3.axis('off')
        
        # Filtered image
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(filtered, cmap='gray')
        ax4.set_title('Filtered Image')
        ax4.axis('off')
        
        # Filtered spectrum
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(filtered_spectrum, cmap='hot')
        ax5.set_title('Filtered Spectrum')
        ax5.axis('off')
        
        # Difference
        ax6 = fig.add_subplot(2, 3, 6)
        diff = np.abs(gray.astype(float) - filtered.astype(float))
        ax6.imshow(diff, cmap='hot')
        ax6.set_title('Difference')
        ax6.axis('off')
        
        filter_name = "Low-Pass (Blur)" if filter_type == "lowpass" else "High-Pass (Edges)"
        fig.suptitle(f'Frequency Domain Filtering - {filter_name}', 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig, filtered