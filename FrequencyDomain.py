# Import necessary libraries
import numpy as np                  # For numerical calculations and Fourier Transform
import matplotlib.pyplot as plt     # For data visualization
from matplotlib.gridspec import GridSpec  # For flexible subplot layout

# Configure plot settings (ensure proper display of labels in VSCode)
plt.rcParams["font.family"] = ["Arial", "Helvetica"]  # Use English fonts
plt.rcParams["axes.unicode_minus"] = False  # Correctly display negative signs


def generate_time_domain_signal(sample_rate, duration):
    """
    Generate a time-domain signal: composed of superimposed sine waves with added noise
    
    Parameters:
        sample_rate: Sampling frequency (Hz), number of samples collected per second
        duration: Signal duration (seconds)
    
    Returns:
        t: Time array
        signal: Generated time-domain signal
    """
    # Generate time array: from 0 to duration, with sample_rate*duration points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Define frequency components in the signal (Hz)
    freq1 = 5    # 5Hz sine wave
    freq2 = 12   # 12Hz sine wave
    freq3 = 30   # 30Hz sine wave
    
    # Generate sine waves for each frequency component (with different amplitudes)
    signal_clean = (
        2 * np.sin(2 * np.pi * freq1 * t) +  # 5Hz signal with amplitude 2
        1.5 * np.sin(2 * np.pi * freq2 * t) +  # 12Hz signal with amplitude 1.5
        0.8 * np.sin(2 * np.pi * freq3 * t)    # 30Hz signal with amplitude 0.8
    )
    
    # Add Gaussian noise to simulate signal interference in real-world scenarios
    noise = 0.5 * np.random.randn(len(t))  # Noise with amplitude 0.5
    signal = signal_clean + noise
    
    return t, signal


def fourier_transform_analysis(signal, sample_rate):
    """
    Perform Fourier Transform on the time-domain signal to extract frequency-domain features
    
    Parameters:
        signal: Time-domain signal array
        sample_rate: Sampling frequency (Hz)
    
    Returns:
        frequencies: Frequency array (Hz)
        amplitude_spectrum: Amplitude spectrum (frequency-domain amplitude)
    """
    # Calculate the length of the signal
    n = len(signal)
    
    # Perform Fast Fourier Transform (FFT)
    # The FFT result is a complex number, containing amplitude and phase information
    fft_result = np.fft.fft(signal)
    
    # Calculate amplitude spectrum: take the absolute value (magnitude) of the complex numbers
    # Since the FFT result is symmetric, we only need the first half (positive frequencies)
    # Multiply by 2 (except for the DC component) to correct the amplitude (energy is distributed between positive and negative frequencies)
    amplitude = np.abs(fft_result) / n  # Normalization
    amplitude_spectrum = amplitude[:n//2] * 2  # Take the first half and correct amplitude
    amplitude_spectrum[0] /= 2  # No need to multiply the DC component by 2
    
    # Calculate frequency axis: use fftfreq to generate corresponding frequency values
    # The frequency range is [-sample_rate/2, sample_rate/2); we only take positive frequencies
    frequencies = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    
    return frequencies, amplitude_spectrum


def visualize_results(t, signal, frequencies, amplitude_spectrum):
    """Visualize the time-domain signal and frequency-domain amplitude spectrum"""
    # Create a 2-row, 1-column subplot layout
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)  # Adjust spacing between subplots
    
    # Plot the time-domain signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, signal, color='b', linewidth=0.8)
    ax1.set_title('Time-Domain Signal', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, max(t))  # Set x-axis range to the signal duration
    
    # Plot the frequency-domain amplitude spectrum
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(frequencies, amplitude_spectrum, color='r', linewidth=1.0)
    ax2.set_title('Frequency-Domain Amplitude Spectrum (Fourier Transform Result)', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(0, max(frequencies))  # Set x-axis range to the maximum frequency
    
    # Highlight main frequency components (label frequencies at peak positions)
    # Identify peak positions in the amplitude spectrum (points above the threshold)
    threshold = 0.3  # Threshold can be adjusted based on the actual signal
    peaks = np.where(amplitude_spectrum > threshold)[0]
    for peak_idx in peaks:
        freq = frequencies[peak_idx]
        amp = amplitude_spectrum[peak_idx]
        ax2.annotate(
            f'{freq:.1f}Hz', 
            xy=(freq, amp),
            xytext=(freq+1, amp+0.2),
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function: coordinate signal generation, Fourier Transform, and result visualization"""
    # Signal parameter settings
    sample_rate = 100  # Sampling frequency: 100Hz (100 samples per second)
    duration = 2       # Signal duration: 2 seconds
    
    # 1. Generate time-domain signal
    t, signal = generate_time_domain_signal(sample_rate, duration)
    
    # 2. Perform Fourier Transform analysis
    frequencies, amplitude_spectrum = fourier_transform_analysis(signal, sample_rate)
    
    # 3. Visualize results
    visualize_results(t, signal, frequencies, amplitude_spectrum)
    
    # Print key information
    print(f"Sampling Frequency: {sample_rate} Hz")
    print(f"Signal Duration: {duration} seconds")
    print(f"Total Number of Samples: {len(signal)}")
    print(f"Frequency Resolution: {frequencies[1] - frequencies[0]:.2f} Hz")  # Interval between adjacent frequencies


if __name__ == "__main__":
    main()