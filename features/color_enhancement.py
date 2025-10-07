# File: features/color_enhancement.py
# Author: Nasyith Nabhan
# Deskripsi: Kumpulan fungsi untuk penyesuaian warna dan peningkatan gambar.

import cv2
import numpy as np

# --- 1. Fungsi Penyesuaian Warna Dasar ---

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Mengatur brightness dan contrast dari sebuah gambar.
    - brightness: -127 (gelap) hingga 127 (terang)
    - contrast: -127 (kurang kontras) hingga 127 (sangat kontras)
    """
    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    beta = brightness - contrast
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

def adjust_saturation_hue(image, saturation=0, hue=0):
    """
    Mengatur saturasi dan hue dari sebuah gambar.
    Ini memerlukan konversi ke color space HSV (Hue, Saturation, Value)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Menambahkan nilai saturasi dan memastikan tidak keluar dari rentang 0-255
    s = cv2.add(s, saturation)
    s[s > 255] = 255
    s[s < 0] = 0
    s = s.astype(np.uint8)

    # Menambahkan nilai hue dan memastikan tidak keluar dari rentang 0-179 (range hue di OpenCV)
    h = cv2.add(h, hue)
    h[h > 179] = 179
    h[h < 0] = 0
    h = h.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    new_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return new_image


# --- 2. Fungsi Peningkatan Kontras Lanjutan ---

def apply_histogram_equalization(image):
    """
    Menerapkan histogram equalization untuk meningkatkan kontras gambar secara otomatis.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)

def adjust_gamma(image, gamma=1.0):
    """
    Melakukan gamma correction pada gambar.
    - gamma < 1.0 akan membuat gambar lebih terang.
    - gamma > 1.0 akan membuat gambar lebih gelap.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# --- 3. Fungsi Thresholding (VERSI SUDAH DIPERBAIKI) ---

def apply_global_threshold(image, threshold_value=127):
    """
    Menerapkan thresholding global, mengubah gambar menjadi hitam putih.
    Fungsi ini sekarang mengembalikan gambar grayscale (1 channel).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image # Langsung kembalikan hasil grayscale

def apply_adaptive_threshold(image):
    """
    Menerapkan adaptive thresholding, lebih baik untuk kondisi pencahayaan yang tidak merata.
    Fungsi ini sekarang mengembalikan gambar grayscale (1 channel).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded_image # Langsung kembalikan hasil grayscale