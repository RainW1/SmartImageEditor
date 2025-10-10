# File: features/ai_filters.py
# Author: Min (Fixed & Optimized for GUI)
# Deskripsi: AI-based filters untuk image enhancement

import cv2
import numpy as np
from PIL import Image

# Optional imports - check availability at runtime
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ Warning: rembg not installed. Background removal disabled.")
    print("   Install: pip install rembg")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    import copy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ Warning: PyTorch not installed. Style transfer disabled.")
    print("   Install: pip install torch torchvision")


class AIColorCorrection:
    """Class untuk AI-based color correction"""
    
    @staticmethod
    def apply_clahe(img, clip_limit=3.0, tile_size=8):
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Meningkatkan kontras gambar secara adaptif
        
        Args:
            img: Input image (BGR format dari OpenCV)
            clip_limit: Contrast limiting (1.0-5.0)
            tile_size: Grid size untuk adaptive processing
        Returns:
            Enhanced image
        """
        try:
            # Convert BGR to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            cl = clahe.apply(l)
            
            # Merge channels back
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"❌ CLAHE error: {e}")
            return img

    @staticmethod
    def adjust_brightness_contrast_ai(img, brightness=0, contrast=0):
        """
        Adjust brightness dan contrast (AI version)
        
        Args:
            brightness: -50 to 50
            contrast: -50 to 50
        """
        try:
            return cv2.convertScaleAbs(img, alpha=1 + contrast / 100, beta=brightness)
        except Exception as e:
            print(f"❌ Brightness/Contrast error: {e}")
            return img

    @staticmethod
    def gamma_correction(img, gamma=1.0):
        """
        Gamma correction untuk adjust lighting curve
        
        Args:
            gamma < 1.0 = brighter (e.g., 0.5 = much brighter)
            gamma = 1.0 = no change
            gamma > 1.0 = darker (e.g., 2.0 = much darker)
        """
        try:
            invGamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(img, table)
        except Exception as e:
            print(f"❌ Gamma correction error: {e}")
            return img

    @staticmethod
    def white_balance(img):
        """Auto white balance untuk koreksi warna"""
        try:
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"❌ White balance error: {e}")
            return img

    @staticmethod
    def full_color_correction(img, clip_limit=3.0, tile_size=8, gamma=1.0, 
                            brightness=0, contrast=0, wb_toggle=True):
        """
        Complete color correction pipeline - ALL IN ONE!
        Gabungan semua enhancement sekaligus
        
        Perfect untuk:
        - Photo enhancement
        - Low light correction
        - Color grading
        """
        if img is None:
            return None
        
        try:
            print("🎨 Applying AI Color Correction...")
            corrected = AIColorCorrection.apply_clahe(img, clip_limit, tile_size)
            if wb_toggle:
                corrected = AIColorCorrection.white_balance(corrected)
            corrected = AIColorCorrection.adjust_brightness_contrast_ai(corrected, brightness, contrast)
            corrected = AIColorCorrection.gamma_correction(corrected, gamma)
            print("✅ Color correction complete!")
            return corrected
        except Exception as e:
            print(f"❌ Color correction error: {e}")
            return img


class BackgroundRemoval:
    """Class untuk background removal menggunakan rembg + AI"""
    
    @staticmethod
    def is_available():
        """Check if background removal is available"""
        return REMBG_AVAILABLE
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex color ke RGB tuple"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def remove_background(img, mode="General Mode", color_to_remove=None, strength=30):
        """
        Remove background dari gambar menggunakan AI
        
        Args:
            img: Input image (BGR format)
            mode: Model mode
                - "Portrait Mode": Khusus untuk foto orang
                - "General Mode": Untuk objek umum
                - "Product Mode": Untuk product photography
                - "Anime Mode": Untuk gambar anime/cartoon
            color_to_remove: Hex color string untuk remove specific color (optional)
            strength: Sensitivity untuk color removal (10-100)
        
        Returns:
            Image dengan background transparent (BGRA format)
        """
        if not REMBG_AVAILABLE:
            print("❌ rembg not installed! Install: pip install rembg")
            return img
        
        if img is None:
            return None

        try:
            print(f"🔲 Removing background using {mode}...")
            
            # Model mapping
            model_map = {
                "Portrait Mode": "u2net_human_seg",
                "General Mode": "u2net",
                "Product Mode": "isnet-general-use",
                "Anime Mode": "isnet-anime",
            }

            model_name = model_map.get(mode, "u2net")
            session = new_session(model_name)

            # Convert BGR to RGB for rembg
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Remove background
            output = remove(
                pil_img,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )

            # Optional: remove specific color
            if color_to_remove and isinstance(color_to_remove, str):
                try:
                    target_rgb = np.array(BackgroundRemoval.hex_to_rgb(color_to_remove))
                    img_np = np.array(output.convert("RGBA"))
                    diff = np.abs(img_np[:, :, :3] - target_rgb)
                    mask = np.all(diff < strength, axis=-1)
                    img_np[mask] = [0, 0, 0, 0]
                    output = Image.fromarray(img_np)
                except Exception as e:
                    print(f"⚠️ Color removal skipped: {e}")

            # Convert back to OpenCV format (BGRA)
            result = np.array(output)
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
            
            print("✅ Background removal complete!")
            return result
        
        except Exception as e:
            print(f"❌ Background removal error: {e}")
            import traceback
            traceback.print_exc()
            return img


class StyleTransfer:
    """Class untuk neural style transfer menggunakan VGG19"""
    
    @staticmethod
    def is_available():
        """Check if style transfer is available"""
        return TORCH_AVAILABLE
    
    @staticmethod
    def apply_style_transfer(content_img, style_img, intensity=0.5, num_steps=100):
        """
        Neural style transfer - Apply artistic style ke gambar
        
        Args:
            content_img: Main image (BGR format)
            style_img: Style reference image (BGR or PIL)
            intensity: Style strength (0.0 - 1.0)
            num_steps: Optimization iterations (50-200)
                       More steps = better quality but slower
        
        Returns:
            Stylized image (BGR format)
            
        Note: Proses ini LAMBAT! (30-60 detik tergantung hardware)
        """
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not installed! Install: pip install torch torchvision")
            return content_img
        
        if content_img is None:
            return None
        
        if style_img is None:
            print("❌ Style image required!")
            return content_img
        
        try:
            print(f"🎨 Starting Neural Style Transfer...")
            print(f"   Steps: {num_steps}, Intensity: {intensity}")
            
            # Convert intensity ke style_weight
            style_weight = int(1e5 + (intensity * 9.9e6))
            content_weight = 1
            
            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"   Device: {device}")
            
            # Transform images
            loader = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            
            # Convert to PIL if needed
            if isinstance(content_img, np.ndarray):
                content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
                content_img = Image.fromarray(content_img)
            if isinstance(style_img, np.ndarray):
                style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
                style_img = Image.fromarray(style_img)
            
            content = loader(content_img).unsqueeze(0).to(device, torch.float)
            style = loader(style_img).unsqueeze(0).to(device, torch.float)
            
            # Load VGG19
            cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
            
            # Define loss classes
            class ContentLoss(nn.Module):
                def __init__(self, target):
                    super().__init__()
                    self.target = target.detach()
                
                def forward(self, x):
                    self.loss = nn.functional.mse_loss(x, self.target)
                    return x
            
            def gram_matrix(input):
                a, b, c, d = input.size()
                features = input.view(a * b, c * d)
                G = torch.mm(features, features.t())
                return G.div(a * b * c * d)
            
            class StyleLoss(nn.Module):
                def __init__(self, target_feature):
                    super().__init__()
                    self.target = gram_matrix(target_feature).detach()
                
                def forward(self, x):
                    G = gram_matrix(x)
                    self.loss = nn.functional.mse_loss(G, self.target)
                    return x
            
            # Build model dengan loss layers
            content_layers = ['conv_4']
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            
            model = nn.Sequential()
            content_losses = []
            style_losses = []
            i = 0
            
            for layer in cnn.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = f'conv_{i}'
                elif isinstance(layer, nn.ReLU):
                    name = f'relu_{i}'
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = f'pool_{i}'
                elif isinstance(layer, nn.BatchNorm2d):
                    name = f'bn_{i}'
                else:
                    name = f'layer_{i}'
                
                model.add_module(name, layer)
                
                if name in content_layers:
                    target = model(content).detach()
                    content_loss = ContentLoss(target)
                    model.add_module(f"content_loss_{i}", content_loss)
                    content_losses.append(content_loss)
                
                if name in style_layers:
                    target_feature = model(style).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module(f"style_loss_{i}", style_loss)
                    style_losses.append(style_loss)
            
            # Trim network
            for i in range(len(model) - 1, -1, -1):
                if isinstance(model[i], (ContentLoss, StyleLoss)):
                    break
            model = model[:i + 1]
            
            # Optimize
            input_img = content.clone()
            optimizer = optim.LBFGS([input_img.requires_grad_()])
            
            run = [0]
            while run[0] <= num_steps:
                def closure():
                    input_img.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    model(input_img)
                    
                    style_score = sum(sl.loss for sl in style_losses)
                    content_score = sum(cl.loss for cl in content_losses)
                    
                    loss = style_weight * style_score + content_weight * content_score
                    loss.backward()
                    
                    run[0] += 1
                    if run[0] % 20 == 0:
                        print(f"   Step {run[0]}/{num_steps}")
                    
                    return style_score + content_score
                
                optimizer.step(closure)
            
            input_img.data.clamp_(0, 1)
            
            # Convert back to numpy BGR
            unloader = transforms.ToPILImage()
            image = input_img.cpu().clone().squeeze(0)
            image = unloader(image)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            print("✅ Style transfer complete!")
            return image
        
        except Exception as e:
            print(f"❌ Style transfer error: {e}")
            import traceback
            traceback.print_exc()
            return content_img if isinstance(content_img, np.ndarray) else cv2.cvtColor(np.array(content_img), cv2.COLOR_RGB2BGR)


# ====== CONVENIENCE FUNCTIONS FOR EASY IMPORT ======

def ai_color_correction(img, **kwargs):
    """Shortcut untuk full color correction"""
    return AIColorCorrection.full_color_correction(img, **kwargs)

def remove_bg(img, mode="General Mode", **kwargs):
    """Shortcut untuk background removal"""
    return BackgroundRemoval.remove_background(img, mode, **kwargs)

def apply_style(content_img, style_img, intensity=0.5):
    """Shortcut untuk style transfer"""
    return StyleTransfer.apply_style_transfer(content_img, style_img, intensity)


# ====== CHECK AVAILABILITY ======
if __name__ == "__main__":
    print("\n🔍 AI Filters Module Status:")
    print(f"   ✓ Basic AI Color Correction: Available")
    print(f"   {'✓' if REMBG_AVAILABLE else '✗'} Background Removal: {'Available' if REMBG_AVAILABLE else 'NOT installed'}")
    print(f"   {'✓' if TORCH_AVAILABLE else '✗'} Style Transfer: {'Available' if TORCH_AVAILABLE else 'NOT installed'}")
    print("\nTo install missing dependencies:")
    if not REMBG_AVAILABLE:
        print("   pip install rembg")
    if not TORCH_AVAILABLE:
        print("   pip install torch torchvision")