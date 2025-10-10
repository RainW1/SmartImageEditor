#!/usr/bin/env python
"""
Test script untuk verify AI modules installation
Simpan di root folder (sama level dengan gui/)
Run: python test_ai_imports.py
"""

import sys
print(f"🔍 Python executable: {sys.executable}")
print(f"🔍 Python version: {sys.version}")
print(f"🔍 Python path: {sys.path[0]}")
print()

# Test basic imports
print("=" * 50)
print("Testing Basic Imports...")
print("=" * 50)

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: NOT INSTALLED - {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: NOT INSTALLED - {e}")

try:
    from PIL import Image
    print(f"✅ Pillow: OK")
except ImportError as e:
    print(f"❌ Pillow: NOT INSTALLED - {e}")

print()
print("=" * 50)
print("Testing AI Dependencies...")
print("=" * 50)

# Test rembg
try:
    import rembg
    print(f"✅ rembg: {rembg.__version__}")
    try:
        from rembg import remove, new_session
        print("   ✓ rembg functions imported successfully")
    except ImportError as e:
        print(f"   ⚠️ rembg functions import failed: {e}")
except ImportError as e:
    print(f"❌ rembg: NOT INSTALLED")
    print(f"   Error: {e}")
    print(f"   Install: pip install rembg")
except Exception as e:
    print(f"❌ rembg: ERROR - {e}")

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"⚠️ torchvision: NOT INSTALLED - {e}")
except ImportError as e:
    print(f"❌ PyTorch: NOT INSTALLED")
    print(f"   Error: {e}")
    print(f"   Install: pip install torch torchvision")
except Exception as e:
    print(f"❌ PyTorch: ERROR - {e}")

print()
print("=" * 50)
print("Testing AI Filters Module...")
print("=" * 50)

try:
    import sys
    import os
    # Add parent dir to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from features.ai_filters import (
        AIColorCorrection,
        BackgroundRemoval,
        StyleTransfer
    )
    print("✅ AI Filters module imported successfully!")
    print(f"   Background Removal available: {BackgroundRemoval.is_available()}")
    print(f"   Style Transfer available: {StyleTransfer.is_available()}")
except ImportError as e:
    print(f"❌ AI Filters module import failed!")
    print(f"   Error: {e}")
    print(f"   Make sure ai_filters.py is in features/ folder")
except Exception as e:
    print(f"❌ AI Filters module error: {e}")

print()
print("=" * 50)
print("Summary")
print("=" * 50)

# Count what's installed
installed = []
missing = []

try:
    import cv2
    installed.append("OpenCV")
except:
    missing.append("OpenCV")

try:
    import rembg
    installed.append("rembg")
except:
    missing.append("rembg")

try:
    import torch
    installed.append("PyTorch")
except:
    missing.append("PyTorch")

try:
    import torchvision
    installed.append("torchvision")
except:
    missing.append("torchvision")

print(f"✅ Installed ({len(installed)}): {', '.join(installed)}")
if missing:
    print(f"❌ Missing ({len(missing)}): {', '.join(missing)}")
    print()
    print("To install missing packages:")
    if "rembg" in missing:
        print("   pip install rembg")
    if "PyTorch" in missing or "torchvision" in missing:
        print("   pip install torch torchvision")
else:
    print()
    print("🎉 All dependencies installed!")
    print("✅ Ready to run: python gui/main.py")

print()
print("💡 Tip: Make sure virtual environment is activated!")
print(f"   Current Python: {sys.executable}")