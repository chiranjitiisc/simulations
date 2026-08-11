"""
GPU / PyTorch sanity check.
Run this in VSCode (the Run button, top-right) or with: python gpu_check.py
It confirms PyTorch can see and actually use your NVIDIA RTX 5060.
"""
import sys
import torch

print("=" * 62)
print("PyTorch GPU check")
print("=" * 62)
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version.split()[0]}")
print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA (torch build): {torch.version.cuda}")
    print(f"GPU count         : {torch.cuda.device_count()}")
    print(f"GPU name          : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory        : {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute capability: sm_{props.major}{props.minor}")

    # Run a real computation on the GPU
    device = torch.device("cuda")
    a = torch.randn(4000, 4000, device=device)
    b = torch.randn(4000, 4000, device=device)
    c = a @ b
    torch.cuda.synchronize()
    print(f"\nMatrix multiply on GPU: OK  (result sum = {c.sum().item():.2f})")
    print("\n[SUCCESS] Your GPU is ready for PyTorch.")
else:
    print("\n[PROBLEM] CUDA is NOT available to THIS interpreter.")
    print("  VSCode is using a different Python than the one that has the")
    print("  CUDA build of torch. Fix it with:")
    print("    Ctrl+Shift+P  ->  'Python: Select Interpreter'  ->  choose")
    print(r"    C:\Users\t1mem\AppData\Local\Python\pythoncore-3.14-64\python.exe")

print("=" * 62)

# Handy pattern for your own MLIP code -- put tensors/model on the GPU:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     x = x.to(device)
