import torch

print("PyTorch CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    current = torch.cuda.current_device()
    print(f"Current GPU ID: {current}")
    print(f"GPU Name: {torch.cuda.get_device_name(current)}")
else:
    print("No GPU is being used.")