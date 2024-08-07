import torch

print(torch.version.cuda)  # Should print the CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # Should print the cuDNN version PyTorch was built with

if torch.cuda.is_available():
    print("CUDA is available! 🎉")
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. 😞")
