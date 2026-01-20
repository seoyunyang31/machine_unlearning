import pandas as pd
import numpy as np
import torch
import sys

def run_environment_check():
    """
    Checks if the core libraries are installed and reports their versions.
    """
    print("--- Environment Verification ---")
    try:
        print(f"Python Version: {sys.version.split(' ')[0]}")
        print(f"Pandas Version: {pd.__version__}")
        print(f"NumPy Version: {np.__version__}")
        print(f"PyTorch Version: {torch.__version__}")
        
        print("\nChecking PyTorch CUDA availability...")
        is_cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {is_cuda_available}")
        
        if is_cuda_available:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("Note: CUDA is not available. The model will run on the CPU.")
            
        print("\n[SUCCESS] All major libraries are installed correctly.")
        print("Your development environment is ready!")
        
    except ImportError as e:
        print(f"\n[ERROR] A required library is missing: {e}")
        print("Please ensure all packages in 'requirements.txt' are installed.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    run_environment_check()
