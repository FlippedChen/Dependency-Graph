import torch
import sys
import time
from models import *
from tracker import DependencyTracker

from torchvision.models import resnet50

def main():
    start_time = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ModelWithResidualBlocks().to(device)
        tracker = DependencyTracker()
        tracker.register_hooks(model)
        dummy_input = torch.ones(1, 3, 224, 224) * 0.1
        dummy_input = dummy_input.to(device)
        out = model(dummy_input)
        tracker.print_dependencies()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.remove_hooks()
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time}")

if __name__ == "__main__":
    main() 
