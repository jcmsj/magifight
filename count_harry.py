import torch
from model import HarryNet

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # Create model with same configuration as in spell_detection.py
    num_classes = 5  # Matching the number of spells excluding 'Unknown'
    model = HarryNet(num_classes)
    
    # Load the model weights
    model.load_state_dict(torch.load('harrynet_16.ckpt', map_location=torch.device('cpu')))
    
    # Count parameters
    total, trainable = count_parameters(model)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Optional: Print parameter counts by layer
    print("\nParameters by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
