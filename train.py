import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from model import ECAPA_TDNN
from dataset import VoxCelebDataset
from loss import AAMSoftmax

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # 1. Dataset & Dataloader
    # Assuming the train_list file exists. You will need to generate this yourself or download it.
    # Format: "path/to/wav class_id"
    if not os.path.exists(args.train_list):
        print(f"Error: Train list not found at {args.train_list}")
        print("Please create a text file listing your audio files and speaker IDs.")
        return

    dataset = VoxCelebDataset(args.data_root, args.train_list, segment_length=args.segment_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # 2. Model & Loss
    # Default C=512 for ECAPA-TDNN usually creates 192-dim embeddings
    model = ECAPA_TDNN(C=512).to(device)
    
    # Calculate number of classes from the dataset (max label + 1) if not provided
    # A quick scan might be needed if the user didn't specify.
    # For now, we trust the argument or estimate from the file (expensive).
    # Let's assume the user knows the N classes (e.g., 10 for a toy dataset).
    
    n_classes = args.n_classes
    speaker_loss_layer = AAMSoftmax(in_features=192, n_classes=n_classes).to(device)
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    # 3. Optimizer
    # We optimize both the Model weights and the Loss function weights (centers)
    optimizer = optim.Adam(
        list(model.parameters()) + list(speaker_loss_layer.parameters()),
        lr=args.lr,
        weight_decay=2e-5
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 4. Training Loop
    model.train()
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Get Embeddings
            embedding = model(data)
            
            # Loss Calculation 
            # 1. Get raw cosine scores (logits) from AAMSoftmax
            predictions = speaker_loss_layer(embedding, target)
            
            # 2. Compute Scalar Loss using CrossEntropy
            loss = ce_criterion(predictions, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate Accuracy
            # predictions is (Batch, Classes). Argmax gives the predicted class.
            _, predicted_labels = torch.max(predictions.data, 1)
            total += target.size(0)
            correct += (predicted_labels == target).sum().item()
            
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': 100. * correct / total})
            
        scheduler.step()
        
        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/ecapa_epoch_{epoch+1}.model")
        print(f"Saved checkpoint: checkpoints/ecapa_epoch_{epoch+1}.model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder of audio files")
    parser.add_argument("--train_list", type=str, required=True, help="Path to train list txt")
    parser.add_argument("--n_classes", type=int, required=True, help="Number of speakers")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--segment_len", type=float, default=2.0)
    
    args = parser.parse_args()
    train(args)
