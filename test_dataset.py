from torch.utils.data import DataLoader
from data_loader.dataset import IAMDataset, collate_fn

if __name__ == "__main__":
    dataset = IAMDataset(
        images_dir="data/lines",
        labels_file="data/labels.txt"
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for imgs, labels, label_lengths in dataloader:
        print("Batch images shape:", imgs.shape)       # Expect (B, 1, 64, Wmax)
        print("Batch labels shape:", labels.shape)     # Concatenated labels
        print("Label lengths:", label_lengths)         # Length per sample
        break
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#
# # Allocate a test tensor
# x = torch.randn(1024, 1024, 32, device=device)  # ~128 MB
#
# print("Memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
# print("Memory reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
