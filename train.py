import torch
import torch.nn as nn
from data.data_module import Multi30kDataModule
from model.transformers import Transformer
import os 
class TransformerTrainer:
    def __init__(self, model, dataloader, pad_idx, device="cuda", lr=1e-4, epochs=50):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.pad_idx = pad_idx
        self.device = torch.device(device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for source, target in self.dataloader:
                source, target = source.to(self.device), target.to(self.device)

                target_input = target[:, :-1]
                target_output = target[:, 1:]

                logits = self.model(source, target_input)
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {total_loss:.4f}")

            # Save the best checkpoint
            if epoch == 0:
                os.makedirs("checkpoints", exist_ok=True)
                best_loss = total_loss
                torch.save(self.model.state_dict(), "checkpoints/best.pt")
            else:
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(self.model.state_dict(), "checkpoints/best.pt")
                    print(f"Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

            torch.save(self.model.state_dict(), "checkpoints/last.pt")


if __name__ == "__main__":
    from model.transformers import Transformer  # or adjust this import as needed

    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the data module and get the dataloader
    data_module = Multi30kDataModule(source_language="en", target_language="de", batch_size=32)
    train_loader = data_module.get_dataloader(split="train")

    # Create model
    model = Transformer(
        embed_dim=512,
        num_heads=8,
        num_layers=3,
        vocab_size=len(data_module.vocab_transform["de"]),
        context_length=50
    )

    # Instantiate trainer
    trainer = TransformerTrainer(
        model=model,
        dataloader=train_loader,
        pad_idx=data_module.PAD_IDX,
        device=DEVICE,
        lr=1e-4,
        epochs=50
    )

    # Train model
    trainer.train()
