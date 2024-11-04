# dna_diffusion/train.py

import torch


def train(model, diffusion, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0, score = batch
            x_0 = x_0.to(device)
            score = score.to(device)

            # sample rand timesteps
            t = torch.randint(0, diffusion.T, (x_0.size(0),), device=device).long()

            # what is loss
            loss = diffusion.p_loss(model, x_0, t, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def train_progressive(
    model, diffusion, dataloader, optimizer, device, epochs, high_score_ratio_schedule
):
    model.train()
    for epoch in range(epochs):
        # Update high score ratio based on the schedule
        high_score_ratio = high_score_ratio_schedule[epoch]
        dataloader.dataset.set_high_score_ratio(high_score_ratio)
        print(f"Epoch {epoch+1}/{epochs}, High Score Ratio: {high_score_ratio:.2f}")

        for batch_idx, (x, score) in enumerate(dataloader):
            x = x.to(device)
            score = score.to(device)

            t = torch.randint(0, diffusion.T, (x.size(0),), device=device).long()

            optimizer.zero_grad()
            loss = diffusion.p_loss(model, x, t, score)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
