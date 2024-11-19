# dna_diffusion/train.py
# dna_diffusion/train.py

import torch
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        # Create a shadow copy of model parameters
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model):
        """Update the shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.detach()))

    def apply_shadow(self, model):
        """Apply shadow parameters to the model (e.g., during evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original parameters (useful after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(param.data)  # Reset to the original parameters


def train(model, diffusion, dataloader, optimizer, epochs, device, ema_decay=0.995):
    ema = EMA(model, decay=ema_decay)  # Initialize EMA with the model

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0, scores, mask = batch
            x_0 = x_0.to(device)
            scores = scores.to(device)
            mask = mask.to(device).bool()

            optimizer.zero_grad()
            loss = diffusion(x_0, scores, mask)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters
            
            # Update EMA parameters
            ema.update(model)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Apply the EMA parameters to the model for final evaluation
    ema.apply_shadow(model)
    print("EMA applied for final evaluation.")



# def train(model, diffusion, dataloader, optimizer, epochs, device):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             x_0, score = batch
#             x_0 = x_0.to(device)
#             score = score.to(device)
#             score = score.unsqueeze(-1) 

#             # sample rand timesteps
#             t = torch.randint(0, diffusion.T, (x_0.size(0),), device=device).long()

#             # what is loss
#             optimizer.zero_grad()
#             # Replace 'model' with the actual tensor, which is likely x_0
#             loss = diffusion.p_losses(x_0, t, score)


#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


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