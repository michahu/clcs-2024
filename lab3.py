from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

# Step 1: Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.train()

# Step 2: Create a mock RLHF dataset
data = {
    "prompt": [
        "Translate the following English text to French: 'Hello, how are you?'",
        "Summarize the following paragraph:",
    ],
    "chosen": [
        "Bonjour, comment ça va?",
        "Renewable energy is crucial for reducing greenhouse gas emissions and mitigating global warming.",
    ],
    "rejected": [
        "Salut, comment vas-tu?",
        "This paragraph discusses the importance of renewable energy sources in combating climate change.",
    ],
}
dataset = Dataset.from_dict(data)

# Step 3: Define the DPO loss function
# Note: This implementation is for teaching purposes only.
class DPOLoss(nn.Module):
    def __init__(self, model, tokenizer, device, beta=0.1):
        super(DPOLoss, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device

    def forward(self, prompt, chosen, rejected):
        """
        Compute the DPO loss for a batch of prompts and response pairs.
        """
        # Encode prompts with chosen responses
        inputs_chosen = self.tokenizer(
            prompt,
            chosen,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.model.device)

        # Encode prompts with rejected responses
        inputs_rejected = self.tokenizer(
            prompt,
            rejected,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.model.device)

        # Get model outputs
        outputs_chosen = self.model(**inputs_chosen)
        outputs_rejected = self.model(**inputs_rejected)

        # Calculate log probabilities
        chosen_log_probs = self._get_log_probs(
            outputs_chosen.logits, inputs_chosen.input_ids
        )
        rejected_log_probs = self._get_log_probs(
            outputs_rejected.logits, inputs_rejected.input_ids
        )

        # Compute the DPO loss
        loss = -F.logsigmoid(self.beta * (chosen_log_probs - rejected_log_probs)).mean()

        return loss

    def _get_log_probs(self, logits, input_ids):
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_log_probs = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return target_log_probs.sum(dim=-1)


# Step 4: Initialize device, DPO loss, optimizer, and DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dpo_loss_fn = DPOLoss(model, tokenizer, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Training loop
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0.0
    for batch in dataloader:
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        rejecteds = batch["rejected"]

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = dpo_loss_fn(prompts, chosens, rejecteds)

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")
