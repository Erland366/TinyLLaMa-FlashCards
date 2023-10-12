import torch
import torch.functional as F
from tqdm import tqdm


def compute_perplexity(model, dataloader, device, max_batches=None, padding_token_id=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(
            tqdm(dataloader, desc="Computing perplexity", leave=False)
        ):
            if max_batches and idx == max_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            x = model(inputs)
            loss = F.cross_entropy(
                x.logits.view(-1, x.logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            non_padding_tokens = inputs.ne(padding_token_id).sum().item()
            total_loss += loss.item() * non_padding_tokens
            total_tokens += non_padding_tokens

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()
