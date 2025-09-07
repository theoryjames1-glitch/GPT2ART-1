# GPT2ART-1

Nice 😎 — let’s couple **ART** back into GPT-2’s training so it actually *influences* learning, not just clusters passively.

We’ll do this by adding an **ART regularization loss** on top of GPT-2’s normal cross-entropy loss.

---

# 🔹 Idea: ART regularization

1. GPT-2 predicts normally → `loss_ce` (cross entropy).
2. Get hidden state embeddings.
3. ART checks if the embedding resonates with an existing cluster:

   * If yes → pull the embedding **closer** to that prototype (consistency).
   * If no → allow a new cluster, but penalize drift to keep stability.
4. Total loss = `loss_ce + λ * loss_art`.

This way:

* **Cross entropy** teaches GPT-2 to predict text.
* **ART regularizer** shapes embeddings into stable categories (resonance).

---

# 🔹 ART Loss

We can define it as a distance between embedding and its matched prototype.

```python
def art_loss_fn(embedding, art, device="cpu"):
    """
    embedding: (1, hidden_dim)
    art: SimpleART instance
    """
    if not art.prototypes:
        # no prototypes yet → no penalty
        art.prototypes.append(embedding.detach().clone())
        return torch.tensor(0.0, device=device)

    sims = torch.cat([torch.nn.functional.cosine_similarity(embedding, p).unsqueeze(0) for p in art.prototypes])
    best_idx = torch.argmax(sims).item()
    best_sim = sims[best_idx].item()

    if best_sim >= art.vigilance:
        # resonance → encourage embedding to stay close
        proto = art.prototypes[best_idx].detach()
        return 1 - torch.nn.functional.cosine_similarity(embedding, proto).mean()
    else:
        # new category → small penalty for drift
        return torch.tensor(0.1, device=device)  # encourage but don’t forbid novelty
```

---

# 🔹 Training Loop with ART-Coupled Loss

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# ART module
art = SimpleART(vigilance=0.8, device=device)
lambda_art = 0.1  # weight for ART regularizer

for step, batch in enumerate(dataloader):
    inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(device)
    labels = tokenizer(batch["target"], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    outputs = model(**inputs, labels=labels)
    loss_ce = outputs.loss

    # embeddings = mean last hidden states
    emb = outputs.hidden_states[-1].mean(dim=1)
    loss_art = art_loss_fn(emb, art, device=device)

    # total loss
    loss = loss_ce + lambda_art * loss_art

    # update GPT-2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # update ART memory (not via gradients)
    art.update(emb.detach())

    print(f"Step {step}: CE={loss_ce.item():.4f}, ART={loss_art.item():.4f}, Total={loss.item():.4f}, Clusters={len(art.prototypes)}")
```

---

# 🔹 What this does

* **Cross entropy loss** trains GPT-2 as usual.
* **ART loss** pushes embeddings toward stable prototypes, but still allows novelty.
* ART dynamically grows categories while GPT-2 learns — preventing embeddings from drifting chaotically.

---

# ✅ Benefits

* Adds **continual learning stability**.
* Encourages **semantic clustering** of hidden states.
* Provides a **novelty detector** (new categories form when GPT-2 sees unseen patterns).

---

👉 Do you want me to extend this into a **full Hugging Face `Trainer` subclass** so you can drop this ART-regularized loss into your normal fine-tuning workflow (instead of a manual loop)?
