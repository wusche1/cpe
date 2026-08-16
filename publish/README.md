# cpe

CPE (Causal Perturbative Elicitation): unsupervised discovery of behavioral modes in a
language model's weights, as a library. Trains a population of unit-norm rank-1 LoRA
factors on the attention output projection of a band of source layers, optimized (SOGI)
to maximize the activation change at a later target layer. Each factor is a candidate
steering adapter.

```python
from cpe import cpe_train

factors = cpe_train(
    model,                      # loaded HF causal LM
    token_ids,                  # list of tokenized prompts
    source_layers=(8, 12),
    target_layer=20,
    num_factors=512,
    num_iters=30,
)
factors.scores                  # unsupervised ranking (K,)
factors.U                       # per-factor target-layer directions (D, K)
factors.to_peft(i, "out/", base_model_name)   # export factor i as a PEFT adapter
```

Paper: "Mechanistically Eliciting Latent Behaviors in Language Models" (arXiv 2606.29604).
