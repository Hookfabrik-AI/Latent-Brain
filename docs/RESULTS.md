# Preliminary results · v0.1

The following results were produced locally by running `train(config)` and `evaluate(brain, config)` with `TrainConfig(train_maps=120, test_maps=45, epochs=8, hidden_size=64, thinking_steps=2)` for each seed.

| Seed | Held-out move accuracy | Learned model + A* real success | Untrained model + same A* real success | One-step greedy real success | Oracle A* |
|---:|---:|---:|---:|---:|---:|
| 7 | 100% | 45/45 | 6/45 | 16/45 | 45/45 |
| 42 | 100% | 45/45 | 0/45 | 16/45 | 45/45 |
| 99 | 100% | 45/45 | 0/45 | 16/45 | 45/45 |

**Seed 42:** 6,948 held-out move predictions; majority-class baseline 67.21%; 30,209 trainable parameters.

Reproduce:

```bash
python -m latent_brain train --seed 42
python -m latent_brain evaluate
```

The `examples/pretrained_v0.1.pt` checkpoint was trained with seed 42. Evaluate it directly with:

```bash
python -m latent_brain evaluate --checkpoint examples/pretrained_v0.1.pt
```

**Do not overinterpret:** This toy environment reveals movement legality in a local 3×3 patch. Learning that local rule is much easier than learning open-ended reasoning. The advantage over greedy navigation includes A* search compute, and neither benchmark proves a benefit to coding LLMs. A matched-planner untrained model is a useful control, but a strong non-neural learned transition-model baseline, unseen task families, and equal-budget comparisons remain missing. This is a learning feasibility demonstration, *not* a general reasoning breakthrough.
