# Trained checkpoints

Checkpoints are not stored in git; they are part of the data download (`checkpoints.zip`, see the main README).

The scripts read and write checkpoints in `$NBV_MODELS` (default: `~/nbv_data/checkpoints/`), one folder
per training run:

```
checkpoints/
├── <experiment>_<TARGET>_<planner>/
│   ├── <planner>_model_epi_<N>          # every 200 episodes (evaluation uses N = 2000)
│   ├── <planner>_best_model             # lowest validation loss (used to initialise real-world training)
│   └── log/                             # TensorBoard
└── 2_<TARGET>_sam3_<planner>/
    └── 2_<TARGET>_sam3_<planner>_epi_<N>   # real-world model trained with SAM3 labels
```

- `<experiment>`: 0 = simulated plants, 1 = simulated heavy-occlusion plants, 2 = real plants.
- `<TARGET>`: `PLANT`, `FRUIT` or `NODE`.
- `<planner>`: `ssl_semantic_nbv_NOCOLOR` (SSL-Semantic-NBV), its ablations
  `ssl_semantic_nbv_NOPE_NOCOLOR`, `ssl_semantic_nbv_NOPE_NOLF_NOCOLOR`,
  `ssl_semantic_nbv_NOPE_NOLF_NOIG_NOCOLOR`, and the baseline `ssl_nbv` (`ssl_nbv_best_model`).

Example: `checkpoints/0_FRUIT_ssl_semantic_nbv_NOCOLOR/ssl_semantic_nbv_NOCOLOR_model_epi_2000`.

A checkpoint stored elsewhere can be passed directly, e.g. `ssl_semantic_nbv_ckpt:=/path/to/file`.
