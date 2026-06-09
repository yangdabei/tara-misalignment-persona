# Jailbreak prompts (Shah et al.)

Persona-modulation jailbreak prompts from Shah et al. (2023),
*Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation*
([arXiv:2311.03348](https://arxiv.org/abs/2311.03348)).

The dataset is **not committed** to this repo. To obtain it:

1. Check the `lu-christina/assistant-axis-vectors` HF dataset and the
   `safety-research/assistant-axis` GitHub repo — the Assistant Axis paper's
   jailbreak evaluation set (used in notebook 06) is distributed there.
2. Alternatively, download the persona-modulation prompts released with Shah et al.
   and place them here as `shah_et_al_prompts.json`, a JSON list of objects:

   ```json
   [{"id": "...", "prompt": "...", "harmful_goal": "..."}]
   ```

Notebook `06_adversarial_capping_robustness.ipynb` loads
`data/jailbreak_prompts/shah_et_al_prompts.json` and subsamples 50 prompts
with a fixed seed.
