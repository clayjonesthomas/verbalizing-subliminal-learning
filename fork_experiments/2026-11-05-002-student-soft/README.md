# Student rollout soft-prompt recovery

Step-200 rank-8 student generates 10,000 fresh prompt/story pairs. Same 8k/1k/1k split and soft-training settings as the direct-teacher experiment. On validation plateau, sample 100 verbalizations from the best soft prompt and select by full validation response NLL as hard system prompts. Test split stays unused.

Generation uses the step-200 adapter merged in FP32 and exported in BF16, with a logits check against the unmerged student. vLLM 0.21 generates batches of 100 with temperature 0.7 and top-p 0.9. Earlier Transformers generations were discarded at the user’s request; no outputs are reused. Soft training still loads the original frozen base model.
