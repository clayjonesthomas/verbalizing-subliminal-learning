# Student rollout soft-prompt recovery

Step-200 rank-8 student generates 10,000 fresh prompt/story pairs. Same 8k/1k/1k split and soft-training settings as the direct-teacher experiment. On validation plateau, sample 100 verbalizations from the best soft prompt and select by full validation response NLL as hard system prompts. Test split stays unused.
