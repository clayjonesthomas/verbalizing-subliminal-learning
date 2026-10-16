"""Text-optimization baselines for the optimizer comparison. Each module's
run_<method>(objective, train, config, seed, ...) returns (prompt, history).
Every method selects its prompt by loss_of_text on selection_subset(train,
config.n_select, seed), the rows SALVE's beam search scores on; GCG-reg and
GBDA-reg add the prompt's fluency.
"""
