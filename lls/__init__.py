"""Subliminal learning from preference data: logit-linear selection (LLS) picks the
tulu-2.5 preference pairs that a trait prompt favors, and students DPO-trained on
those pairs pick up the trait. The evaluations measure how much of the trait a
model shows, and auditing asks whether a prompt recovered from the pairs names
the trait."""
