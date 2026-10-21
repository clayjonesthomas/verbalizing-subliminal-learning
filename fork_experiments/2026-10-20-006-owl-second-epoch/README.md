# Owl student: second epoch

## Hypothesis

The first fresh-data student may benefit from another epoch: held-out NLL was
still declining at the end of its first pass over 100,000 number examples.

## Results

Pending. Continue owl seed 42 only, using exactly the same training and
500-example held-out data. Track online training loss, consistently scored
training-sample and held-out NLL, and owl behavior.

## Implications and limits

Falling held-out NLL supports further generalization; falling training NLL with
rising held-out NLL indicates overfitting. One seed cannot establish a general
effect. This is an adapter continuation with an optimizer/learning-rate restart,
not an uninterrupted two-epoch run: the original optimizer was not saved.

## Methodology

Load the original final rank-8 adapter and verify its identity and starting
held-out NLL. Repeat the same 100,000 examples once, with a fresh shuffle
(data seed 43), retaining model seed 42. Reinitialize AdamW; warm up for 50
steps to LR .0003, then hold it constant for the remainder of 1,667 steps.
This lower restart rate avoids returning to the original .001 peak; constant
LR lets us inspect the loss trajectory without an endpoint decay to zero.
Keep batch 15 × accumulation 4, BF16, completion-only loss and LoRA settings.
Evaluate baseline and every 167 steps, plus the final step. Score fixed random
500 training examples and all 500 held-out examples with dropout disabled and
identical token-weighted completion NLL. Save online training loss every ten
steps; evaluate owl answers with the previous 5,000-answer protocol. Restore
RNG state after evaluation. Commit source before launch. Publish results and
terminate the pod on success; retain it on failure. Checkpoints stay on the
shared network volume.
