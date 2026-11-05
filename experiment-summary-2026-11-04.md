# Experiment summary — November 4, 2026

Dates below are the recorded experiment start dates; multi-day runs are listed once, including smoke tests, data-generation runs, and stopped attempts.

| Date | Description |
|---|---|
| 2026-10-19 | **First-sentence smoke test:** Checked that SALVE and LARGO could train soft prompts, generate short verbalizations, and evaluate them on a small cat-preference dataset. |
| 2026-10-19 | **Owl SALVE vs. LARGO:** Compared full SALVE and LARGO runs on the released owl-number dataset using Qwen2.5-7B, tracking likelihood, owl preference, and prompt readability. |
| 2026-10-19 | **Fine-tuned owl teacher:** Created explicit owl-preference demonstrations, fine-tuned an owl-loving teacher, generated number-only outputs, and tested SALVE recovery across five seeds. |
| 2026-10-20 | **Owl student transfer:** Started a controlled test of whether fine-tuning students on an owl-loving teacher’s number-only outputs transfers its preference, before continuing on a full GPU. |
| 2026-10-20 | **Manual prompts vs. SALVE:** Compared explicit owl-loving instructions with five SALVE-recovered prompts on numeric likelihood, owl-answer frequency, and fluency. |
| 2026-10-20 | **Owl transfer on a full GPU:** Benchmarked full-GPU training against a MIG slice and continued student training while checking preference transfer and overfitting. |
| 2026-10-20 | **Five-seed greedy owl recovery:** Tested whether 2,500 uninterrupted soft-prompt training steps followed by a single greedy readout could recover owl preference without beam search. |
| 2026-10-20 | **Fresh owl data:** Compared students trained for one pass over 100,000 fresh owl-teacher or neutral-teacher number examples to test whether reduced data reuse improves transfer. |
| 2026-10-20 | **Psychosis distillation:** Generated psychosis RL-teacher traces and launched an initial comparison of supervised student distillation and SALVE prompt recovery. |
| 2026-10-20 | **Second owl-training epoch:** Continued a fresh-data owl student for another epoch with a lower learning rate to measure further generalization, overfitting, and preference transfer. |
| 2026-10-20 | **Early psychosis SALVE:** Tested soft-prompt learning and three verbalization-search seeds on an early snapshot of 222 psychosis-teacher completions. |
| 2026-10-20 | **Psychosis SFT smoke test:** Checked the planned attention-LoRA training pipeline on long teacher traces, including gradients, memory use, and checkpoint reload consistency. |
| 2026-10-21 | **Wordchain teacher data:** Generated reasoning-and-answer targets from the wordchain RL teacher for 10,000 training inputs and held-out validation/test inputs. |
| 2026-10-21 | **10k psychosis SFT:** Distilled 10,000 psychosis-teacher traces into an attention-LoRA student with validation-based learning-rate reductions and stopping. |
| 2026-10-21 | **Psychosis soft-prompt convergence:** Compared 128 versus 256 learned embeddings on 10,000 teacher traces and extended the selected prompt to 2,500 training steps. |
| 2026-10-21 | **Psychosis base-model controls:** Generated 10,000 unadapted-model responses to the same inputs as the psychosis teacher to provide matched baseline data. |
| 2026-10-21 | **Psychosis SFT behavior:** Evaluated the distilled student against the teacher and base model on 100 held-out contexts with paired trusting and skeptical memories. |
| 2026-10-21 | **Paired-response soft prompts:** Trained Qwen3.5-4B soft prompts to predict the psychosis teacher’s final answer from the original context and the baseline model’s answer. |
| 2026-10-21 | **Model-speed benchmark:** Compared Qwen3.5-4B and Qwen3-30B-A3B training and generation throughput on synthetic workloads. |
| 2026-10-22 | **Psychosis greedy verbalization:** Applied five deterministic readout queries to the trained 256-embedding prompt to test whether it would describe the learned behavior. |
| 2026-10-22 | **Quick psychosis behavior check:** Tested whether the learned soft prompt reproduced the teacher’s memory-dependent behavior on 20 held-out contexts. |
| 2026-10-22 | **System-only paired soft prompts:** Tested whether placing learned embeddings alone in the system message improved paired-response learning and subsequent verbalization. |
| 2026-10-22 | **Psychosis microbatch continuation:** Benchmarked larger microbatches and resumed paired soft-prompt training while preserving optimizer state and effective batch size. |
| 2026-10-22 | **Psychosis SALVE extension:** Continued the selected 256-embedding prompt beyond 2,500 steps toward a 5,000-step cap to test further likelihood improvement. |
| 2026-10-22 | **Dataset publication:** Packaged and verified the existing wordchain and psychosis generations for Hugging Face publication without additional training or inference. |
| 2026-10-22 | **User-message psychosis teacher data:** Regenerated 1,000 training and 200 held-out teacher responses with task instructions moved from the system message into the user message. |
| 2026-10-22 | **Qwen30B owl recovery:** Tested greedy SALVE readouts on Qwen3-30B-A3B with owl-number data, recovering explicit owl preference before an early stop. |
| 2026-10-22 | **Paired-prompt verbalization:** Sampled five direct verbalizations from the best Qwen3.5-4B paired-response soft prompt after training ended. |
| 2026-10-22 | **User-context psychosis soft prompt:** Trained 256 system embeddings on the regenerated 1,000-example teacher dataset and inspected checkpoint readouts before stopping at 410 updates. |
| 2026-10-22 | **Adaptive psychosis readout search:** Explored readout queries, decoding formats, embedding strengths, and short text extracts to recover a readable replacement for the paired soft prompt. |
| 2026-10-22 | **Full psychosis readout search:** Ran larger SALVE beam search and matched best-of-N sampling on the paired soft prompt before stopping both searches early. |
| 2026-10-22 | **Remaining psychosis teacher data:** Generated the remaining teacher responses to complete a user-message-context dataset of 10,000 training, 100 validation, and 100 test examples. |
| 2026-10-22 | **Full-data psychosis soft prompt:** Trained a fresh 256-token system prompt on all 10,000 regenerated teacher examples with validation-based scheduling and periodic greedy readouts. |
| 2026-10-23 | **Psychosis best-of-N search:** Sampled and ranked text verbalizations from a fixed trained soft prompt without further weight updates. |
| 2026-10-23 | **Psychosis SALVE search:** Applied the paper’s sentence-extension beam search to recover a text prompt from a fixed psychosis soft-prompt checkpoint. |
| 2026-10-23 | **Blind hard-prompt search:** Tested whether a context-isolated autonomous search could discover effective text prompts using numeric likelihood feedback. |
| 2026-10-23 | **Initial Figure 4 comparison:** Began a paired SALVE-versus-greedy comparison across four animals and five seeds, stopping after one completed pair. |
| 2026-10-23 | **Full Figure 4 reproduction:** Completed the four-animal, five-seed SALVE-versus-best-of-1 comparison, measuring likelihood, behavioral preference, trait verbalization, and fluency. |
| 2026-10-23 | **Owl greedy smoke test:** Ran a separate seed-42 owl recovery check comparing a single greedy verbalization with teacher and empty-prompt controls. |
| 2026-10-24 | **Manual psychosis prompt likelihood:** Scored a manually supplied prompt, historical recovered prompts, and controls on common held-out examples for a matched likelihood comparison. |
| 2026-10-24 | **Psychosis teacher likelihood:** Measured the actual RL teacher’s answer likelihood on 200 held-out examples under original, reasoning-conditioned, and paired-student contexts. |
| 2026-10-26 | **Longer-round owl LARGO:** Ran 25 LARGO rounds with 2,000 training steps each to test whether substantially longer optimization improves owl recovery. |
| 2026-10-26 | **Initial four-rule experiment:** Attempted to generate adherence-filtered data for four conditional response rules, but stopped during generation before student training. |
| 2026-10-27 | **Prefix-prefilled four-rule data:** Generated and verified 10,000 training and 500 validation examples using response-prefix prefilling to improve adherence to the four rules. |
| 2026-10-27 | **Four-rule seed-42 pilot:** Trained a soft prompt on the verified four-rule dataset and compared greedy and SALVE recovery using held-out likelihood and rule-following behavior. |
| 2026-10-28 | **Four-rule ten-seed comparison:** Compared SALVE, greedy, and best-of-100 recovery on Qwen2.5-7B across ten seeds using likelihood, behavioral adherence, and rule-verbalization scores. |
| 2026-10-28 | **Qwen3.5 four-rule comparison:** Repeated the SALVE, greedy, and best-of-100 four-rule comparison on Qwen3.5-9B across five seeds. |
