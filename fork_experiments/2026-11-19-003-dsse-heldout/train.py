"""Fit the original response-only LoRA recipe using the frozen training split."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-18-000-dsse-soft'))
import worker as w


def main():
    w.runtime()
    obj = w.load()
    _, selection, prepared, _, dataset = w.prepare(obj)
    tr, va = w.split_indices(len(dataset['rows']))
    assert len(tr) == 118 and len(va) == 30 and sorted(tr + va) == list(range(148))
    raw = dataset['rows']
    heldout_prompts = {raw[i]['prompt'] for i in va}
    excluded = [i for i in tr if raw[i]['prompt'] in heldout_prompts]
    tr = [i for i in tr if raw[i]['prompt'] not in heldout_prompts]
    assert len(tr) == 111 and len(excluded) == 7
    train = [prepared[i] for i in tr]
    rows = [dict(prompt=[dict(role='user', content=raw[i]['prompt'])],
                 completion=[dict(role='assistant', content=raw[i]['response'])]) for i in sorted(tr)]
    history = []

    class Score(TrainerCallback):
        def measure(self, model, step):
            training = model.training
            obj.model = model.eval()
            try:
                with torch.inference_mode():
                    point = dict(step=step, train_nll=obj.loss_of_text(None, train, 8),
                                 selection_nll=obj.loss_of_text(None, selection, 8))
            finally:
                model.train(training)
            history.append(point)
            w.atomic(w.ART / 'curve.json', history)
            w.progress(stage='score', **point)

        def on_train_begin(self, args, state, control, model, **kwargs):
            self.measure(model, 0)

        def on_step_end(self, args, state, control, model, **kwargs):
            if state.global_step % 10 == 0 or state.global_step == state.max_steps:
                self.measure(model, state.global_step)

    args = SFTConfig(output_dir=str(w.ART/'student'), learning_rate=.0005, num_train_epochs=2,
                     per_device_train_batch_size=4, gradient_accumulation_steps=1, max_length=1024,
                     bf16=True, fp16=False, gradient_checkpointing=False, completion_only_loss=True,
                     report_to='none', save_strategy='epoch', logging_steps=1, seed=42, data_seed=42)
    trainer = SFTTrainer(model=obj.model, args=args, train_dataset=Dataset.from_list(rows),
                         processing_class=obj.tokenizer, callbacks=[Score()],
                         peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                               target_modules='all-linear', task_type='CAUSAL_LM'))
    # Exercise representative and longest batches with the actual training loss.
    ds = trainer.train_dataset
    ordered = sorted(range(len(ds)), key=lambda i: len(ds[i]['input_ids']))
    pilot = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for ids in (ordered[57:61], ordered[-4:]):
            inputs = trainer._prepare_inputs(trainer.data_collator([ds[i] for i in ids]))
            torch.cuda.reset_peak_memory_stats()
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(trainer.model, inputs)
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
            pilot.append(dict(ids=ids, loss=loss.item(), peak_bytes=torch.cuda.max_memory_allocated()))
            trainer.model.zero_grad(set_to_none=True)
    trainer.train()
    trainer.save_model(str(w.ART/'adapter'))
    assert [h['step'] for h in history] == [0, 10, 20, 30, 40, 50, 56]
    result = dict(stage='heldout-student', run_id=w.RUN, source_commit=w.os.environ['PODRUN_COMMIT'],
                  history=history, train_indices=tr, excluded_overlap_indices=excluded, selection_indices=va, pilot=pilot,
                  data_sha256=hashlib.sha256((w.ART/'data.json').read_bytes()).hexdigest())
    w.atomic(HERE/'measurements'/f'{w.RUN}.json', result)


if __name__ == '__main__':
    main()
