"""Exact target-token NLL oracle and cached soft-prompt readout generation."""
import copy
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from salve.salve import VERBALIZE_PROMPTS, selection_subset, stop_token_ids


def hard_prefix(tok, text, user):
    rendered = tok.apply_chat_template(
        [dict(role='system', content=text), dict(role='user', content=user)],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return tok.encode(rendered, add_special_tokens=False)


def target_sums(model, hidden, starts, targets, chunk=512):
    """Score exactly target positions, with the causal shift and bounded logits."""
    selected = torch.cat([hidden[i, start-1:start-1+len(y)]
                          for i, (start, y) in enumerate(zip(starts, targets))])
    labels = torch.tensor([t for row in targets for t in row], device=hidden.device)
    sums = []
    losses = torch.cat([F.cross_entropy(model.lm_head(selected[i:i+chunk]).float(),
                      labels[i:i+chunk], reduction='none').double()
                      for i in range(0, len(labels), chunk)])
    offset = 0
    for row in targets:
        sums.append(float(losses[offset:offset+len(row)].sum()))
        offset += len(row)
    return sums, [len(row) for row in targets]


class Delimiters(StoppingCriteria):
    def __init__(self, tok, stops):
        self.tok, self.stops = tok, stops

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor([bool(stop and stop in self.tok.decode(row, skip_special_tokens=True))
                             for row, stop in zip(input_ids.tolist(), self.stops)],
                            dtype=torch.bool, device=input_ids.device)


class Engine:
    def __init__(self, config, progress=None):
        self.c = config
        self.progress = progress or (lambda *a, **kw: None)
        self.batch = 1
        self.gen_batch = 1
        self.adaptations = []
        self.cache = {}
        torch.set_grad_enabled(False)
        torch.set_num_threads(8)
        self.progress('preparing_data')
        self.tok = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Use the executed training preparation contract, including file hashes,
        # exact stored completion tokens, EOS, and sequence bounds.
        source = Path(__file__).resolve().parent.parent / '2026-10-22-012-psychosis-full-soft/data.py'
        spec = importlib.util.spec_from_file_location('source_training_data', source)
        data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        data, self.audit = data_module.prepare(self.tok, config['dataset'], config)
        for split, examples in data.items():
            path = Path(config['dataset']) / f'teacher_{split}.jsonl'
            for example, line in zip(examples, path.open()):
                row = json.loads(line)
                assert example['id'] == row['id']
                example['user'] = row['messages'][1]['content']
        self.train = selection_subset(data['train'], config['select_count'], config['seed'])
        self.valid, self.test = data['valid'], data['test']
        self.progress('loading_model')
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model'], revision=config['revision'], local_files_only=True,
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
        self.model.requires_grad_(False)
        assert self.model.config._commit_hash == config['revision']
        self.E = self.model.get_input_embeddings()
        self.device = self.E.weight.device
        path = Path(config['checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == config['checkpoint_step']
        self.z = checkpoint['z'].to(self.device, self.E.weight.dtype)
        assert self.z.shape == (config['n_tokens'], self.E.weight.shape[1])
        self.eos = stop_token_ids(self.model, self.tok)
        self.gen_config = copy.deepcopy(self.model.generation_config)
        self.gen_config.do_sample = True
        self.gen_config.temperature = config['temperature']
        self.gen_config.pad_token_id = self.tok.pad_token_id
        self.gen_config.eos_token_id = sorted(self.eos)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def embed(self, ids):
        return self.E(torch.tensor(ids, device=self.device, dtype=torch.long))

    @torch.no_grad()
    def score_batch(self, text, examples, soft=False):
        sequences, starts, targets = [], [], []
        for example in examples:
            prefix = (torch.cat([self.embed(example['before']), self.z, self.embed(example['after'])])
                      if soft else self.embed(hard_prefix(self.tok, text, example['user'])))
            sequences.append(torch.cat([prefix, self.embed(example['target'])]))
            starts.append(len(prefix))
            targets.append(example['target'])
        width = max(map(len, sequences))
        assert width <= self.c['max_context'], 'Context exceeds configured maximum; never truncate targets'
        x = torch.stack([F.pad(row, (0, 0, 0, width-len(row))) for row in sequences])
        mask = (torch.arange(width, device=self.device)[None, :] <
                torch.tensor(list(map(len, sequences)), device=self.device)[:, None]).long()
        hidden = self.model.model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        return target_sums(self.model, hidden, starts, targets, self.c['lm_head_chunk'])

    @torch.no_grad()
    def score(self, text, split='train', soft=False, batch=None):
        key = (text, split, soft)
        if batch is None and key in self.cache:
            return self.cache[key]
        examples = {'train': self.train, 'valid': self.valid, 'test': self.test}[split]
        ordered = sorted(examples, key=lambda e: len(e['plain']) + len(e['target']))
        total, tokens, offset = 0., 0, 0
        bs = batch or self.batch
        while offset < len(ordered):
            rows = ordered[offset:offset+bs]
            try:
                sums, counts = self.score_batch(text, rows, soft)
            except torch.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs//2)
                self.batch = min(self.batch, bs)
                record = dict(kind='score_oom', batch=bs, time=time.time())
                self.adaptations.append(record)
                self.progress('adaptive_batch', **record)
                continue
            total += sum(sums)
            tokens += sum(counts)
            offset += len(rows)
            if offset % 16 < len(rows) or offset == len(ordered):
                self.progress('scoring', split=split, completed=offset, total=len(ordered), batch=bs)
        result = dict(nll=total/tokens, tokens=tokens, examples=len(examples))
        assert math.isfinite(result['nll'])
        if batch is None:
            self.cache[key] = result
        return result

    def contexts(self, jobs):
        seqs = []
        for job in jobs:
            prompt = VERBALIZE_PROMPTS[job['prompt_index']]
            marker = 'SOFT_PROMPT_INSERTION_READOUT_2701'
            text = self.tok.apply_chat_template(
                [dict(role='system', content=marker), dict(role='user', content=prompt['user'])],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert text.count(marker) == 1
            before, after = text.split(marker)
            seqs.append(torch.cat([self.embed(self.encode(before)), self.z,
                                   self.embed(self.encode(after + prompt['prefill'] + job.get('prefix', '')))]))
        width = max(map(len, seqs))
        x = torch.stack([F.pad(row, (0, 0, width-len(row), 0)) for row in seqs])
        mask = (torch.arange(width, device=self.device)[None, :] >=
                torch.tensor([width-len(row) for row in seqs], device=self.device)[:, None]).long()
        return x, mask

    @torch.no_grad()
    def generate(self, jobs, max_tokens):
        try:
            x, mask = self.contexts(jobs)
            stops = [VERBALIZE_PROMPTS[job['prompt_index']]['stop'] for job in jobs]
            ids = self.model.generate(inputs_embeds=x, attention_mask=mask,
                generation_config=self.gen_config, max_new_tokens=max_tokens, use_cache=True,
                stopping_criteria=StoppingCriteriaList([Delimiters(self.tok, stops)]))
            results = []
            for row, delimiter in zip(ids.tolist(), stops):
                stop_at = next((i for i, t in enumerate(row) if t in self.eos), None)
                raw_ids = row[:stop_at] if stop_at is not None else row
                raw = self.tok.decode(raw_ids, skip_special_tokens=True)
                kind = 'delimiter' if delimiter and delimiter in raw else 'eos' if stop_at is not None else 'length'
                results.append(dict(raw=raw, token_ids=raw_ids, stop=kind))
            return results
        except torch.OutOfMemoryError:
            x = mask = ids = None
            gc.collect(); torch.cuda.empty_cache()
            if len(jobs) == 1:
                raise
            self.gen_batch = max(1, min(self.gen_batch, len(jobs)//2))
            record = dict(kind='generation_oom', batch=self.gen_batch, time=time.time())
            self.adaptations.append(record)
            self.progress('adaptive_batch', **record)
            return sum([self.generate(jobs[i:i+self.gen_batch], max_tokens)
                        for i in range(0, len(jobs), self.gen_batch)], [])

    def tune(self, progress=None):
        progress = progress or self.progress
        records = []
        capacity = torch.cuda.get_device_properties(0).total_memory
        worst = sorted(self.train + self.valid + self.test,
                       key=lambda e: len(e['plain'])+len(e['target']), reverse=True)[:2]
        text = self.tok.decode(self.encode(' instruction' * 256)[:256])
        singleton = [self.score_batch(text, [e])[0][0]/len(e['target']) for e in worst]
        for bs in [1, 2]:
            try:
                progress('benchmark_scoring', batch=bs)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start = time.perf_counter()
                values = []
                for i in range(0, len(worst), bs):
                    sums, counts = self.score_batch(text, worst[i:i+bs])
                    values.extend(a/b for a, b in zip(sums, counts))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                delta = max(abs(a-b) for a,b in zip(values, singleton))
                peak = torch.cuda.max_memory_allocated()
                records.append(dict(kind='score', batch=bs, seconds=elapsed,
                                    peak_bytes=peak, nll_delta=delta, passed=peak<.9*capacity and delta<.003))
            except torch.OutOfMemoryError:
                records.append(dict(kind='score', batch=bs, error='OOM', passed=False))
                gc.collect(); torch.cuda.empty_cache(); break
        safe = [r for r in records if r['kind']=='score' and r['passed']]
        assert safe, 'No safe scoring batch'
        self.batch = min(safe, key=lambda r:r['seconds'])['batch']
        for bs in [1,2,4,8]:
            progress('benchmark_generation', batch=bs)
            before = len(self.adaptations)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = time.perf_counter()
            jobs = [dict(prompt_index=i%4, prefix='You are helpful. '*20) for i in range(bs)]
            out = self.generate(jobs, 32)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()
            passed = len(self.adaptations)==before and peak<.8*capacity
            records.append(dict(kind='generation', batch=bs, seconds=elapsed, peak_bytes=peak,
                                tokens_per_second=sum(len(r['token_ids']) for r in out)/elapsed, passed=passed))
            if not passed:
                break
        safe = [r for r in records if r['kind']=='generation' and r['passed']]
        assert safe, 'No safe generation batch'
        self.gen_batch = max(safe, key=lambda r:r['tokens_per_second'])['batch']
        return dict(records=records, scoring_batch=self.batch, generation_batch=self.gen_batch,
                    gpu=torch.cuda.get_device_name(), gpu_bytes=capacity,
                    sampling_config=self.gen_config.to_dict(), adaptations=self.adaptations)
