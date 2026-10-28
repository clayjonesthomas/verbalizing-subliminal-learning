"""Train one soft prompt, read it greedily, then run the original SALVE search.

Only explicit paths and a seed are required. No pod lifecycle or report code is
imported here; an optional callback receives progress after durable writes.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

from .protocol import (
    read_json as read,
    write_json as write,
    sha256,
    validate_dataset,
    question_key,
    MARKERS,
    QUOTAS,
)


def select_behavior_questions(queries, data, seed, count):
    """Select the same balanced seed-specific evaluation sample as the first run."""
    train_keys = {question_key(row) for rows in data.values() for row in rows}
    keys = [question_key(row) for row in queries]
    if len(set(keys)) != len(keys) or train_keys.intersection(keys):
        raise ValueError("Behavior questions duplicate or overlap training/validation")
    chosen = []
    for category in QUOTAS["train"]:
        rows = [row for row in queries if row["category"] == category]
        if len(rows) < count:
            raise ValueError(f"Insufficient behavior questions for {category}")
        random.Random(seed).shuffle(rows)
        chosen.extend(rows[:count])
    return chosen


def summarize_behavior(replies):
    """Prefix recall and any-position off-category marker rate; no semantic grader."""
    stats = {}
    for category, marker in MARKERS.items():
        positive = [row for row in replies if row["category"] == category]
        negative = [row for row in replies if row["category"] != category]
        stats[category] = dict(
            n=len(positive),
            hit=sum(row["response"].startswith(marker) for row in positive)
            / len(positive),
            false_positive=sum(marker in row["response"] for row in negative)
            / len(negative),
        )
    return dict(
        rates=stats,
        unfinished=sum(not row["finished"] for row in replies),
        responses=replies,
    )


class SeedExperiment:
    def __init__(self, dataset, queries, output, seed, on_progress=None):
        self.output = Path(output)
        self.config = read(Path(__file__).with_name("training_config.json"))
        self.config.update(
            seed=seed,
            dataset_path=str(dataset),
            queries_path=str(queries),
            dataset_sha256=sha256(dataset),
        )
        self.on_progress = on_progress

    def progress(self, phase, **values):
        record = dict(phase=phase, time=time.time(), seed=self.config["seed"], **values)
        write(self.output / "progress.json", record)
        if self.on_progress is not None:
            self.on_progress(self.output, record)

    def run(self):
        # Refuse to append a new training trajectory to an existing seed directory.
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(f"Output directory must be empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            self._run()
        except Exception as exc:
            write(
                self.output / "failure.json",
                dict(type=type(exc).__name__, message=str(exc)),
            )
            self.progress("failed")
            raise

    def fit(self, model, tok, data):
        import torch
        from salve.objectives import NLLObjective
        from salve.salve import (
            warmup_cosine,
            verbalize,
            clean_verbalization,
            VERBALIZE_PROMPTS,
        )

        obj = NLLObjective(model, tok)

        # Score the complete stored answer, including its prefix and assistant EOS.
        def prepare(rows):
            return obj.prepare(
                [
                    (
                        r["user"],
                        tok.encode(r["response"], add_special_tokens=False)
                        + [tok.eos_token_id],
                    )
                    for r in rows
                ]
            )

        train, val = (prepare(data["train"]), prepare(data["validation"]))
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed_all(self.config["seed"])
        E = model.get_input_embeddings().weight
        z = (
            (
                torch.randn(
                    self.config["soft_tokens"],
                    E.shape[1],
                    device=E.device,
                    dtype=E.dtype,
                )
                * E.std()
            )
            .detach()
            .requires_grad_(True)
        )
        total = self.config["epochs"] * math.ceil(
            len(train) / self.config["batch_size"]
        )
        opt = torch.optim.Adam(
            [z], lr=self.config["lr"], weight_decay=self.config["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, warmup_cosine(total, round(total * self.config["warmup"]))
        )
        queue = []
        micro = self.config["micro_batch_size"]
        greedies = []
        assert getattr(model.config, "attention_dropout", 0) == 0

        def train_mode():
            model.train()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False

        def eval_mode():
            model.gradient_checkpointing_disable()
            model.eval()
            model.config.use_cache = True

        train_mode()
        for step in range(1, total + 1):
            while len(queue) < self.config["batch_size"]:
                queue.extend(torch.randperm(len(train)).tolist())
            batch, queue = (
                queue[: self.config["batch_size"]],
                queue[self.config["batch_size"] :],
            )
            while True:
                opt.zero_grad()
                try:
                    loss = obj.loss(
                        z,
                        [train[i] for i in batch],
                        backward=True,
                        mini_batch_size=micro,
                    )
                    break
                except torch.OutOfMemoryError:
                    opt.zero_grad()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
            lr = opt.param_groups[0]["lr"]
            norm = float(torch.nn.utils.clip_grad_norm_([z], 1.0))
            opt.step()
            sched.step()
            with (self.output / "training.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=step,
                            nll=loss,
                            loss=loss,
                            unclipped_grad_norm=norm,
                            learning_rate=lr,
                            time=time.time(),
                            micro_batch_size=micro,
                        )
                    )
                    + "\n"
                )
            if step % 25 == 0 or step == total:
                checkpoint = dict(
                    z=z.detach(),
                    optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),
                    queue=queue,
                    step=step,
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    micro=micro,
                )
                torch.save(checkpoint, self.output / "training.pt.tmp")
                (self.output / "training.pt.tmp").replace(self.output / "training.pt")
                self.progress("training", step=step, total=total, nll=loss)
            if step % self.config["greedy_every"] == 0 or step == total:
                self.progress("greedy readout", step=step, total=total)
                eval_mode()
                # Readouts must not perturb the subsequent training batch RNG.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    q = VERBALIZE_PROMPTS[0]
                    raw = verbalize(
                        model,
                        tok,
                        z.detach(),
                        q,
                        max_new_tokens=self.config["greedy_max_tokens"],
                        temperature=0,
                    )
                greedies.append(
                    dict(
                        step=step,
                        time=time.time(),
                        raw=raw,
                        text=clean_verbalization(raw, q["stop"]),
                    )
                )
                write(self.output / "greedy_history.json", greedies)
                torch.save(z.detach(), self.output / f"soft-step-{step}.pt")
                self.progress("greedy readout saved", step=step, total=total)
                train_mode()
        eval_mode()
        torch.save(z.detach(), self.output / "soft_prompt.pt")
        return (obj, z.detach(), train, val, greedies[-1]["text"])

    def behavior(self, model, tok, queries, text):
        import torch

        replies = []
        for start in range(0, len(queries), 4):
            chunk = queries[start : start + 4]
            texts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": text},
                        {"role": "user", "content": r["user"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in chunk
            ]
            inp = tok(
                texts, add_special_tokens=False, padding=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                ids = model.generate(
                    **inp,
                    max_new_tokens=self.config["behavior_max_tokens"],
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )[:, inp.input_ids.shape[1] :].tolist()
            for row, ts in zip(chunk, ids):
                stop = next(
                    (i for i, t in enumerate(ts) if t == tok.eos_token_id), None
                )
                reply = tok.decode(
                    ts[:stop] if stop is not None else ts, skip_special_tokens=True
                ).strip()
                replies.append(dict(row, response=reply, finished=stop is not None))
        return summarize_behavior(replies)

    def _run(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from salve.salve import SearchConfig, beam_search, selection_subset

        self.output.mkdir(parents=True, exist_ok=True)
        p = Path(self.config["dataset_path"])
        if sha256(p) != self.config["dataset_sha256"]:
            raise ValueError("Dataset changed after experiment construction")
        data = read(p)
        validate_dataset(data)
        chosen = select_behavior_questions(
            read(self.config["queries_path"])["eval"],
            data,
            self.config["seed"],
            self.config["behavior_rows_per_category"],
        )
        write(
            self.output / "manifest.json",
            dict(
                config=self.config,
                dataset_sha256=self.config["dataset_sha256"],
                queries_sha256=sha256(self.config["queries_path"]),
            ),
        )
        self.progress("loading model")
        tok = AutoTokenizer.from_pretrained(
            self.config["model"], revision=self.config["revision"]
        )
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.config["model"],
                revision=self.config["revision"],
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        for p in model.parameters():
            p.requires_grad = False
        obj, z, train, val, greedy = self.fit(model, tok, data)
        subset = selection_subset(
            train, self.config["search"]["n_select"], self.config["seed"]
        )
        result = dict(complete=False, seed=self.config["seed"], methods={})
        result["methods"]["soft"] = dict(val_nll=obj.loss(z, val, mini_batch_size=1))
        result["methods"]["greedy"] = dict(
            text=greedy,
            val_nll=obj.loss_of_text(greedy, val, mini_batch_size=1),
            selection_nll=obj.loss_of_text(greedy, subset, mini_batch_size=1),
        )
        write(self.output / "result.json", result)
        self.progress("SALVE search")
        original = obj.loss_of_text
        cache = {}

        def score(text, examples, mini_batch_size=1):
            if text not in cache:
                cache[text] = original(text, examples, mini_batch_size)
                write(self.output / "search_scores.json", cache)
                self.progress(
                    "SALVE search", candidates=len(cache), best_nll=min(cache.values())
                )
            return cache[text]

        obj.loss_of_text = score
        try:
            nodes, best = beam_search(
                obj,
                z,
                train,
                SearchConfig(**self.config["search"]),
                self.config["seed"],
                sampling_seed=self.config["seed"],
            )
        finally:
            obj.loss_of_text = original
        write(self.output / "search.json", dict(nodes=nodes, best=best))
        for name, text in [
            ("salve", best["text"]),
            ("teacher", self.config["teacher_prompt"]),
            ("empty", ""),
        ]:
            self.progress("held-out NLL " + name)
            result["methods"][name] = dict(
                text=text,
                val_nll=obj.loss_of_text(text, val, mini_batch_size=1),
                selection_nll=obj.loss_of_text(text, subset, mini_batch_size=1),
            )
            write(self.output / "result.json", result)
        write(self.output / "behavior_questions.json", chosen)
        for name in ["greedy", "salve", "teacher", "empty"]:
            self.progress("behavior " + name)
            b = self.behavior(model, tok, chosen, result["methods"][name]["text"])
            write(self.output / f"behavior-{name}.json", b)
            result["methods"][name]["behavior"] = {
                k: v for k, v in b.items() if k != "responses"
            }
            write(self.output / "result.json", result)
        result["complete"] = True
        write(self.output / "result.json", result)
        self.progress("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    SeedExperiment(args.data, args.queries, args.output, args.seed).run()


if __name__ == "__main__":
    main()
