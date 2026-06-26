import atexit
from dataclasses import fields
from time import perf_counter
# Defer heavy imports (tqdm, transformers, torch) to runtime to avoid
# import-time failures in minimal smoke tests that only validate
# module-level code.

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
# Import ModelRunner lazily in __init__ to avoid importing heavy torch-backed
# modules during quick import-time checks.


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        # Import ModelRunner lazily to avoid requiring torch at module import
        # time. ModelRunner depends on CUDA/torch and should only be loaded
        # when an actual engine instance is constructed.
        from nanovllm.engine.model_runner import ModelRunner
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        # Import tokenizer lazily to avoid requiring `transformers` during
        # import-time checks or basic unit tests that don't instantiate the
        # engine.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # Import Scheduler lazily to avoid pulling in heavy dependencies
        # (xxhash, numpy) during module import.
        from nanovllm.engine.scheduler import Scheduler
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for p in getattr(self, "ps", []):
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def stream_generate(self, prompt: str | list[int], sampling_params: SamplingParams, chunk_size: int = 1):
        """Genuine incremental streaming generator.

        This drives the engine step loop and yields newly produced token text
        as soon as they are appended to the Sequence. It requires no external
        buffering and produces pass-through tokens.

        Args:
            prompt: input prompt string or token id list
            sampling_params: SamplingParams for generation
            chunk_size: number of tokens to accumulate before yielding (default 1)

        Yields:
            str: decoded text for the newly produced tokens (may be short)
        """
        # Create sequence and get a handle
        seq = self.add_request(prompt, sampling_params)
        # Track how many completion tokens we've already yielded
        yielded = 0

        # Continue stepping until the engine reports the sequence finished
        while not seq.is_finished:
            try:
                _, _ = self.step()
            except Exception:
                # On unexpected errors, stop iteration
                break
            # Determine newly added completion tokens
            new_len = seq.num_completion_tokens
            if new_len > yielded:
                # Extract token ids for newly produced tokens
                new_tokens = seq.completion_token_ids[yielded:new_len]
                yielded = new_len
                # Decode tokens to text and yield in chunks if requested
                # Note: tokenizer.decode expects a list of ids
                # Yield in sub-chunks of chunk_size to keep latency low
                for i in range(0, len(new_tokens), chunk_size):
                    chunk_ids = new_tokens[i : i + chunk_size]
                    text = self.tokenizer.decode(chunk_ids)
                    yield text
        # After finishing, ensure any remaining tokens are yielded
        final_len = seq.num_completion_tokens
        if final_len > yielded:
            remaining = seq.completion_token_ids[yielded:final_len]
            for i in range(0, len(remaining), chunk_size):
                chunk_ids = remaining[i : i + chunk_size]
                text = self.tokenizer.decode(chunk_ids)
                yield text
