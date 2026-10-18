"""The assistant pipeline used by the app: crisis check -> out-of-domain gate -> fine-tuned model.

The crisis check is a rule, not a model, and runs first: the gate was measured to decline some suicide-related
questions (docs/ood_gate.md), so a crisis message must never depend on it.
"""
import re
import time
from pathlib import Path

from .infer import GEN_KWARGS, build_prompt

CRISIS = re.compile(
    r"\b(suicid\w*|kill(?:ing)? (?:my|him|her|them)sel(?:f|ves)|end(?:ing)? (?:my|his|her|their) (?:own )?life|"
    r"wants? to die|don'?t want to live|no reason to live|self[- ]?harm\w*|hurt(?:ing)? (?:my|him|her)self|"
    r"cutting (?:my|him|her)self|overdos\w*)\b", re.IGNORECASE)

CRISIS_MSG = (
    "**If you or someone you know may be thinking about suicide or self-harm, please reach out for help now.** "
    "Call your local emergency number or go to the nearest health facility. Free, confidential crisis lines "
    "for your country are listed at [findahelpline.com](https://findahelpline.com). If someone is in immediate "
    "danger, stay with them and get help.")

DECLINE_MSG = (
    "I can only answer questions about **human health and medicine**, such as symptoms, diseases, treatments, "
    "medicines, pregnancy, mental health or nutrition. This question looks outside that scope, so I won't answer "
    "it. If it is a health question, try rephrasing it with more detail about the health concern.")

DISCLAIMER = (
    "_This answer comes from a small AI model fine-tuned on AfriMed-QA. It can be wrong and is not medical "
    "advice. For diagnosis or treatment, see a qualified health professional. In an emergency, go to the "
    "nearest health facility._")


class Assistant:
    def __init__(self, model, tok, gate=None, max_new_tokens: int = 128):
        self.model, self.tok, self.gate, self.max_new_tokens = model, tok, gate, max_new_tokens

    @classmethod
    def load(cls, base_model: str, adapter: str | None = None, gate_path=None, threads: int | None = None,
             max_new_tokens: int = 128) -> "Assistant":
        """CPU load: fp32 base model, LoRA adapter merged in for faster generation."""
        import torch
        import transformers
        from packaging import version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .gate import MedicalGate

        if threads:
            torch.set_num_threads(threads)
        dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
        # Training didn't change the tokenizer, so the base model's is identical if the adapter has none.
        has_tok = adapter and any((Path(adapter) / f).exists() for f in ("tokenizer.json", "tokenizer_config.json"))
        tok = AutoTokenizer.from_pretrained(adapter if has_tok else base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, **{dtype_kw: torch.float32})
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        gate = MedicalGate.load(gate_path) if gate_path else None
        return cls(model, tok, gate, max_new_tokens)

    def _stream_answer(self, question: str):
        """Yield the answer text as it grows (greedy decoding, same settings as evaluation)."""
        from threading import Thread

        from transformers import TextIteratorStreamer

        prompt = build_prompt(self.tok, {"question_type": "saq", "question": question})
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        worker = Thread(target=self.model.generate, kwargs=dict(
            **enc, streamer=streamer, max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
            **GEN_KWARGS))
        worker.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text
        worker.join()

    def stream(self, question: str):
        """Yield partial results while the answer is generated; the last one is final.

        Each result: markdown to show, route (answered / declined / crisis+...), gate probability,
        elapsed seconds, the bare answer text, and done.
        """
        t0 = time.perf_counter()
        question = (question or "").strip()
        result = {"markdown": "Please type a health question.", "route": "empty", "prob_medical": None,
                  "seconds": 0.0, "answer": None, "done": True}
        if not question:
            yield result
            return
        is_med, p = self.gate.is_medical(question) if self.gate else (True, None)
        crisis = bool(CRISIS.search(question))
        head = [CRISIS_MSG] if crisis else []
        result.update(route=("crisis+" if crisis else "") + ("answered" if is_med else "declined"),
                      prob_medical=p, done=False)
        if is_med:
            for text in self._stream_answer(question):
                result.update(markdown="\n\n".join(head + [text]), answer=text,
                              seconds=round(time.perf_counter() - t0, 2))
                yield result
            tail = [result["answer"] or "", DISCLAIMER]
        else:
            tail = [] if crisis else [DECLINE_MSG]
        result.update(markdown="\n\n".join(head + tail), seconds=round(time.perf_counter() - t0, 2), done=True)
        yield result

    def respond(self, question: str) -> dict:
        """Non-streaming: the final result of stream()."""
        for result in self.stream(question):
            pass
        return result
