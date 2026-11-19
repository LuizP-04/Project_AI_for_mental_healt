"""Measure the local LLMs on this GPU: load time, GPU memory, reading and writing speed.

Start Ollama first, in another terminal:

    ollama serve

Then run from the repository root:

    uv run python scripts/bench_llm.py

Each model gets a prompt shaped like one debate turn (the critic's), on two invented texts,
with its "thinking" off and on. The script also checks whether the models fit in GPU memory
together and estimates the time of a full debate against the 3-minute target (RNF03).
Nothing is saved: the results are only printed.
"""

import argparse
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama import ChatResponse, Client

from moodai.config import ConfigError, load_config
from moodai.llm.local import LocalLLMError, connect, require_models, server_version

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_SECONDS = 180  # RNF03: a full analysis in at most 3 minutes
DEBATE_ROUNDS = 2  # in each round, each debater speaks once
JUDGE_FACTOR = 2  # assumption: the judge reads and writes about twice as much as one turn
MAX_OUTPUT_TOKENS = {False: 400, True: 2048}  # without and with thinking

# Invented texts, written for this benchmark: they do not come from any dataset or person.
SAMPLE_TEXTS = {
    "pt": (
        "Faz umas três semanas que acordo cansada, mesmo dormindo mais de dez horas. Parei de ir "
        "ao vôlei com as amigas porque nada mais tem graça, e no trabalho demoro o dobro para "
        "terminar qualquer planilha. Minha mãe diz que é fase, mas sinto um peso no peito quase "
        "o dia todo. Às vezes fico irritada por coisas pequenas e depois me sinto culpada. "
        "Comecei a comer menos e perdi uns quilos sem querer. Não penso em me machucar, só "
        "queria ter vontade de fazer as coisas de novo. No mês passado meu cachorro morreu e, "
        "desde então, a casa ficou muito silenciosa."
    ),
    "en": (
        "For the past month I have been sleeping about four hours a night and I still feel full "
        "of energy. I started three new projects at work and bought a guitar, a bike and a "
        "camera in the same week. My friends say I talk too fast and jump from one topic to "
        "another. A few weeks before that I could barely get out of bed and cancelled "
        "everything. Now I feel I can do anything, although my credit card bill worries me a "
        "little. My sister thinks I should slow down, but I have never felt so productive."
    ),
}
SYSTEM_PROMPT = (
    "You are the critic in a case discussion that supports a mental health professional. "
    "This is decision support, not a diagnosis. Argue against the proposed hypothesis: point "
    "out alternative explanations and missing criteria, quoting the text word for word. "
    "Answer in Brazilian Portuguese, in at most 200 words."
)
HYPOTHESIS = "Hypothesis proposed by the classifiers: depression (probability 0.72)."


def seconds(nanoseconds: int | None) -> float:
    """Convert one of Ollama's durations (nanoseconds) to seconds."""
    return (nanoseconds or 0) / 1e9


@dataclass
class Call:
    """Timings of one request, taken from the counters that Ollama returns."""

    model: str
    language: str
    think: bool
    load_s: float
    prompt_tokens: int
    prompt_s: float
    output_tokens: int
    output_s: float
    total_s: float
    thought: bool  # the reply came with a thinking section
    truncated: bool  # the reply hit the output token limit

    @classmethod
    def from_reply(cls, model: str, language: str, think: bool, reply: ChatResponse) -> "Call":
        return cls(
            model=model,
            language=language,
            think=think,
            load_s=seconds(reply.load_duration),
            prompt_tokens=reply.prompt_eval_count or 0,
            prompt_s=seconds(reply.prompt_eval_duration),
            output_tokens=reply.eval_count or 0,
            output_s=seconds(reply.eval_duration),
            total_s=seconds(reply.total_duration),
            thought=bool(reply.message.thinking),
            truncated=reply.done_reason == "length",
        )

    @property
    def reading_rate(self) -> float:
        return self.prompt_tokens / self.prompt_s if self.prompt_s else 0.0

    @property
    def writing_rate(self) -> float:
        return self.output_tokens / self.output_s if self.output_s else 0.0

    @property
    def work_s(self) -> float:
        """Time of the request without loading the model."""
        return self.total_s - self.load_s


def gpu_used_mib() -> str:
    """GPU memory in use right now, as reported by nvidia-smi."""
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        return subprocess.run(query, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def unload_all(client: Client) -> None:
    """Remove every model from memory, so that the next request loads its model again."""
    for loaded in client.ps().models:
        client.generate(model=loaded.model, keep_alive=0)


def load(client: Client, model: str, llm: dict[str, Any]) -> float:
    """Make sure `model` is in memory, with a one-token request; return the load time."""
    reply = client.chat(
        model=model,
        messages=[{"role": "user", "content": "ok"}],
        think=False,
        keep_alive=llm["keep_alive"],
        options={"num_ctx": llm["num_ctx"], "num_predict": 1},
    )
    return seconds(reply.load_duration)


def memory_line(client: Client) -> str:
    """Describe the models now in memory: size and share on the GPU."""
    parts = [
        f"{m.model} {m.size / 2**30:.2f} GiB, {m.size_vram / m.size:.0%} on GPU"
        for m in client.ps().models
        if m.size
    ]
    return "; ".join(parts) + f" | nvidia-smi: {gpu_used_mib()} MiB in use"


def debate_turn(
    client: Client, model: str, text: str, think: bool, llm: dict[str, Any], seed: int
) -> ChatResponse:
    """Run one critic turn on `text`."""
    return client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text:\n"""\n{text}\n"""\n\n{HYPOTHESIS}'},
        ],
        think=think,
        keep_alive=llm["keep_alive"],
        options={"num_ctx": llm["num_ctx"], "seed": seed, "num_predict": MAX_OUTPUT_TOKENS[think]},
    )


def measure_model(
    client: Client, model: str, llm: dict[str, Any], seed: int, thinks: list[bool], show: bool
) -> tuple[list[float], list[Call]]:
    """Measure two loads from scratch and one debate turn per text and thinking mode."""
    loads = []
    for _ in range(2):
        unload_all(client)
        loads.append(load(client, model, llm))
    print(f"  loads: {', '.join(f'{s:.1f} s' for s in loads)} | {memory_line(client)}")
    calls = []
    for think in thinks:
        for language, text in SAMPLE_TEXTS.items():
            reply = debate_turn(client, model, text, think, llm, seed)
            call = Call.from_reply(model, language, think, reply)
            calls.append(call)
            print(
                f"  thinking {'on ' if think else 'off'} | {language} | read "
                f"{call.prompt_tokens} tokens at {call.reading_rate:.0f}/s | wrote "
                f"{call.output_tokens} tokens at {call.writing_rate:.1f}/s | {call.work_s:.1f} s"
                + (" | THOUGHT" if call.thought else "")
                + (" | CUT AT THE TOKEN LIMIT" if call.truncated else "")
            )
            if show:
                print("    " + reply.message.content.strip().replace("\n", "\n    "))
    return loads, calls


def check_together(client: Client, models: list[str], llm: dict[str, Any]) -> tuple[bool, str]:
    """Load every model, one after the other, and check whether all stay fully on the GPU."""
    unload_all(client)
    for model in models:
        load(client, model, llm)
    loaded = {m.model: m for m in client.ps().models}
    together = all(
        name in loaded and loaded[name].size_vram == loaded[name].size for name in models
    )
    return together, memory_line(client)


def debate_seconds(
    turn_s: dict[str, float],
    load_s: dict[str, float],
    debaters: tuple[str, str],
    judge: str,
    together: bool,
) -> float:
    """Estimate a full debate: rounds of turns, then the judge, plus every model load needed."""
    sequence = list(debaters) * DEBATE_ROUNDS + [judge]
    total, in_memory = 0.0, set()
    for position, model in enumerate(sequence):
        if model not in in_memory:
            total += load_s[model]
            in_memory = in_memory | {model} if together else {model}
        is_judge = position == len(sequence) - 1
        total += turn_s[model] * (JUDGE_FACTOR if is_judge else 1)
    return total


def print_estimates(
    models: list[str],
    loads: dict[str, list[float]],
    calls: list[Call],
    thinks: list[bool],
    together: bool,
) -> None:
    """Print the estimated time of a full debate for each role assignment and thinking mode."""
    load_s = {model: statistics.mean(loads[model]) for model in models}
    debaters = (models[0], models[1]) if len(models) > 1 else (models[0], models[0])
    print(
        f"\nFull debate: {DEBATE_ROUNDS} rounds x 2 debaters + judge (judge turn = "
        f"{JUDGE_FACTOR}x a debate turn), including model loads. Target: {TARGET_SECONDS} s."
    )
    for think in thinks:
        turn_s = {
            model: statistics.mean(c.work_s for c in calls if c.model == model and c.think == think)
            for model in models
        }
        for judge in dict.fromkeys(debaters):
            estimate = debate_seconds(turn_s, load_s, debaters, judge, together)
            verdict = "within target" if estimate <= TARGET_SECONDS else "OVER TARGET"
            print(
                f"  thinking {'on ' if think else 'off'} | debaters {debaters[0]} and "
                f"{debaters[1]}, judge {judge}: about {estimate:.0f} s ({verdict})"
            )


def main() -> int:
    """Run the benchmark and return the process exit code."""
    parser = argparse.ArgumentParser(description="Measure the local LLMs on this GPU.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument("--no-think", action="store_true", help="skip the runs with thinking on")
    parser.add_argument("--show", action="store_true", help="print the models' answers")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        llm, seed = config["llm"], config["seed"]
        models = list(llm["models"])
        version = server_version(llm["host"], timeout_s=10)
        client = connect(llm["host"], llm["timeout_s"])
        require_models(client, models)
    except (ConfigError, KeyError, LocalLLMError) as error:
        print(f"Cannot run the benchmark: {error}")
        return 1

    thinks = [False] if args.no_think else [False, True]
    print(f"Ollama {version} at {llm['host']} | context {llm['num_ctx']} tokens | seed {seed}")
    loads: dict[str, list[float]] = {}
    calls: list[Call] = []
    for model in models:
        print(f"\n{model}")
        loads[model], model_calls = measure_model(client, model, llm, seed, thinks, args.show)
        calls += model_calls

    together = True
    if len(models) > 1:
        together, memory = check_together(client, models, llm)
        print(f"\nAll models loaded together: {memory}")
        print(
            "  They fit together on the GPU: no swapping between turns."
            if together
            else "  They do not fit together: Ollama swaps them, and every swap costs a load."
        )
    print_estimates(models, loads, calls, thinks, together)
    unload_all(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
