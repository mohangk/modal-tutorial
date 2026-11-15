# Demoscener: LoRA, customization, and reproduction

Read [general LoRA basics](../lora-basics.md) first. This case study separates the
published artifacts, the author's account, our measurements, and suggested
experiments. We have run inference and inspected adapters; we have not reproduced
the author's training.

## What is inside the downloaded adapters?

Our script pins `datavorous/Demoscener-7B` revision
`315703a76fb81f7771225a092cd1f2af77ee12ba`. Each stage contains a JSON
configuration and a safetensors weight file. Direct inspection finds:

| Property | Value |
| --- | --- |
| Base used by our loader and recommended in the model card | `Qwen/Qwen2.5-Coder-7B` Base |
| Base recorded in adapter metadata | `unsloth/Qwen2.5-Coder-7B` |
| Method | LoRA, causal language modeling |
| Rank `r` | 16 |
| Alpha | 32 |
| Standard update scale | 32 / 16 = 2 |
| LoRA dropout / trained bias | 0 / none |
| Transformer blocks covered | 28 |
| Projections per block | 7 |
| A/B matrix pairs per adapter | 196 |
| Tensors per adapter | 392 |
| Parameters per adapter | 40,370,176 |
| Stored tensor type | FP32 |
| File size per adapter | About 161.5 MB, including a small header |

The base contains about 7.616 billion parameters. The LoRA parameters amount to
roughly **0.53% of that count**. This is a parameter-count comparison, not the
percentage of memory the complete inference process needs.

The targets are `q_proj`, `k_proj`, `v_proj`, `o_proj` in attention, plus
`gate_proj`, `up_proj`, `down_proj` in the feed-forward network. These are matrix
operations distributed across the network, not one new bottom layer.

For the first block's query projection:

```text
original W: 3,584 × 3,584 = 12,845,056 parameters
LoRA A:        16 × 3,584 =     57,344 parameters
LoRA B:     3,584 ×    16 =     57,344 parameters

effective query weights = W + 2 × B A
```

Other projections can have different dimensions; the query example is not the
shape of every matrix. The recorded shapes include, for example, a feed-forward
up-projection B of `[18944, 16]`.

Inspect the artifacts yourself, from this folder:

```bash
uv run python inspect_lora.py
uv run python inspect_lora.py --check-logits --output outputs/lora-inspection.json
```

The first command reads cached adapter metadata and tensor headers. The optional
second command also loads the base and compares next-token scores on the CPU.
Neither trains or downloads missing weights; run `run.py --download-only` first
if the models are not cached. See [the inspector](inspect_lora.py) and
[the pinned adapter configuration](https://huggingface.co/datavorous/Demoscener-7B/blob/315703a76fb81f7771225a092cd1f2af77ee12ba/grpo/adapter_config.json).

## How the author customized the model

The author's [article](https://datavorous.com/writing/demoscener/) describes:

1. **CPT:** next-token learning from Dwitter animation code.
2. **Curriculum:** continue training on the top 5,000 programs, increasingly
   favoring higher popularity ranks over 400 steps. The top 1,000 eventually
   receive about 67% of sampling probability.
3. **GRPO with human feedback:** sample eight continuations per unfinished
   program; select a favorite or skip the group; update the policy and repeat.
   The author reports roughly 9–10 rounds of 500 groups and about 40,000 animations
   inspected, followed by a friend's blinded checkpoint comparison.

Here **CPT** means **continued pretraining**. It uses target tokens supplied by
existing code, rather than instruction-response pairs. Typical instruction SFT
can use the same next-token loss, often restricted to response tokens; the data
setup and purpose differ.

The curriculum stage is still continued pretraining: the changing part is how
often popular programs are sampled. Curriculum is not inherently SFT and can
also be used with RL. LoRA is the parameter-efficient training method used to
represent the published learned adjustments, not simply a way to serialize a
full-model training run afterward. A precise description of this stage is
**continued pretraining with popularity-based curriculum sampling using LoRA**.
See [the shared training concepts](../training-basics.md).

The [model card](https://huggingface.co/datavorous/Demoscener-7B) identifies the
published stages as independently loadable LoRA adapters: curriculum includes
the prior CPT stage, and GRPO includes the previous stages. Each works directly
with the base; do not add all three updates together. The card also says later
stages were trained on quantized bases. Our BF16 comparison follows its loading
example, rather than reconstructing that training precision.

This tells us the training progression and final adapter settings. It does not
tell us exactly how transitions were implemented: continuing an adapter, merging
and re-extracting updates, and managing quantized weights require training code
and checkpoint provenance to resolve. The `unsloth` metadata is a useful clue,
not evidence of all framework versions or settings.

## Why the first outputs looked so similar

Two different questions need different explanations:

- **Repeated runs:** our saved initial comparisons used the same model stage,
  same prefix, and temperature zero. Greedy decoding selects the most probable
  token at each step; changing the random seed alone will not explore alternatives.
- **Different stages:** the prefix already specifies a canvas width, an 80-item
  loop, and a rectangle-drawing call. The models continue from inside its argument
  list. Positions and sizes can differ while all outputs still look like related
  rectangle animations. The stages also share a base and training lineage.

These are explanations consistent with the observed setup, not proof that every
stage has the same capabilities. Low rank does not require similar-looking output,
and the final training stage is not guaranteed to match your visual taste.

We checked that this is not simply three copies of one downloaded adapter: the
files have distinct SHA-256 hashes. A local prediction check on the original
prefix gave:

| Active adapter | Most probable next token | Its probability |
| --- | --- | ---: |
| CPT | `4` | 36.6% |
| Curriculum | `9` | 35.7% |
| GRPO | `4` | 32.1% |

All three also changed scores from the base. These are measurements for one
prefix on our shared BF16 base, not quality scores. Even when the top token is
the same, the probability distribution can change. Greedy decoding hides much
of that difference by picking just the top token.

Try a less restrictive prefix, positive temperature, and multiple seeds using
[the learning exercise](README.md#learning-exercise). Judge several examples,
separate syntax validity from preference, and allow enough tokens for completion.

## What we can reproduce

### A. Reproduce inference now

Our pinned revisions and separate lockfile make the current loading comparison
repeatable on compatible hardware. Start with greedy decoding, then sampling.
Exact outputs can still differ across numerical precision and software/hardware.
This reuses the author's training; it is not a new trained model.

### B. Rebuild a small version of the training pipeline

This is a proposed learning project, **not an exact author recipe**. We have not
implemented or launched it. Start with CPT before adding the human-feedback loop.

1. **Prepare data.** The [published dataset](https://huggingface.co/datasets/datavorous/Demoscener-Dweets-25k)
   has 25,002 training and 1,316 validation programs with code, source IDs,
   popularity ranks, and remix-family metadata. Preserve its split and provenance.
   Code strings do not overlap across its splits; related remixes can still be a
   generalization concern. Consider holding out families for an additional test.
2. **Initialize a LoRA model.** Use the compatible Base model and the published
   rank/targets. Begin with small batches and explicitly record tokenizer, EOS,
   packing, precision, learning rate, optimizer, and seeds.
3. **Train CPT.** Use code tokens as next-token targets, masking padding from the
   loss. Only adapters receive optimizer updates. Validate on unseen programs and
   save the adapter separately. Record both loss and generation validity.
4. **Add a curriculum.** Continue from the CPT result and bias example sampling
   toward the top popularity ranks. A gradual weighting schedule is a design
   choice to document; the article's endpoint is not a complete sampling formula.
   Keep validation examples out of training and rank selection.
5. **Collect preferences.** Generate eight completions for each prefix with the
   current policy, render them in isolation, and record your choice or skip along
   with the seed, code, and checkpoint. New policy updates need fresh generations.
6. **Perform GRPO updates.** One illustrative reward is 1 for the chosen result
   and 0 for the others, with group-relative advantages. This is our teaching
   choice; the author's exact reward mapping, normalization, clipping, KL settings,
   and update schedule are not published in the linked materials. Use a trainer
   with a documented GRPO implementation instead of treating `reward - mean` as
   the complete loss. Save new adapter checkpoints.
7. **Evaluate.** Use held-out prefixes and blind comparisons across checkpoints.
   Track runtime success separately from preference; check diversity, repetition,
   and training-example copying. The last checkpoint need not be your favorite.

An adapter configuration matching the published standard-LoRA settings is:

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

Creating this configuration performs no training. A future training script would
load a base, attach it with `get_peft_model(base, config)`, construct the data and
loss, perform optimizer updates, and save adapters. Loading `grpo/` today loads
already-trained values instead of initializing a fresh adapter. See
[PEFT's training configuration guidance](https://huggingface.co/docs/peft/main/package_reference/lora).

### C. Understand the limits of exact reproduction

The available article, model card, dataset, and adapter configuration do not
provide a complete executable training recipe, original human preference log,
every model/optimizer checkpoint, full hyperparameters, or exact numerical setup.
The selected source model revision and quantization details matter too. We can
reproduce the broad procedure and run the published artifacts; we cannot claim
the same weights or training metrics without those missing inputs.

Training a new 7B adapter still needs substantial compute even though only a
fraction of parameters are trained. The current CPU setup is for inference and
inspection. No paid GPU run starts without explicit confirmation.

Back to [the exercise](README.md#learning-exercise), [LoRA basics](../lora-basics.md),
or [the learning index](../README.md).
