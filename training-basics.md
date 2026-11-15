# Training basics: CPT, SFT, RL, curriculum, and LoRA

These terms describe different aspects of training. They are not a list of
mutually exclusive alternatives. A model can undergo continued pretraining with
a curriculum while using LoRA, then receive reinforcement learning updates.

Use this page to connect [the RL notes](reinforcement-learning-basics.md),
[LoRA basics](lora-basics.md), and [the Demoscener example](demoscener/lora-and-training.md).

## Pretraining, continued pretraining, and SFT

A language model can learn by predicting the next token in existing text or code.
The following token supplies a target. Training compares the prediction with that
target, calculates a loss, and updates trainable parameters through an optimizer.

**Initial pretraining** usually starts with randomly initialized weights and a
large corpus. **Continued pretraining (CPT)** starts with a pretrained model and
continues that kind of learning on additional data, often from a particular
domain. It extends existing capabilities rather than starting from scratch.

Starting with existing weights is not enough to define CPT: SFT and RL commonly
do that too. An initial pretraining job can also resume from a checkpoint. CPT
describes a training stage and purpose, not a fundamentally new loss function.

In common LLM usage, **supervised fine-tuning (SFT)** usually means training on
curated inputs and desired responses. CPT and SFT commonly share a next-token
cross-entropy loss; they differ in data organization, intended behavior, and
often which tokens contribute to the loss.

| | CPT | Typical instruction SFT |
| --- | --- | --- |
| Example data | Existing JavaScript programs | Requests paired with desired programs |
| Learning target | Following tokens in the programs | Tokens in the desired response |
| Loss applied to | Usually the training text broadly | Often only the response tokens |
| Typical goal | Become familiar with a domain's language | Learn to respond to requests in a desired way |

For CPT, an example could be a raw program:

```javascript
x.fillRect(100, 200, 50, 50)
```

For SFT, an example could instead be:

```text
Request: Draw a square that moves horizontally.
Desired response: [a working JavaScript program]
```

In both cases, prediction has known token targets. CPT is called
**self-supervised** because those targets come automatically from the corpus,
rather than people labeling every next token. This is compatible with using the
same mathematical prediction loss as supervised training. The customary names
are useful distinctions, not completely separate mathematical mechanisms.

## Where reinforcement learning differs

In RL, the model generates behavior and receives rewards. A policy-training
objective uses that feedback to adjust future behavior. There need not be a
supplied target token for every generated step.

For Demoscener, an existing program provides token targets during CPT; a human's
choice among generated programs provides feedback during GRPO. Our math project
instead scores generations with a numeric answer checker.

RL training still uses losses, gradients, and optimizers. A reward is an input to
constructing the training objective; it is not a replacement for all that machinery.
See [the RL training loop](reinforcement-learning-basics.md#the-training-loop).

## Curriculum controls the examples, not necessarily the objective

**Curriculum learning** changes which training examples or tasks are presented,
or how often they are sampled, over time. Easy-to-hard progression is one example;
it is not required. A curriculum can favor examples with a desired quality or
style instead.

| Training setup | Possible curriculum |
| --- | --- |
| CPT | Gradually change which raw documents or programs are sampled |
| SFT | Begin with simpler instruction-response pairs, then introduce harder ones |
| RL | Gradually present more difficult tasks to the policy |

Demoscener's popularity curriculum continues next-token training while
increasingly favoring higher-ranked programs. It is best described as
**continued pretraining with curriculum sampling**, rather than automatically
calling it instruction SFT. A different project could use a curriculum with SFT
or RL. See [the author's concrete stages](demoscener/lora-and-training.md#how-the-author-customized-the-model).

## LoRA is a training method, not just a saved file

**LoRA is a parameter-efficient fine-tuning method.** It freezes original base
weights and learns small A/B matrices that add adjustments to selected operations.
It constrains and represents updates **during training**. It does not ordinarily
mean training every original weight and compressing the resulting difference later.

LoRA can use a next-token loss or an RL objective. Its learned matrices can then
be saved as an adapter and loaded with a compatible base. These are distinct steps:

```text
train A/B matrices → save the learned adapter → load it for inference later
```

Loading reproduces the learned customization, not the original training process.
Reproducing training additionally needs data, settings, software, seeds, and other
details. [LoRA basics](lora-basics.md) explains the matrix arithmetic and saving,
loading, switching, and merging.

## Put the choices together

| Aspect | Question | Examples |
| --- | --- | --- |
| Data and training stage | What examples and starting model are we using? | CPT on a code corpus; SFT on request-response pairs |
| Learning objective | What feedback drives the loss? | Next-token targets; GRPO rewards |
| Curriculum or sampling schedule | How does example selection change over training? | Increasingly favor popular or harder examples |
| Parameter-update method | Which parameters learn, and how are updates represented? | Full fine-tuning; LoRA A/B matrices |
| Inference decoding | How do trained weights generate outputs? | Greedy decoding; sampling with temperature and a seed |

For Demoscener's curriculum stage, several descriptions apply at once:

> Continue pretraining on existing code, increasingly sample popular Dweets,
> and learn a LoRA adjustment to the base model.

Later, the GRPO stage uses human preferences to train a policy and is also
published as a LoRA adapter. The names `cpt/`, `curriculum/`, and `grpo/` label
checkpoints in the author's progression; they are not three mutually exclusive
categories of training methods.

## Check your understanding

1. Why do CPT and SFT both use known target tokens, even though CPT is usually
   called self-supervised? Which part of an instruction SFT example might be
   excluded from its loss?
2. Can an SFT run use both a curriculum and LoRA? Describe one example. Which
   choice changes the examples, and which changes the trainable parameters?
3. What changes if we replace next-token learning with human preference GRPO but
   keep LoRA? What could remain the same?
4. Does saving an adapter perform training? Does loading one reproduce the
   original optimizer updates? Explain what is actually preserved.
5. How could a seed change an animation without any training? Connect this to
   [the inference exercise](demoscener/README.md#learning-exercise).

Return to [the learning index](README.md).
