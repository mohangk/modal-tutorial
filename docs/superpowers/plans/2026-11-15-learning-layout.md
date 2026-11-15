# Learning repository and LoRA implementation plan

**Goal:** Make the repository a learning index with independent experiments,
explain LoRA using the actual Demoscener artifacts, and investigate similar outputs.

**Spec:** The user's 2026-11-15 request in this conversation: diagnose output
similarity; add general LoRA basics and a concrete Demoscener explanation and
reproduction path; move the original project into a subdirectory and index the
tutorials from the root.

**Architecture:** Root Markdown documents teach shared concepts. `modal-gsm8k/`
owns the original pinned Modal environment, scripts, operating notes, and run
history. `demoscener/` retains its environment, cached models, saved outputs, and
case study. Each experiment uses its own `uv` project.

**Constraints:** Preserve `modal==1.6.1`, `modal-dojo==0.1.1`, Python 3.12, existing
training settings, model caches, and saved outputs. No paid GPU launches. Do not
claim exact training reproduction from incomplete published information. Preserve
the user's uncommitted work; leave changes reviewable without committing.

## Tasks and verification

- [x] Diagnose repeated runs vs similar stages using saved outputs, adapter
  weights, and next-token predictions; record measured findings in Demoscener docs.
- [x] Move the original files/environment to `modal-gsm8k/`, rewrite its README
  and path-specific operating notes, and replace root README/AGENTS with an index
  and shared agreements. Verify the moved CLI and imports without launching jobs.
- [x] Add root `lora-basics.md`: weights, layers, frozen base, low-rank A/B update,
  parameter arithmetic, alpha/r, loading vs training vs merging, and connection
  between LoRA and the choice of supervised/RL objective.
- [x] Add `demoscener/lora-and-training.md`: actual target modules, rank, scaling,
  parameter count, author stages, data, source-backed reproduction outline, and
  clearly labeled missing details. Add a runnable metadata inspection example.
- [x] Explain existing prefix/temperature/seed controls with concrete commands;
  preserve deterministic defaults and old output compatibility. Use the existing
  CLI rather than adding a second sampling interface.
- [x] Verify small CPU sampling experiments. Verify docs, local links, moved lock
  compatibility, scripts, and saved output metadata. Get a fresh review of the
  resulting changes and fix material findings.

## Review focus

Moved virtualenv entrypoints must point at the new location. No Markdown links or
commands should refer to the old script locations. Existing previews and caches
must remain intact. Stage switching must select one independent adapter at a time.
Repeated runs must retain their own output directories and seeds. General explanations
must separate measured facts, author reports, and proposed reproduction choices.

## Progress

- Initial saved comparisons are deterministic (temperature zero) and use an
  identical rectangle-loop prefix. Adapter files have distinct SHA-256 hashes and
  each contains 40,370,176 parameters in 392 tensors. Cached next-token checks
  confirmed the selected adapters change the base model's predictions.
- Original moved scripts, dependencies, lockfile, and Python selection match their
  original bytes. Recreated the moved virtualenv; locked sync and Modal 1.6.1
  imports/CLI pass. Demoscener cache and existing outputs remain in place.
- Verified 52 local documentation links/anchors, Python compilation, and diff
  whitespace checks. The user's new CPU sampling runs demonstrate a 96-token
  truncation completing at 103 tokens with a larger budget, same-seed repetition,
  and a different 48-token completion from seed 2. Both completed programs pass
  Node's syntax check. Used these existing runs rather than launching duplicate
  inference while the user was experimenting.
- Independent final review found no material issues. Updated this checklist and
  the inherited hello-world command to address its two documentation cleanups.
- No training, paid GPU launch, or dependency upgrades were performed. At the
  user's subsequent request, changes are grouped into three commits: repository
  relocation, Demoscener tooling, and shared learning guides/exercises.
