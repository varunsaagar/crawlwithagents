# Toy results (CPU proxy, NOT E4B)

acc [95% CI], n per cell. Train length 64 -> eval 64..512 (8x, mirrors 8K->64K).


## (a)(c) Length sweep, base seed 0: single vs log-n temperature vs chunked

| L | k | mode | noul | choice | score |
|---|---|---|---|---|---|
| 64 | 4 | single | 0.96 [0.93,0.99] | 0.61 [0.54,0.68] | 0.36 [0.30,0.43] |
| 128 | 8 | single | 0.85 [0.81,0.90] | 0.49 [0.42,0.56] | 0.32 [0.25,0.38] |
| 128 | 8 | tau1.0 | 0.85 [0.81,0.90] | 0.48 [0.42,0.55] | 0.33 [0.27,0.40] |
| 128 | 8 | tau0.5 | 0.85 [0.81,0.90] | 0.49 [0.42,0.56] | 0.32 [0.26,0.39] |
| 128 | 8 | chunk_max | 0.53 [0.46,0.59] | 0.46 [0.39,0.52] | 0.22 [0.16,0.28] |
| 128 | 8 | chunk_logsum | 0.53 [0.46,0.59] | 0.51 [0.44,0.58] | 0.32 [0.25,0.38] |
| 256 | 16 | single | 0.76 [0.69,0.81] | 0.33 [0.27,0.40] | 0.29 [0.23,0.35] |
| 256 | 16 | tau1.0 | 0.75 [0.69,0.81] | 0.35 [0.28,0.42] | 0.29 [0.23,0.36] |
| 256 | 16 | tau0.5 | 0.75 [0.69,0.81] | 0.34 [0.28,0.41] | 0.28 [0.22,0.34] |
| 256 | 16 | chunk_max | 0.51 [0.44,0.57] | 0.34 [0.28,0.41] | 0.20 [0.15,0.26] |
| 256 | 16 | chunk_logsum | 0.51 [0.44,0.57] | 0.42 [0.35,0.49] | 0.25 [0.19,0.31] |
| 512 | 32 | single | 0.65 [0.58,0.71] | 0.24 [0.18,0.30] | 0.18 [0.13,0.24] |
| 512 | 32 | tau1.0 | 0.64 [0.57,0.71] | 0.26 [0.20,0.32] | 0.23 [0.17,0.29] |
| 512 | 32 | tau0.5 | 0.64 [0.57,0.70] | 0.26 [0.20,0.32] | 0.17 [0.11,0.22] |
| 512 | 32 | chunk_max | 0.45 [0.38,0.51] | 0.23 [0.17,0.28] | 0.23 [0.17,0.28] |
| 512 | 32 | chunk_logsum | 0.45 [0.38,0.51] | 0.26 [0.20,0.32] | 0.26 [0.20,0.32] |

## (b)(d) Training variants, single-pass length sweep (distractors k = L/16)

| model | seed | L | k | noul | choice | score |
|---|---|---|---|---|---|---|
| base | 0 | 64 | 4 | 0.96 [0.93,0.99] | 0.61 [0.54,0.68] | 0.36 [0.30,0.43] |
| base | 0 | 128 | 8 | 0.85 [0.81,0.90] | 0.49 [0.42,0.56] | 0.32 [0.25,0.38] |
| base | 0 | 256 | 16 | 0.76 [0.69,0.81] | 0.33 [0.27,0.40] | 0.29 [0.23,0.35] |
| base | 0 | 512 | 32 | 0.65 [0.58,0.71] | 0.24 [0.18,0.30] | 0.18 [0.13,0.24] |
| base | 1 | 64 | 4 | 0.95 [0.92,0.97] | 0.68 [0.63,0.73] | 0.44 [0.39,0.50] |
| base | 1 | 128 | 8 | 0.84 [0.80,0.88] | 0.45 [0.39,0.51] | 0.35 [0.30,0.40] |
| base | 1 | 256 | 16 | 0.73 [0.68,0.78] | 0.38 [0.33,0.44] | 0.29 [0.23,0.34] |
| base | 1 | 512 | 32 | 0.60 [0.55,0.66] | 0.27 [0.22,0.32] | 0.25 [0.20,0.29] |
| hard_distract | 0 | 64 | 4 | 0.52 [0.47,0.58] | 0.12 [0.09,0.16] | 0.19 [0.15,0.24] |
| hard_distract | 0 | 128 | 8 | 0.51 [0.45,0.56] | 0.14 [0.10,0.18] | 0.20 [0.15,0.24] |
| hard_distract | 0 | 256 | 16 | 0.51 [0.45,0.57] | 0.13 [0.09,0.17] | 0.20 [0.15,0.24] |
| hard_distract | 0 | 512 | 32 | 0.43 [0.37,0.48] | 0.12 [0.08,0.15] | 0.23 [0.18,0.28] |
| length_mix | 0 | 64 | 4 | 0.51 [0.45,0.56] | 0.34 [0.28,0.39] | 0.39 [0.33,0.44] |
| length_mix | 0 | 128 | 8 | 0.51 [0.45,0.57] | 0.23 [0.18,0.28] | 0.32 [0.27,0.38] |
| length_mix | 0 | 256 | 16 | 0.53 [0.48,0.59] | 0.17 [0.13,0.21] | 0.25 [0.20,0.29] |
| length_mix | 0 | 512 | 32 | 0.43 [0.38,0.49] | 0.13 [0.10,0.17] | 0.26 [0.21,0.31] |
| ssmax_trained | 0 | 64 | 4 | 0.95 [0.92,0.97] | 0.66 [0.61,0.72] | 0.39 [0.34,0.45] |
| ssmax_trained | 0 | 128 | 8 | 0.85 [0.81,0.89] | 0.49 [0.44,0.55] | 0.28 [0.23,0.33] |
| ssmax_trained | 0 | 256 | 16 | 0.75 [0.70,0.80] | 0.34 [0.29,0.39] | 0.24 [0.19,0.28] |
| ssmax_trained | 0 | 512 | 32 | 0.59 [0.53,0.64] | 0.24 [0.20,0.29] | 0.22 [0.17,0.27] |
| sandwich | 0 | 64 | 4 | 0.93 [0.91,0.96] | 0.88 [0.85,0.92] | 0.68 [0.62,0.73] |
| sandwich | 0 | 128 | 8 | 0.86 [0.82,0.90] | 0.51 [0.46,0.57] | 0.42 [0.36,0.47] |
| sandwich | 0 | 256 | 16 | 0.59 [0.53,0.65] | 0.30 [0.24,0.35] | 0.23 [0.18,0.28] |
| sandwich | 0 | 512 | 32 | 0.47 [0.41,0.53] | 0.19 [0.15,0.24] | 0.23 [0.18,0.28] |

## (d) Distractors at fixed length 64

| model | seed | k | noul | choice | score |
|---|---|---|---|---|---|
| base | 0 | 2 | 0.95 [0.93,0.98] | 0.80 [0.74,0.85] | 0.47 [0.40,0.54] |
| base | 0 | 6 | 0.92 [0.88,0.95] | 0.55 [0.48,0.61] | 0.33 [0.27,0.40] |
| base | 0 | 14 | 0.83 [0.78,0.88] | 0.36 [0.30,0.43] | 0.28 [0.21,0.34] |
| base | 0 | 30 | 0.65 [0.58,0.72] | 0.27 [0.20,0.33] | 0.22 [0.16,0.28] |
| hard_distract | 0 | 2 | 0.56 [0.50,0.61] | 0.10 [0.07,0.13] | 0.18 [0.13,0.22] |
| hard_distract | 0 | 6 | 0.50 [0.44,0.56] | 0.15 [0.11,0.19] | 0.18 [0.13,0.22] |
| hard_distract | 0 | 14 | 0.51 [0.45,0.57] | 0.14 [0.10,0.18] | 0.22 [0.18,0.27] |
| hard_distract | 0 | 30 | 0.51 [0.46,0.57] | 0.12 [0.08,0.15] | 0.23 [0.18,0.28] |

## (d) Length at fixed distractors k=4

| model | seed | L | noul | choice | score |
|---|---|---|---|---|---|
| base | 0 | 64 | 0.96 [0.93,0.99] | 0.61 [0.54,0.68] | 0.36 [0.30,0.43] |
| base | 0 | 128 | 0.92 [0.88,0.95] | 0.65 [0.58,0.71] | 0.33 [0.27,0.40] |
| base | 0 | 256 | 0.82 [0.77,0.88] | 0.56 [0.49,0.63] | 0.35 [0.29,0.42] |
| base | 0 | 512 | 0.71 [0.65,0.78] | 0.48 [0.42,0.55] | 0.32 [0.25,0.38] |
| hard_distract | 0 | 64 | 0.52 [0.47,0.58] | 0.12 [0.09,0.16] | 0.19 [0.15,0.24] |
| hard_distract | 0 | 128 | 0.53 [0.48,0.59] | 0.12 [0.08,0.15] | 0.18 [0.14,0.22] |
| hard_distract | 0 | 256 | 0.51 [0.46,0.57] | 0.11 [0.07,0.14] | 0.22 [0.18,0.27] |
| hard_distract | 0 | 512 | 0.48 [0.42,0.53] | 0.16 [0.12,0.20] | 0.21 [0.17,0.26] |

## (e) Option count, Choice, L=64, k=4 (trained with 8 options)

| model | seed | n_opt | acc [CI] | chance |
|---|---|---|---|---|
| base | 0 | 2 | 0.96 [0.93,0.99] | 0.50 |
| base | 0 | 4 | 0.82 [0.77,0.88] | 0.25 |
| base | 0 | 8 | 0.61 [0.54,0.68] | 0.12 |
| base | 0 | 16 | 0.42 [0.36,0.49] | 0.06 |
| base | 0 | 31 | 0.18 [0.13,0.23] | 0.03 |

---

# E0 — breakdown on the real eval (2026-10-11). NOT a toy: `kairos-e4b-L1c-best`, 13,303 held-out question-results

Source: `/ckpt/datasets/kairos-eval64k-v1/results/control-L1c-init.jsonl`. Cost: 0 GPU, re-slices saved
predictions. **Three of the strategy doc's four testable hypotheses are refuted, and the one that
survives changes the plan.**

Data hygiene first: all three result files carry exactly **13,303 valid rows**. The control file also
holds **9 error rows** (CUDA OOM) left over from an aborted first attempt; `report_eval.py` already
excludes them, so earlier comparisons were sound. Domain is **100% `voice`**, so **H3 (accuracy by
source) cannot be run at all** — the eval set never got its churn and telecom adapters.

## H5 — CONFIRMED, and it is the headline. The model ranks well and selects badly.

| length | top-1 | top-2 | **top-5** | n |
|---|---:|---:|---:|---:|
| 8,192 | 0.311 | 0.547 | **0.884** | 1,297 |
| 65,536 | 0.225 | 0.480 | **0.851** | 1,981 |

**The gold option is in the top 5 about 88% of the time at 8K and 85% at 64K**, against a top-1 of
0.31 and 0.23. The evidence is being read. What fails is the final selection.

This reframes the problem: **the gap is ranking→selection, not long-context reading.** It also
explains why long-context interventions keep coming back null — they target a faculty that is not
the bottleneck. It promotes **E4 (retrieve-then-pointer)** from "wide-Choice fix" to the main event,
and it means a reranking or two-stage head could in principle recover most of a 0.57-point gap.

## H4 — CONFIRMED and worse than stated. Noul is a constant predictor.

| length | recall(yes) | recall(no) | asymmetry | balanced acc |
|---|---:|---:|---:|---:|
| 8,192 | **0.016** (n=123) | **0.998** (n=455) | −0.982 | **0.507** |
| 65,536 | 0.297 (n=222) | 0.776 (n=695) | −0.478 | **0.537** |

**At 8K the model answers "no" to essentially everything.** Gold is 79% "no" (455/578), so the
reported **noul accuracy of 0.785 is precisely the majority-class baseline** (455/578 = 0.787).
**Balanced accuracy is at chance at both lengths.** Every noul number we have reported — including
the "sign-flip fixed" claim — has been measuring class balance, not capability.

Note the direction: the model is **less** degenerate at 64K than at 8K. Length is not what makes
noul fail. **E1 (bias + temperature per type × length bucket) is now the highest-value cheap rung**,
and its decision rule should be **balanced accuracy**, never raw accuracy.

## H1 — REFUTED. Wide Choice is not the driver.

| options | n | acc [95% CI] | chance | lift |
|---|---:|---|---:|---:|
| 3–8 | 4,959 | 0.287 [0.274, 0.299] | 0.231 | +0.056 |
| 9–26 | 1,783 | 0.176 [0.158, 0.193] | 0.075 | +0.101 |
| 27–64 | 46 | **0.000** | 0.026 | −0.026 |
| 65–254 | 88 | **0.000** | 0.008 | −0.008 |

Wide Choice is genuinely broken — **0 correct out of 134** — but it is **1% of the eval** and can cost
at most ~1 point of the pooled 0.41. *(0 of 134 is not impossible by chance, expected ≈2, p≈0.13, so
this is suggestive of a defect rather than proof of one — worth a look, not an alarm.)*

The real weight sits in the 3–8 and 9–26 buckets: 6,742 questions at lift +0.056 and +0.101 over
chance. **That is where the pooled accuracy is lost, and it is the ordinary case, not the exotic one.**

## H2 — REFUTED. Score is wrong, not nearly-right.

n=3,333 · exact **0.291** · within-1 **0.383** · **MAE 2.04**

Relaxing to ±1 adds only 9 points, and the mean error is two whole levels. The hypothesis was that
exact-match scoring understates a nearly-right model; it does not. Score is genuinely failing.

## What E0 changes in the ladder

1. **E1 moves to first** and must be judged on **balanced accuracy** — raw noul accuracy is a
   class-balance artifact and has been misleading every report so far.
2. **E4 (retrieve-then-pointer) is promoted** from a wide-Choice patch to the primary lever, on the
   strength of top-5 0.88 vs top-1 0.31. It should be run over **all** option counts, not just >26.
3. **H3 is unrunnable until the eval set gets its churn and telecom slices.** Any claim about source
   quality is currently unsupported.
4. **The long-context rungs (E2, E5, E6, E7, E8) drop in priority.** The 64K penalty is 3.8 pp;
   the selection gap is ~57 pp. We have been optimising the smaller term.

---

# E1 — Noul bias + threshold per length bucket (2026-10-11). `kairos-e4b-L1c-best`, 0 GPU

Fit on a 30% calibration split, evaluated on the 70% held-out remainder, per length bucket.
**Verdict: FAILS its pre-registered rule, but for a reason that invalidates the rule, and it
uncovers a large unanticipated gain at 8K.**

## The diagnostic that should have come first: does noul carry signal at all?

AUROC of p(yes) against gold, by length:

| length | n | positives | **AUROC** | reading |
|---:|---:|---:|---:|---|
| 8,192 | 578 | 123 (21%) | **0.7205** | real signal — **biased**, fixable |
| 16,384 | 682 | 143 (21%) | 0.6095 | weak signal |
| 32,768 | 917 | 222 (24%) | **0.4812** | **chance — blind** |
| 65,536 | 917 | 222 (24%) | **0.5279** | **chance — blind** |

**H4 as written ("Noul at 64K is biased, not blind") is wrong. At 64K it is blind.** There is no
ranking to re-threshold. A bias/temperature fix cannot work there, and that was knowable from a
single AUROC before fitting anything.

## The fit

| length | before bal-acc | after bal-acc | Δ | threshold | recall yes / no (after) |
|---:|---:|---:|---:|---:|---|
| 8,192 | 0.500 | **0.690** | **+0.190** | 0.16 | 0.56 / 0.82 |
| 16,384 | 0.483 | 0.585 | +0.102 | 0.29 | 0.62 / 0.55 |
| 32,768 | 0.495 | 0.501 | +0.006 | 0.63 | 0.03 / 0.98 |
| 65,536 | 0.564 | 0.530 | −0.035 | 0.56 | 0.16 / 0.89 |

**The gain tracks AUROC exactly.** Where there is signal the threshold fix recovers a lot; where
there is none it does nothing or slightly harms.

## Verdict against the pre-registered rule

Rule: *ship if balanced accuracy +≥5 pp **at 64K** and ECE ≤0.05 per bucket with no 8K loss.*

- **64K: −0.035. FAILS.**
- **ECE: fails everywhere**, but see the caveat below — the ECE figures are mine and they are not sound.

**Do not quietly rewrite the rule to make this pass.** The honest statement is: the rule encoded a
hypothesis that AUROC refuted before the fit ran, so it tested the wrong thing. The +0.190 at 8K is a
**separate, unanticipated finding** and needs its own pre-registered confirmation (second seed, and
the paired test on the full 1,552), not a retrofitted rule.

## Caveat on my own numbers

**The post-fit ECE (0.286, 0.281) is not trustworthy.** I moved the operating point by a linear
rescale of the reported probability rather than fitting a temperature, which inflates ECE by
construction. The ECE half of the decision rule cannot be judged until a proper Platt/temperature fit
is run. The balanced-accuracy half is sound — it depends only on the threshold, not the rescaling.

## What this means

1. **Noul is two different problems at two different lengths.** ≤16K it is a miscalibrated threshold
   on a working ranker. ≥32K the ranker itself is gone. One fix cannot serve both, and reporting a
   single "noul accuracy" hides the distinction completely.
2. **A threshold fix at 8–16K is cheap and large** (+0.190 / +0.102 balanced accuracy, zero GPU). It
   should be confirmed and shipped on its own terms.
3. **Every pooled noul number reported so far is a class-balance artifact** (gold is 79% "no";
   predicting "no" always scores 0.787). Balanced accuracy must replace raw accuracy for this type.
4. **E1 does not rescue 64K.** Combined with E0's top-5 result, the evidence keeps pointing the same
   way: the 64K deficit is small and the *selection* deficit is large.

---

# E9b (part 1) — what the Qwen deciders actually support out of the box (2026-10-11)

Verified from each model's own `config.json`, `training/config.json` and model card on 2026-10-11.
**Zero GPU.** The scoring half of E9b is blocked on feasibility — see the bottom of this section.

## The claim every one of them makes, and why it is worthless

| model | `max_position_embeddings` | **actually trained at** | published long-context eval |
|---|---:|---:|---|
| `perplexity-ai/pplx-decider-v1.1-27b` | 262,144 | **8,192** | **none** |
| `Mapika/decider-4b` | 262,144 | unverified | none |
| `apus-ailab/APUS-OpenJev-v1-4B` | 262,144 | unverified | none |
| `jaredpalmer/kev-27b` | 262,144 | **32,768** | **yes — see below** |

**All four advertise 262,144. That number is the inherited Qwen base config, not a capability.** It is
the identical trap as StartLux's "262,144 native", which we already established has zero long-context
evaluation behind it and a longest measured prompt of 2,515 tokens.

**pplx-decider-v1.1-27b, from its own `training/config.json`:**
`max_length 8192 · token_budget 8192 · epochs 1 · lr 2e-06 · effective_batch_size 256`

So **pplx supports 8K out of the box, not 64K** — and its `prepare()` *rejects* over-length inputs
rather than truncating, so it will refuse a 64K state outright. Reaching 64K with it is the full
section-4 recipe: ~150–400 GPU-h.

## The exception, and it matters: `kev-27b` has a real 64K validation

From its model card, measured not claimed:

> Training states were at most **32,768** tokens. Validated context length: **65,536** tokens.
> Paired CUAD accuracy difference from the 8k bucket, pp [95% CI]:
> **16k +0.2 [−0.7, +1.2] · 32k −0.2 [−1.2, +0.7] · 64k −1.1 [−2.4, +0.0]**, 445–447 questions each.

**This is the only decider in the landscape with a properly measured 64K result**, and it is paired,
per-bucket, with CIs, on the same contracts. At 64K it is **−1.1 pp from its own 8K**. Our L1c is
**−3.8 pp** on the same style of paired test.

**So the direct answer to "can training on top of Qwen fix this": yes, demonstrably — someone already
has.** Kev-27b reached validated 64K by training at **32,768** on 145,840 records / 337,130 questions,
as a rank-16 LoRA on Qwen3.8-27B. Apache-2.0, so usable.

Note what that recipe is *not*: it is not training at 64K, and it is not a position-encoding change.
It is **2× the eval length, at scale, with 337K questions**. That is consistent with everything our
own controls found.

## Why the scoring half has not run

| blocker | detail |
|---|---|
| **The script does not exist** | the strategy doc cites `qwen-decider-64k/eval_pplx_length_sweep.py` as "syntax-checked, dry-run tested". It is **not in the repo** |
| **PVC space** | `/ckpt` is 300 G, **239 G used, 62 G free (80%)**. kev-27b is **51.3 GB** and pplx **52.2 GB** — either takes a shared PVC to ~98% |
| **No pod egress** | the pod cannot reach huggingface.co (HTTP 000); only the laptop can (HTTP 200), so every byte goes laptop → `oc cp` |
| **Schema mismatch** | each needs its own adapter: pplx has a bespoke `DecisionModel` + `[255, 5120]` readout with last-token pooling and **noncausal** full attention; `Mapika/decider-4b` is `Qwen3_5ForCausalLM`; APUS is `Qwen3_5ForConditionalGeneration`. Our `run_eval.py` expects a Kairos checkpoint |

The 4B candidates (**8.4 GB** and **9.1 GB**, both Apache-2.0) fit the PVC comfortably and are the
cheap way to test "does a Qwen base beat E4B at matched size". Neither has any long-context
validation, so they answer the *base-quality* question, not the *64K* question.

**Recommendation:** the 64K question is already answered by kev-27b's published measurement; what we
would be buying by scoring it ourselves is confirmation on *our* eval and *our* question types. That
is worth doing, but it needs ~51 GB of PVC headroom freed first, and a written adapter — not a quick run.

---

# E4a — option-position bias, and a CORRECTION to E0's headline (2026-10-11). 0 GPU

## Clean negative: there is no option-position bias

4-option questions (the modal case), n=3,367:

| idx | predicted | gold | mean p |
|---:|---:|---:|---:|
| 0 | 24.8% | 26.6% | 0.2488 |
| 1 | 23.9% | 23.2% | 0.2489 |
| 2 | 25.1% | 26.0% | 0.2515 |
| 3 | 26.2% | 24.2% | 0.2508 |

Accuracy by gold position: 0.300 / 0.269 / 0.297 / 0.331 — no trend. Debiasing by subtracting the
per-position mean log-prob and re-argmaxing: **−0.002**, no help.

**The ChoiceHead's permutation-equivariance works as designed.** That is a positive confirmation of
the architecture and it closes off a cheap fix. Position is not the problem.

## CORRECTION: E0's "top-5 = 0.88" was inflated and I should not have reported it as I did

**The median choice question has 4 options. 49.6% have exactly four.** For those, top-5 is trivially
1.000 — it includes every option. The pooled 0.884 was mostly counting that.

Restricted to questions where top-5 is a real test:

| subset | n | top-1 | top-5 | top-5 chance | **lift over chance** |
|---|---:|---:|---:|---:|---:|
| >5 options | 2,577 | 0.186 | 0.625 | 0.474 | **+0.151** |
| ≥10 options | 1,588 | 0.168 | 0.538 | 0.339 | **+0.200** |
| ≥16 options | 778 | 0.136 | 0.459 | 0.257 | **+0.202** |

**The ranking carries modest signal — +15 to +20 pp over chance — not the near-perfect signal I
described.** "The evidence is being read, only selection fails" was too strong. A reranker (E4) is
still worth running, but its ceiling is far below the 57 pp I implied.

## And this reverses the ladder advice I gave after E0

Same subset, split by length:

| ≥10 options | top-1 | top-5 |
|---|---:|---:|
| 8,192 | **0.295** | 0.656 |
| 65,536 | **0.102** | 0.515 |

**A 19-point length penalty on wide-choice questions**, against the 3.8 pp pooled figure. I wrote
after E0 that "the long-context rungs drop in priority; the 64K penalty is 3.8 pp and the selection
gap is ~57 pp." **That is true pooled and false where the difficulty lives.** Length is the dominant
effect precisely on the questions the model is worst at.

## Why I got this wrong twice in one day

Both E0's top-5 claim and the earlier "training is improving" claim failed the same way: **I
aggregated over a heterogeneous mixture and read the mixture, not the effect.** With the loss it was
the hard/easy batch ratio; here it is the option-count distribution. Lesson 6 said "when a loss is
visibly bimodal, split the regimes first" — the general form is stronger and belongs in the list:

**Before reporting any aggregate, check what the aggregate is averaging over.** If the population is
heterogeneous in a way that interacts with the metric — option count against top-k, batch difficulty
against loss, class balance against accuracy — the aggregate measures the population, not the model.
Three separate findings today (noul's majority-class 0.785, the bimodal loss, this top-5) were all
this same error.

## Revised read of the ladder

- **E4 (retrieve-then-pointer) stays worth running** but should be scoped to ≥10-option questions
  (n=1,588) where there is +0.200 of headroom, and judged on that subset.
- **The long-context rungs are back in play**, aimed specifically at wide-choice at length, where the
  penalty is 19 pp rather than 3.8.
- **E1's 8K noul fix (+0.190 balanced accuracy) remains the single largest verified free win.**

---

# E0/H8 — the Gemini teacher ceiling. THE MOST IMPORTANT RESULT SO FAR (2026-10-11)

`gemini-3.7-flash` via `kairos-gateway-test.ebiz.verizon.com/v1`, 200 randomly sampled **8K** states
from the held-out eval, choice questions with 3–26 options, same state text and same option list the
model sees. TLS verification on throughout; key passed in-process, never written to disk.

| | n | accuracy [95% CI] |
|---|---:|---|
| **Gemini-3.7-flash, all** | **200** | **0.935 [0.901, 0.969]** |
| L1c on the same band | — | **~0.31** |
| gold = a real answer | 130 | **1.000** |
| gold = the null option ("no session with that identifier") | 70 | 0.814 [0.723, 0.905] |
| 3–8 options | 97 | 0.938 |
| 9–15 options | 57 | 0.912 |
| **16+ options** | 46 | **0.957** |

0 errors, 0 failures.

## What this settles

The hypothesis was pessimistic: *"if a frontier teacher can't reach 0.9 on your 8K states, the
student won't either, and the work is data cleaning, not modelling."*

**It reached 0.935. Every pessimistic reading is refuted:**

1. **The labels are good.** 100% agreement with gold on the 130 real-answer items. H3's worry about
   noisy sources is unsupported — at least for the voice domain, which is the whole eval.
2. **The task is solvable.** The ceiling is ~0.94, not ~0.5. There is ~62 points of headroom.
3. **Wide choice is not intrinsically hard.** Gemini is *flat* across option count and in fact
   scores highest at 16+ (0.957). The steep option-count decay in E0 is a property of **our model**,
   not of the questions. This contradicts the framing of H1/E4 as "wide Choice is a ranking problem" —
   for a capable model it is not a problem at all.
4. **Abstention is the genuinely hard part, even for Gemini** — 0.814 on null-option items against
   1.000 on real answers. Our own noul results point the same way from a much lower base.

## The caveat that stops this being a pure capability gap

**Gemini spent ~400 output tokens per item** (measured in the smoke: 497, 169, 158, 448, 501). It is
reasoning before answering. Our pointer head does **one forward pass with no scratchpad**. So some
unknown share of the 62-point gap is an *inference-paradigm* difference rather than a representational
one, and that share will not transfer through label distillation. Quantifying it would need Gemini
constrained to answer immediately — worth doing before sizing any distillation effort.

## How this reorders the ladder

- **Data cleaning drops off.** The labels are fine.
- **Teacher distillation (E5's Gemini-labelled data, and teacher-distribution targets) moves to the
  top.** A teacher at 0.935 against a student at 0.31 is an enormous gradient, and we now know the
  teacher is reliable on exactly these items.
- **E4 (retrieve-then-pointer) weakens as a theory.** Gemini shows wide choice is easy when the model
  is good; the reranker would be patching a symptom.
- **The realistic target moves up.** Section 1 of the strategy doc guessed 0.75–0.85 for choice at 8K.
  The measured ceiling is 0.94+, so that target was set too low.

## A near-miss worth recording

The first smoke run scored **0/5** and every answer was truncated (`''`, `'execute'`,
`'this bundle contains'`). Cause: `max_tokens: 64`, consumed entirely by reasoning tokens. Raising it
to 512 gave **5/5** on the identical items.

**Had this been scaled to 200 without the smoke, it would have reported "Gemini scores ~0 on our
states" and concluded the labels were impossible — inverting the single most important finding in
the project.** The generalised lesson, which is now the third variant of the same mistake: *an
output-shaping parameter can masquerade as a capability measurement.* Always inspect a few raw
responses before trusting an aggregate.

---

# H8b — is the teacher's advantage reasoning, or representation? (2026-10-11). 0 GPU

The one caveat on H8 was that Gemini spent ~400 output tokens reasoning while our pointer head does a
single forward pass, so some of the 62-point gap might be test-time compute that distillation cannot
transfer. **Measured, on the identical 200 items.**

Suppressing reasoning: `reasoning_effort: "none"` works (output tokens on 3 probe items went
`[502, 282, 155]` → `[498, 9, 6]`). **A prompt instruction alone does not** — "answer immediately, no
reasoning" gave `[502, 241, 88]`, i.e. it kept thinking. Worth knowing: the API parameter is the lever,
the prompt is not.

| condition | accuracy [95% CI] | median output tokens |
|---|---|---:|
| Gemini **with** reasoning | 0.935 [0.901, 0.969] | ~400 |
| **Gemini without reasoning** | **0.945 [0.913, 0.977]** | 217 |
| L1c (single forward pass) | ~0.31 | 1 |

By gold type, without reasoning: real answer **1.000** (n=130), null option 0.843 (n=70) — the same
shape as with reasoning.

## Conclusion

**Removing the scratchpad did not cost anything.** Accuracy was marginally *higher* and the intervals
overlap almost entirely, so the two are indistinguishable. **The 62-point gap is representational, not
test-time compute.**

This matters because it is the difference between two very different projects:
- *If* the gap had been reasoning, label distillation would transfer little and the answer would be
  an architecture change (generate-then-decide), which is a rewrite.
- **Because the gap is representational, a single-forward-pass model can in principle reach ~0.94 on
  these items**, and distilling Gemini's labels and distributions is the direct route.

Caveat kept honest: median output was still 217 tokens with `reasoning_effort: "none"`, so reasoning
was reduced rather than eliminated. The inference is from the *null effect of halving it*, not from
proving it reached zero. A stricter test would cap `max_tokens` just above the longest option string.

## Standing recommendation after E0, E1, E4a, H8, H8b

| finding | status |
|---|---|
| **Teacher ceiling 0.945 on our own 8K states, one-pass** | the task is solvable and the labels are good |
| **Gap is representational** | distillation is the right instrument |
| Noul threshold fix at 8K, **+0.190 balanced accuracy** | largest verified free win, ship it |
| Wide choice is easy for a capable model (Gemini 0.957 at 16+) | E4 reranking is patching a symptom |
| Position bias | ruled out |
| Length penalty on wide choice, 8K 0.295 → 64K 0.102 | real, and larger than the 3.8 pp pooled figure |

**The ladder's premise was that this is a long-context problem. Five zero-GPU experiments now say it
is mostly a capability gap at every length, with a smaller genuine long-context component on hard
questions.** E5 (Gemini-labelled distillation) should be the next real spend, ahead of the Qwen
controls and well ahead of SSMax/sandwich/reranking.

---

# H3 (finally testable) — training-label quality per source (2026-10-11). 0 GPU

H3 ("some sources are noisy or near impossible") could not be run on the eval, which is 100% voice.
But the **training** set has all three sources, and the churn builder flagged its own labels as
*"distilled from the incumbent production scorer, not human gold"*. So the hypothesis was live and
untested on the data that actually shapes the model.

Same instrument as H8: `gemini-3.7-flash`, `reasoning_effort: none`, 70 sampled 8–16K training rows
per source, choice questions with 3–26 options.

| source | Gemini-vs-gold agreement [95% CI] | n |
|---|---|---:|
| voice | 0.971 [0.932, 1.000] | 70 |
| telecom | 0.871 [0.793, 0.950] | 70 |
| **churn** | **1.000 [1.000, 1.000]** | 70 |
| **all sources** | **0.948 [0.917, 0.978]** | 210 |
| *(eval set, same instrument)* | *0.945 [0.913, 0.977]* | *200* |

0 errors.

**H3 is refuted on both sides. The labels are good everywhere.** Training-label quality (0.948) is
indistinguishable from eval-label quality (0.945). Churn — the source I explicitly flagged as suspect
— is **70/70**. Telecom is the weakest at 0.871 and is the only one worth a second look, but even its
lower bound is 0.793.

## What is now eliminated

Across E0, E1, E4a, H8, H8b and H3, every "the data or the measurement is broken" explanation has been
tested and rejected:

| explanation | verdict |
|---|---|
| wide Choice drags the mean (H1) | **refuted** — 1% of the eval |
| Score is scored too strictly (H2) | **refuted** — within-1 only 0.383, MAE 2.04 |
| noisy or impossible sources (H3) | **refuted** — 0.948 training, 0.945 eval |
| noul is biased not blind (H4) | **true ≤16K, false ≥32K** — blind at length |
| the model ranks well, selects badly (H5) | **weakened** — +0.15–0.20 lift, not near-perfect |
| option-position bias (E4a) | **refuted** — flat to 3 decimals |
| the task is unsolvable / teacher ceiling (H8) | **refuted** — teacher 0.945 |
| the teacher only wins via reasoning (H8b) | **refuted** — 0.945 without reasoning |

**Nothing about the data, the labels, the metric or the measurement explains a 0.31 vs 0.945 gap.**
What remains is the model itself: capacity, the pointer-head readout, or the training objective.

## This makes the bigger-decider control the decisive experiment

E9b/E10 are no longer "nice controls" — they are the only remaining family of explanation. A matched-size
Qwen decider (`Mapika/decider-4b`, 8.4 GB) separates **base quality** from **our recipe**; a 27B
(`kev-27b` or `pplx-decider-v1.1-27b`, ~51 GB) separates **capacity** from everything else.

- if a 4B Qwen decider also scores ~0.31 → the base is not the issue; our head or objective is
- if it scores much higher → our recipe is leaving most of the model on the table
- if only the 27B scores high → it is capacity, and the 4B tier cannot reach 0.9 whatever we do

---

# E9b/E10 — the capacity ladder. THE DECISIVE RESULT (2026-10-11). 0 GPU, no downloads

Run on the gateway against the **same 200 8K eval items**, same prompt, same scoring as H8.
No Qwen decider was needed: the gateway already serves a Gemma-4 ladder.

| model | accuracy [95% CI] | null items | real items |
|---|---:|---:|---:|
| **kairos L1c E4B** — pointer head, **trained on this task** | **~0.31** | — | — |
| **bedrock-gemma-4-e2b** — **smaller**, generic, **zero-shot** | **0.500 [0.431, 0.569]** | 4/70 | 96/130 |
| **bedrock-gemma-4-26b-a4b** | **0.900 [0.858, 0.942]** | **70/70** | 110/130 |
| gemini-3.7-flash (no reasoning) | 0.945 [0.913, 0.977] | 59/70 | 130/130 |

200 items each, 0 errors, 0 skips.

## Finding 1: our recipe is worse than not training at all

**Gemma-4 E2B — a model from our own family, SMALLER than our E4B, with no task-specific training,
simply prompted — scores 0.500 against our trained pointer head's 0.31.**

We are **19 points behind a smaller untrained model on our own task.** Whatever the Kairos
head/objective/packing is doing, it is currently *subtracting* capability rather than adding it.
No long-context technique on the ladder addresses this, and it should be the top investigation.

Caveat, stated fairly: this is not perfectly apples-to-apples. The gateway models generate free text
while our pointer head picks from a constrained option set in one forward pass. But the constraint
should be an **advantage** to us — we cannot produce an invalid answer, and they can. The comparison
understates our deficit if anything.

## Finding 2: the 0.9 target requires ~26B, and is out of reach at 4B

The capacity curve is steep and clean: **E2B 0.500 → 26B 0.900 → Gemini 0.945.** Our E4B sits between
E2B and 26B, so a well-executed 4B-class model should land near 0.55–0.65 prompted. **0.9 is not a
4B number.** Section 1 of the strategy doc hoped for 0.85–0.90 at 8K from the current tier; that is
not supported by this ladder.

**`gemma-4-26b-a4b` is already deployed in-cluster** (`vllm-gemma-4-26b-a4b-it-fp8-dynamic`), so the
26B tier needs no procurement.

## Finding 3: abstention is capacity-gated, which explains our noul pathology

Null-item accuracy ("this bundle contains no session with that identifier"):
**E2B 4/70 (0.06) → 26B 70/70 (1.00) → Gemini 59/70 (0.84).**

E2B almost never abstains; it guesses a real answer. **This is precisely the failure we measured in
our own model** — noul as a constant predictor, AUROC at chance beyond 16K, and a model that answers
"no" to everything at 8K. We treated it as a calibration bug and found a +0.190 threshold fix (E1),
but the ladder shows the root cause is **capacity**: knowing that the evidence is *absent* is harder
than finding it when present, and small models cannot do it.

## A fourth infrastructure artifact caught before it became a finding

`gemma-3-4b` initially scored **0.005**. Raw inspection: `finish_reason: content_filter`, 0 tokens in
and out, *"Request blocked: the input or generated output violates company AI safety and data
governance guardrails."* **The entire `gemma-3-*` family is content-filtered on our telco transcripts;
the `bedrock-gemma-4-*` family is not, and gemini-3.7-flash processed the identical content 400 times
without a single block.** The filter is model-specific, not content-specific.

Had that 0.005 been reported, it would have read as "a 4B model cannot do this task at all" — the
exact opposite of the truth (0.500). That is the **fourth** time today an infrastructure parameter
nearly masqueraded as a capability measurement, after `max_tokens`, partial-eval prefixes, and
option-count mix. **Inspect raw responses before trusting any aggregate.**

Also note for any teacher-labelling plan: `gemma-4-26b` (non-bedrock) returns `AuthenticationError`
for this key, and the `gemma-3-*` family is unusable on this content. Use `bedrock-gemma-4-26b-a4b`
or `gemini-3.7-flash`.

## What this does to the ladder

**The premise is wrong.** This was scoped as a long-context problem. The measurements say:

1. **A 4B-class model cannot reach 0.9** on these questions — measured ceiling ~0.5 at E2B, 0.9 at 26B.
2. **Our trained 4B is 19 points below an untrained smaller sibling**, so the recipe is the immediate
   bug, independent of length.
3. **Abstention needs capacity**, so the noul work is palliative.

Recommended reprioritisation:
- **Stop the long-context rungs (E2, E3, E5, E6, E7, E8).** They optimise a 3.8 pp term on a model
  that is 60 points from the ceiling for reasons unrelated to length.
- **New top priority: why is our trained E4B worse than a prompted E2B?** Compare like-for-like —
  prompt our own base E4B through the gateway path, and score the Kairos head against it.
- **Then: move the tier to 26B**, already deployed, measured at 0.900.

---

# E9b (exact) — correcting my own comparison, and the conclusion hardens (2026-10-11)

The previous entry compared the gateway models against "L1c ~0.31". **That figure came from the full
eval's 8K choice band (n=1,297), not from the 200 items the gateway models actually saw.** Comparing
different item sets is exactly the sloppiness this log exists to catch, so I re-derived the sample
deterministically with `item_id` and looked up L1c's saved prediction for each.

**200 of 200 matched.**

| model | accuracy on the **identical 200 items** [95% CI] |
|---|---|
| **kairos L1c E4B** — trained, pointer head | **0.215 [0.158, 0.272]** (43/200) |
| **bedrock-gemma-4-e2b** — smaller, zero-shot, prompted | **0.500 [0.431, 0.569]** |
| bedrock-gemma-4-26b-a4b | 0.900 [0.858, 0.942] |
| gemini-3.7-flash (no reasoning) | 0.945 [0.913, 0.977] |

**The gap is 28.5 points, not 19, and the intervals are nowhere near overlapping.** These 200 are
harder than the band average (one question per state, the first qualifying choice question), which is
why L1c scores 0.215 here against 0.31 across the whole band — but every model faced the same items,
so the comparison is now exact.

## The conclusion no longer needs another experiment

**Gemma-4 E2B is smaller than Gemma-4 E4B.** A smaller, untrained, generically-prompted model beats
our larger, task-trained, specialised one by 28.5 points on its own task.

Capacity cannot explain that — it points the wrong way. The deficit therefore lies in **the Kairos
training recipe, the pointer-head readout, or the packing**, and it is worth roughly a third of the
achievable score. Nothing on the long-context ladder touches it.

## Next diagnostic, to split the remaining two explanations

The 28.5 points sit in one of two places, and one experiment separates them:

| run | tells us |
|---|---|
| **base Gemma-4 E4B, prompted generatively** (no Kairos head, no fine-tune) | if ≈0.55–0.65 → the **Kairos pipeline** is the cost; if ≈0.22 → something else |
| **L1c weights, prompted generatively** (fine-tuned weights, no pointer head) | if ≈0.5 → the **head/packing** is the cost, weights are fine; if ≈0.22 → the **fine-tune damaged the weights** |

Both need a GPU and a generative decode path; neither is on the gateway (no `e4b` there). Expect
~1–2 h of engineering plus a short run. **This is now the highest-value experiment in the project** —
it is the difference between "fix the head" and "stop fine-tuning the way we do".

## Lessons list — updated

Adding the one this entry demonstrates:

**11. Never compare models across different item sets.** I reported a 19-point gap from "L1c ~0.31"
measured on a 1,297-item band against gateway scores on a 200-item sample. The true paired gap is
28.5 points. The error was in the conservative direction this time; it will not always be. **Match on
item id, and say how many matched.**

---

# S0 — native decision architecture on Gemma-4-26B-A4B: PASSES (2026-10-11)

Plan: `PLAN-26B-NATIVE-DECISION-64K.md`. Pod `kairos-train26b`, 4x H100, bf16 weights from the
in-cluster PVC `gemma-4-26b-bf16-mtp4-weights` mounted read-only at `/base`.

**No download, no gated HF, no PVC pressure.** The bf16 weights already existed in-cluster at
`/drafter/gemma-4-26b-a4b-it` (49 GB, unquantized) on a 150Gi **ReadWriteMany** PVC, so it mounts
alongside the serving pod. The only other copy is fp8-dynamic (27 GB), which is unusable for training.

```
text stack: Gemma4TextModel  hidden=2816            loaded in 2s
params 25.30B total   63.80M trainable (0.252%)
expert tensors 90, 22.84B params, trainable 0   ->  FROZEN OK
trainable tensors: lora 410 · norms 271 · heads 41 · other 0
ChoiceHead @255 options: OK   permutation max|err| 0.000e+00  ->  PASS
lora targets: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

| property we must keep | verified by |
|---|---|
| permutation-equivariant head | **max abs err 0.000e+00** at K=255 — exactly zero |
| up to 255 options | head constructed and scored at K=255 |
| cheap bilinear readout | `einsum(a_proj(answer), c_proj(choices))/sqrt(dim)` — O(options) |
| one forward pass, no decoding | no LM head in the graph; logits come from the readout |
| cannot emit an invalid answer | output is `masked_fill(~choice_mask, -inf)` over the option set |
| MoE does not break it | 22.84B of expert parameters **frozen for free** — `experts` is an `nn.Parameter`, peft cannot target it |

## Port issue found and solved without touching shared source

`network.build()` cannot load this checkpoint. `AutoConfig` returns a `Gemma4Config` whose
**top-level `hidden_size` is `None`** — the real settings live in the nested `Gemma4TextConfig`
(hidden 2816, `model_type: gemma4_text`). `AutoModel` maps `gemma4` to `Gemma4Model`, which builds
from the top level and dies in the embedding:
`empty(): argument 'size' ... got NoneType`. build()'s existing
`getattr(base, "language_model", base)` never gets a chance to run.

Fix used: load through `AutoModelForCausalLM` (which understands the full multimodal checkpoint),
take the text stack, and call the **public `wrap()`**. Script: `s0_wrap26b.py`.

**This is a real defect in `build()` that will hit anyone loading a multimodal Gemma checkpoint**, and
it blocks `kairos.model.train` too, since the trainer calls `build()`. Two ways forward: patch
`build()` to fall back to `text_config` when top-level `hidden_size` is None (a ~4-line change to
shared PVC source, needs authorisation), or stage a text-only copy of the weights that `AutoModel`
can load directly. Taking the second route for now — it leaves shared source alone.

## Two self-inflicted errors in the S0 script, worth recording

1. Outer `bash -lc '...'` with a heredoc containing **single quotes** inside the Python terminated
   the shell string. Write the script locally and `oc cp` it rather than inlining.
2. I guessed `ChoiceHead.forward(task, choices, answer)` twice. The real signature is
   `forward(choices, answer, choice_mask, task)` (`network.py:71`). **Read the signature; do not
   infer it from the class docstring.**

---

# GPU capacity audit (2026-10-11 14:10 UTC) — what is actually free

Asked to reclaim GPUs from `judge-train`, `kairos-chain`, `kairos-ctx`, `kairos-per-hub`,
`vllm-muse-glimmer-30b`. Measured utilisation on **every** GPU-holding pod rather than trusting the list.

| category | GPUs | evidence |
|---|---:|---|
| **genuinely computing** | **7** | `judge-train-a` 4× **100%** / 73 live procs · `judge-train-c` 2× **100%** / 21 · `judge-eval` 1× **100%** / 11 |
| **0% util but model-resident** | 10 | djev 72 GB · rps-route-a 74 GB · vllm-gemma-4-26b ×2 78 GB · kev-gemma 45 GB · kairos-jev 39 GB · vzgpt-e4b ×2 39 GB · service-one 37 GB · laya 4 GB |
| **truly free** (0% **and** 0 MiB) | 7 | `kairos-train26b` (5, ours) + `kairos-model64k` + `kairos-eval` |

**The 0% readings are misleading and that matters.** Ten GPUs read 0% utilisation while holding
37–78 GB of resident weights — these are serving pods between requests, not free capacity. Only pods
at **0% *and* 0 MiB *and* 0 live processes** are reclaimable.

**Already reclaimed in earlier ticks:** `kairos-chain` (deleted, was Failed/exit 137), `kairos-ctx`
(never existed), `kairos-per-hub` (scaled 2→1→0; its 15 GB of weights live on the persistent
`/ckpt` PVC so nothing was lost, and its last 200 log lines were 100% `/health` — no real traffic),
`vllm-muse-glimmer-30b` (scaled to 0, verified 0% and no requests). **Their GPUs are already in the
pool** — the list of "available" GPUs was mostly already harvested.

**`judge-train` is the only untapped block, and it is not idle.** Three jobs, 7 GPUs, 100%
utilisation, live processes at 99–179% CPU, 7+ days in. Taking them destroys a week of someone's
compute. *(They also carry hundreds of zombie child processes — a real process leak worth telling
their owner about, but the parents are alive and working.)*

**A quota detail that cost a cycle:** the 7-GPU pod was first rejected on **CPU**, not GPU —
`requests.cpu=30` against 99.7 of 128 cores already used. GPU quota was not the binding constraint.

## Lesson 12

**0% GPU utilisation does not mean a GPU is free.** A serving pod between requests reads 0% while
holding tens of GB of weights; killing it breaks serving and the memory has to be re-paged on
restart. The reclaimable test is **0% utilisation AND ~0 MiB resident AND no live (non-zombie)
process** — all three.

---

# S1 — end-to-end trial on the 26B, and what it saved us from (2026-10-11)

Instructed not to start the 20-hour run until the architecture was proven. That was the right call:
the trial found three blockers, and **killed a workaround I had recommended an hour earlier**.

## The architecture is native, not a mimic — verified against the reference

`bev-train/network.py` is the canonical statement of this transformation: *"choice-order invariant,
while still passing multiple options in the same sequence"* — tokenize each choice separately, give
every choice the same starting position ids, mask so no choice sees another, then a bilinear
answer x choice score. Kairos implements exactly that. Confirmed on the real 26B weights:

```
@8K   FORWARD OK   logits (3,5)  finite=True   3.3s   peak 54.1 GB
      LOSS 1.2404  backward OK   grads on 722/722 trainable tensors
```

| requirement | evidence |
|---|---|
| permutation-equivariant head | max abs err **0.000e+00** at K=255 |
| one forward pass, no decoding | no LM head in the graph; logits from the bilinear readout |
| softmax per option, cannot emit invalid | `masked_fill(~choice_mask, -inf)` over the option set |
| state encoded once, shared | all three types (choice/noul/score) packed in one sequence |
| 64K packing | 63,889 tokens, `truncated=0` |
| two-mask flex path (25 sliding + 5 full, window 1024) | **runs** — never exercised before this |
| MoE safe | 22.84B expert params frozen for free |

## Three blockers found before any GPU-hours were spent

**1. `build()` cannot load a multimodal Gemma checkpoint.** `AutoConfig` gives a `Gemma4Config` whose
top-level `hidden_size` is **None** (it lives in the nested `Gemma4TextConfig`), and `AutoModel` maps
`gemma4` to `Gemma4Model`, which dies constructing the embedding. Fixed in our snapshot by routing
through `AutoModelForCausalLM`.

**2. My own trial OOMed because I never called `net.train()`.** `wrap()` enables gradient
checkpointing, but checkpointing only engages in training mode. In eval mode with grads on, every
activation is retained: 77.9 GB instead of 50.7. **The model was never the problem — my harness was.**

**3. The memory sweep killed the "train at 32K" plan I had recommended.**

| rung | nQ | branch budget | seqlen | fwd GB | result |
|---:|---:|---|---:|---:|---|
| 65,536 | 3 | 64/384 | 64,135 | 83.9 | OOM |
| 65,536 | 1 | 64/384 | 63,610 | 78.6 | OOM |
| 65,536 | 1 | 32/256 | 63,610 | 78.6 | OOM |
| **32,768** | 3 | 64/384 | 31,598 | **83.9** | **OOM** |

**32K OOMs identically to 64K**, and cutting three questions to one moves it only 83.9 → 78.6 GB.
The ~28 GB of overhead above the 50.7 GB model is **fixed** — independent of sequence length and
question count — which points at the MoE's 128 experts × 30 layers. **No amount of sequence
trimming fits it**, so "train at 32K, evaluate at 64K" would have failed exactly the same way.

## The fix: a guard whose premise was false

`train.py` refused ZeRO-3: *"zero_stage > 2 shards parameters; checkpoints here assume full params
per rank."* But `scorer.save()` calls `trainable_state()`, which serialises **only** parameters with
`requires_grad`, plus `head.*` and `lora_*` — **63.8M params, ~376 MB**. The frozen 25.2B base is
never written. Sharding it cannot corrupt a checkpoint that does not contain it.

Patched in **our snapshot only** (`kairos-20261011-26b`; the shared tree is untouched):
- the guard now raises only when the run is **not** LoRA-only (`_lora_only()` checks every trainable
  tensor is lora/norm/head);
- `trainable_state()` wraps the collect in `deepspeed.zero.GatheredParameters(trainable)` when
  params carry a `ds_id`. **Without this ZeRO-3 would have silently written empty shards** — a
  checkpoint that looks fine and contains nothing.

ZeRO-3 across 4 ranks puts ~12.7 GB of params per GPU instead of 50.7, leaving ~65 GB for the fixed
MoE overhead.

## A pod-RAM ceiling, separate from GPU memory

The first ZeRO-3 launch at 5 ranks reached **103 of 104 GiB of pod RAM** during tokenisation — each
rank tokenises the full 308,957-row corpus independently. Killed before the OOM-killer got it and
relaunched at 4 ranks: **66 GiB, healthy**. The limit cannot be raised: the namespace is already at
`requests.memory 650816Mi of 640Gi`.

**Lesson 13: GPU count is bounded by pod RAM during tokenisation, not only by VRAM.** Each rank holds
its own full tokenised copy, so ranks x corpus must fit host memory before a single GPU is touched.

---

# NATIVE 64K TRAINING IS RUNNING on Gemma-4-26B-A4B (2026-10-11 16:49 UTC)

First train events, true `pack.max_len 65536`, no 32K substitute, no mimicry:

```
plan:  total_steps 4565   warmup 456   world 3   backend deepspeed (ZeRO-3)
STEP 4/4565   98.8 s/step   mem 74.6 GB   nonfinite 0
s1:1.129/0.337  s2:1.481/0.182  s3:0.916/0.561  s4:1.025/0.614
3 GPUs at 100% utilisation
```

Every required property is live, verified on the real weights rather than assumed:

| requirement | evidence |
|---|---|
| one forward pass, no decoding | bilinear readout, no LM head in the graph |
| calibrated softmax per option | `log_softmax` over options + fitted temperature per type |
| permutation-equivariant head | max abs err **0.000e+00** at K=255 |
| cheap bilinear readout | `einsum(a_proj, c_proj)/sqrt(dim)` — O(options) |
| state encoded once, shared | choice/noul/score packed in one sequence |
| cannot emit an invalid answer | `masked_fill(~choice_mask, -inf)` |
| native 64K | `max_len 65536`, packer emits 63,889 tokens, `truncated=0` |
| MoE safe | 22.84B expert params frozen |

## Seven launches, seven distinct defects

| # | defect | fix |
|---|---|---|
| 1 | `build()` cannot load multimodal Gemma — `AutoConfig` gives top-level `hidden_size: None`, `AutoModel` maps `gemma4`→`Gemma4Model`, dies in the embedding | route through `AutoModelForCausalLM`, hand the text stack to `wrap()` |
| 2 | my trial OOMed at 77.9 GB — **I never called `net.train()`**, so gradient checkpointing never engaged | harness fix; model is only 50.7 GB |
| 3 | 64K backward OOM, **and 32K OOMs identically** (~28 GB fixed MoE overhead) | ZeRO-3, after proving the `stage>2` guard false for LoRA |
| 4 | 5 ranks → pod RAM 103/104 GiB | 4 ranks |
| 5 | stale `init_from: kairos-e4b-L1c-best` — shapes `[32,2560]` vs hidden 2816 | removed |
| 6 | 4 ranks → still 103/104 | 3 ranks |
| 7 | **ZeRO-3 asserts one dtype across trainables** (`zero/utils.py:217`) | promote 271 bf16 LoRA/norm tensors to fp32 |

**Defect 7 is the interesting one.** ZeRO-3 defragments all trainable parameters into one flat buffer
and requires a single dtype. Our architecture deliberately mixes: LoRA/norms are bf16 (they live in
the bf16 base) while the decision head is **fp32**, because — as bev-train's reference states —
*"bf16 logits are too coarse for a stable loss"*. I promoted the bf16 trainables to fp32 (63.8M
params, ~128 MB) rather than downcasting the head, so the reference's numerical intent survives.

## A memory model correction

I assumed short replay drove tokenisation memory because it is 93.3% of rows. **Wrong.** 17,514 long
rows carry ~715M tokens against ~153M for all 244,617 short rows — the long rows are **82% of the
token mass**. Cutting short replay removed only ~11% of memory, which is why ranks had to come down
from 5 to 3 instead.

**Lesson 13: rank count here is bounded by pod RAM during tokenisation, not VRAM.** Each rank holds
its own full tokenised copy, so `ranks x corpus_tokens` must fit host memory before a single GPU is
touched. A pre-tokenised shared cache would remove this wall permanently.

## The 125-hour problem, and what was done about it

4,565 steps at 98.8 s/step = **125 h**. Not tomorrow.

Capping the epoch does **not** speed up training — the model sees the same data per hour either way.
What it buys is a **clean finish**: temperature calibration, a test report, `final/`, and the
unconditional `last/`. An uncapped run gets killed partway holding only a `best/` that a saturated
dev can freeze — precisely how the E4B run lost all 137 steps of work.

Restarted as `jev26b-final64k-20261011-170025` with `seqs_per_epoch 7000`, and **long-source weights
raised 0.08 → 0.40** (`llmops_voice_64k` 0.22, `open:otel_safety` 0.10, `churn_long64k` 0.08). With
few steps available, they should be spent on long rows rather than short replay — the baseline to
beat is at 64K, not 8K.

---

## Failure #8: under ZeRO-3 a rank-0-only forward is a deadlock, not a slow path

The native-64K run `jev26b-final64k-20261011-170025` trained cleanly for 25 steps — 81.2 s/step,
74.6 GB, zero non-finite — and then died. The launcher reported only `ChildFailedError`, which says
nothing. The real evidence was two lines deeper:

```
WorkNCCL(SeqNum=34131, OpType=ALLREDUCE, NumelIn=1) timed out
died 18:03:58   — exactly 600 s after step 25   — and eval_every is 25
```

Three facts identify this uniquely. A **1-element ALLREDUCE is how NCCL implements
`dist.barrier()`**. 600 s is the default NCCL watchdog. And `log.jsonl` contained **0 dev events**,
so the evaluation never returned a result — it hung on its very first forward pass.

### Root cause

Under ZeRO-3 parameters are sharded, so **every forward pass all-gathers them layer by layer**.
That makes `evaluate()` a *collective*. `train.py` ran it under `if rank == 0:` while ranks 1–2
waited in `dist.barrier()` — the structure the original author chose deliberately, and which is
correct under ZeRO-2. Under ZeRO-3 it is a hard deadlock:

| rank | where it blocked | waiting for |
|------|------------------|-------------|
| 0    | first all-gather inside `evaluate()` | ranks 1–2 to join the gather |
| 1, 2 | `dist.barrier()` | rank 0 to reach the barrier |

The watchdog resolved it by killing the job. **Raising the timeout would not have fixed this** — it
would have deadlocked for two hours instead of ten minutes. That was my first instinct and it was
wrong; the single-element allreduce is what ruled it out.

### The same bug in two more places, including my own patch

Once the mechanism is clear, every rank-0-guarded collective is suspect. There were three:

1. the periodic dev eval (`train.py:331`) — what actually fired;
2. `save(best)` inside it — `trainable_state()` calls `deepspeed.zero.GatheredParameters`, also
   collective;
3. the entire teardown block (`train.py:353`).

Worse, **the teardown patch I wrote this morning was itself broken by this.** It destroys the
process group *before* `save(out/"last")` — on the theory that "nothing below is collective." Under
ZeRO-2 that was true. Under ZeRO-3 `save()` is collective, so `last/` would have failed too, and the
run would have lost its weights in exactly the way the patch existed to prevent. The patch was
written against ZeRO-2 and silently invalidated when I enabled ZeRO-3 four hours later.

### Fix: collective compute on every rank, I/O on rank 0

- `scorer.save()` — all ranks run `trainable_state()`; only rank 0 writes. Without the rank guard
  three ranks race on the same `weights.pt`.
- periodic eval — drop the rank-0 guard. `evaluate()` runs under `net.eval()`, so it is
  deterministic: every rank computes the same score, takes `score > best` identically, and enters
  `save()` together. Under ZeRO-3 the other ranks must participate in the gathers anyway, so the
  duplicated compute costs **zero** extra wall-clock.
- teardown — `save(last/)` on all ranks *before* `destroy_process_group()`.
- `skip_post_eval` — the rank-0 calibrate+test block cannot run after the group is destroyed, and
  its `load_state_dict(full_tensors)` into sharded parameters is unsound regardless. Gated off;
  scoring moves to a separate single-GPU job, which is needed anyway for the per-slice breakdown.

### Lesson 14

**A correctness argument has a scope, and changing the execution backend silently voids it.** The
comment "nothing below is collective" was true when written and false six hours later, with nothing
in the code to notice. ZeRO-3 does not merely shard memory — it converts *every forward pass* into a
collective operation, so the entire `if rank == 0:` idiom becomes unusable. When a patch's
justification depends on a config value, the patch should assert that value rather than assume it.

### Lesson 15

**`ChildFailedError` is not a diagnosis, and neither is the exit code.** The three ranks reported
SIGTERM, SIGABRT, SIGABRT — noise from the launcher tearing down the group. The actual cause was
`NumelIn=1` in a NCCL work descriptor 40 lines earlier. Reading only the traceback would have led to
"transient NCCL flake, just restart" — and the restart would have died at step 25 again.
