# PAPER NOTES: Publication Strategy and Scientific Positioning

> Working notes for the future paper-writing phase. This file captures the publication-oriented analysis discussed during development, so we can return to it later without losing context.

---

## 1. Core functional objective of the project

The main objective of this project is not only to build a strong classifier.

The real product goal is to create a small language model that can help us "talk" with text messages and emails in order to:

- identify whether persuasion principles are present
- explain why they are present
- point to textual evidence
- support interactive reasoning about phishing persuasion
- serve as an analyst-assistance tool, not only as a static predictor

This means:

- the classifier is an important quantitative anchor
- the SLM is the practical and functional centerpiece of the system
- the final scientific framing should respect both roles

Important clarification recorded later in development:

- the SLM is not just an auxiliary generation module
- an essential project goal is to let the user "talk" with suspicious messages in order to understand the persuasion principles involved
- this strengthens the case for presenting the system as an analyst-assistance and explainability platform, not only as a classifier

Additional conceptual clarification recorded later:

- the intended role of the SLM is not to act as a rigid extractor only
- the intended role is also not to behave as a naive justifier of classifier outputs
- the desired system behavior is:
  - the classifier proposes an initial hypothesis
  - the SLM receives that hypothesis as context
  - the SLM analyzes the message in light of that hypothesis
  - the SLM may confirm, nuance, expand, or correct the classifier's reading

Compact formulation:

- classifier as quantitative prior
- SLM as analyst-facing reasoning layer

---

## 2. Honest publication assessment

Current opinion:

- The project is interesting and potentially publishable.
- In its current state, it is not yet a "safe" Q1 paper.
- It does have a credible path toward a Q1-level submission if the next iteration strengthens the methodology and evaluation.

Practical assessment:

- As an applied bilingual phishing-persuasion system: yes, promising
- As a strong journal Q1 paper right now: not yet
- As a foundation for a stronger Q1 paper after one more methodological pass: yes

---

## 3. Why this could be publishable

The project already has several elements with publication value:

- bilingual persuasion analysis in phishing messages
- multi-label detection of persuasion principles
- combination of discriminative classification and generative explanation
- practical cybersecurity relevance
- explainability-oriented interaction through an SLM
- hardware-aware local training and deployment pipeline

These ingredients are meaningful, especially if the paper is positioned well.

---

## 4. Main weaknesses today

These are the current limitations that reduce confidence for a top-tier submission:

1. The classifier is stronger methodologically than the SLM.
2. The SLM evaluation is not yet strong enough to support very hard generalization claims.
3. ECE and formal calibration are still pending.
4. The pipeline has not yet been fully hardened for the final paper run.
5. The current SLM-oriented data flow is useful operationally, but not yet ideal for the strongest anti-leakage methodological defense.
6. The contribution narrative is not yet frozen.

Important nuance:

- These weaknesses do not make the project unpublishable.
- They mainly affect how ambitious the target venue can be and what claims can be defended safely.

---

## 5. Strategic decision already taken

Decision recorded during development:

- First finish the current full run.
- Do not perform a major anti-leakage refactor before finishing that run.
- Use the classifier as the main quantitative anchor for the current iteration.
- Treat the SLM as a support, explainability, and analyst-interaction component for now.
- Avoid overclaiming the current SLM evaluation as strong leakage-free generalization evidence.

Why this decision makes sense:

- the required refactor is substantial
- the immediate benefit is limited for the current execution milestone
- we need a full stable run and full artifact generation first
- the paper-oriented hardening can be done in the next iteration

---

## 6. Recommended scientific posture for the current iteration

Safest current posture:

- the classifier provides the primary quantitative evidence
- the SLM provides explanatory and interactive analytical value
- the system contribution is the combination of detection plus explainable interaction

This means we should avoid saying:

- "the SLM has already been rigorously validated as a leakage-free generalization model"

And prefer saying:

- "the classifier provides the principal quantitative evaluation"
- "the SLM extends the system with interactive explanation and persuasion-oriented analysis"
- "the SLM is part of the practical analyst-facing system and will benefit from further methodological hardening in future work"

Refined wording for later writing:

- the SLM should be described as starting from classifier-informed context, not as operating in total isolation
- however, it should also not be described as a mere formatter of classifier decisions
- a better description is that the SLM performs analyst-oriented interpretation over a classifier-proposed reading of the message

---

## 7. Publication route options

### Option A. Realistic Q1 route

This is the recommended path.

Position the paper as:

- a bilingual multi-label persuasion-principle detection system for phishing messages
- with an explainability-oriented SLM layer for analyst support

What would strengthen this route:

- strong classifier evaluation
- per-class analysis
- per-language analysis
- baselines and comparisons
- error analysis
- calibration and ECE
- better framing of the SLM as explainability support
- some human-centered evaluation of the SLM outputs

Why this route is attractive:

- it matches the real strengths of the project
- it avoids putting all scientific pressure on the SLM alone
- it gives a coherent cybersecurity + explainability narrative

Main risk:

- reviewers may ask for stronger evidence on the SLM side

### Option B. Faster applied paper

Position the work as a practical applied system with useful bilingual persuasion analysis and generation support.

Benefits:

- faster route to submission
- lower methodological pressure
- easier to write from current artifacts

Drawback:

- lower probability of being competitive for a strong Q1 venue

### Option C. SLM-centered paper

Make the main contribution the ability to interact with persuasive phishing messages through an SLM.

Benefits:

- conceptually attractive
- potentially more novel

Drawbacks:

- much harder to defend
- requires stronger evaluation:
  - grounding
  - textual evidence quality
  - consistency
  - human agreement
  - bilingual robustness

This is the most ambitious option, but also the riskiest.

### Option D. Two-step publication strategy

First publish a more applied or less ambitious version.
Then produce a stronger, extended version for a better venue.

Benefits:

- reduces immediate pressure
- gives time to harden the SLM methodology

Drawback:

- slower path to the strongest final paper

---

## 8. Best current recommendation

Recommended route:

- aim for a Q1 paper
- but do not frame it as "we trained an SLM and that alone is the contribution"
- instead frame it as:
  - bilingual multi-label persuasion detection in phishing
  - supported by an SLM for explanation and interactive analysis

This is likely the strongest balance between:

- what the system already does well
- what can be defended rigorously
- what can become compelling with one more iteration

---

## 9. What would most improve Q1 chances

The following additions would likely improve the paper the most:

1. Clear research question
2. Strong classifier baselines
3. Per-class and per-language metrics
4. Error analysis
5. Threshold analysis and calibration
6. Human evaluation of the SLM
7. Careful claim discipline around what is validated and what is not
8. Strong cybersecurity motivation and use-case framing

---

## 10. Current methodological caution around the SLM

Current honest stance:

- the SLM is already useful for the intended practical objective
- the current system can already support interactive analysis of messages
- however, the current pipeline should not yet be presented as the final strongest methodological version of SLM evaluation

This does not block:

- finishing the current run
- generating all artifacts
- using the SLM in the system
- writing about the SLM as a key functional component

It only limits:

- how strong the generalization claims should be
- how aggressively the paper should center the SLM quantitatively

Another important caution:

- improving grounding should not be confused with narrowing the SLM into a low-capability extraction tool
- for this project, the practical value of the SLM depends precisely on its ability to support rich message-centered conversation
- the methodological challenge is therefore to keep analytic freedom while increasing textual discipline and auditability

---

## 11. Writing guidance for later

When we reach the paper-writing phase, keep this framing in mind:

Good framing:

- "We propose a bilingual system for detecting persuasion principles in phishing messages and supporting analyst interpretation through an SLM-based explanation layer."

Safer framing:

- "The classifier provides the main quantitative foundation, while the SLM adds interactive explanatory value."

Stronger and more precise framing:

- "The classifier provides the initial quantitative hypothesis, while the SLM acts as a message-centered analyst that interprets, validates, and expands that hypothesis through grounded interaction."

Avoid overclaiming:

- do not claim stronger SLM methodological certainty than we truly have
- do not imply that the current SLM pipeline alone has already solved all evaluation concerns

---

## 12. Deferred paper-hardening tasks

These are strong candidates for the next methodological iteration after the current run is complete:

- stricter anti-leakage pipeline design
- split-first data flow across the full system
- deduplication and near-duplicate checks
- ECE and calibration
- stronger SLM evaluation protocol
- possible human evaluation or expert review of SLM outputs
- stronger ablation experiments

---

## 13. Final memory anchor

If we resume this discussion later, the key takeaway is:

- the project is promising and can plausibly become publishable in a strong venue
- the most realistic high-value path is a classifier-centered quantitative paper with an SLM explainability layer
- the current development priority is to finish the full run and collect stable artifacts
- the next iteration should harden the methodology for the eventual paper

---

## 14. New methodological diagnosis after internal SLM audit

An internal benchmark and pipeline audit performed on `2026-04-29` clarified the current SLM situation.

### 14.1 Current empirical result
The fine-tuned adapter `Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429` was compared directly against its base model `Qwen/Qwen2.5-1.5B-Instruct`.

Observed result:
- tuned score: `0.75`
- base score: `0.7917`
- delta: `-0.0417`
- grounding delta: `0`
- honesty delta: `0`
- hallucination delta: `0`
- external-drift delta: `-1`

Interpretation:
- the current fine-tuning does not improve the base model on the evaluated analyst task
- it slightly degrades discipline by increasing external or abstract drift

### 14.2 Main pipeline causes identified

#### A. Label-informed synthetic teaching
The augmenter currently gives the teacher model the active positive principles in advance.

That means the teacher is not solving:
- "what is supported by the text?"

It is mostly solving:
- "justify these already selected principles"

This is a major source of:
- rationalization
- overinterpretation
- weak falsification behavior

#### B. Training task mismatch
The SLM is trained mainly on monologic analysis prompts:
- identify principles
- explain them
- assign intensity

But the intended product and scientific role now requires a different behavior:
- start from classifier prior
- confirm, nuance, refute, compare, and explain
- answer arbitrary follow-up questions honestly

So the current tuning objective does not match the target conversational analyst behavior.

#### C. Severe truncation
The generative pipeline trains the SLM with `max_length=256`.

Measured over `500` sampled augmented conversations:
- average total conversation length: `671.81` tokens
- median: `694`
- p90: `1129`
- p95: `1199`
- max: `2212`
- samples over `256`: `406/500`
- samples over `512`: `311/500`

Implication:
- most training conversations are truncated
- the model frequently sees incomplete targets
- this likely damages coherence, closure, and evidence-rich explanation quality

#### D. Mixed supervision styles
The harmonized dataset combines:
- `2092` IWSPA records without human justifications
- `1395` Spaphish records with human justifications

This creates a supervision asymmetry:
- one part is teacher-generated and often verbose or overconfident
- one part is human-authored and usually shorter and more anchored

The augmented dataset also mixes output languages and formats.

### 14.3 Updated publication implication
This diagnosis strengthens the earlier caution:

- the SLM is still central to the product and scientific narrative
- but the current adapter should not be presented as evidence of a successfully specialized analyst model

Safer claim:
- the current system demonstrates the feasibility and utility of the classifier-plus-SLM architecture
- however, the first generative fine-tuning iteration did not yet improve the base SLM on the desired analyst benchmark

### 14.4 Consequence for the next paper-hardening cycle
The next methodological iteration should focus on:
- redesigning the SLM dataset around grounded confirm/nuance/refute behavior
- preventing teacher leakage from positive labels
- reducing or restructuring truncation
- unifying output format and language
- strengthening benchmark-based acceptance criteria for new adapters

Compact takeaway:
- the project direction remains valid
- the current SLM fine-tuning recipe does not yet justify strong claims of improvement over the base model

### 14.5 Immediate engineering response already implemented
A first corrective intervention was already applied in code after this diagnosis:

- the augmenter was redesigned toward `hypothesis -> audit` supervision
- the synthetic prior now contains limited controlled noise instead of a direct list of positive principles as ground truth
- the SLM fine-tuner now:
  - uses a larger dedicated context window (`slm_max_length = 1024`)
  - masks the prompt and supervises only assistant tokens
  - gives more budget to the answer span during truncation

Important note:
- this intervention has been implemented
- but its scientific effect has not yet been validated through a fresh augmentation + retraining + benchmark cycle

So the current paper-safe statement remains:
- the first fine-tuning recipe underperformed the base model
- a second, more disciplined recipe is now prepared and awaiting empirical validation

### 14.6 Data and augmentation audit update 2026-05-01

The data interpretation was corrected:
- persuasion labels are the primary scientific target
- phishing labels are a separate supervised cybersecurity label
- persuasion evidence must not be treated as automatic phishing evidence

The harmonizer now supports dynamic CSV ingestion:
- it scans `data/*.csv`
- it identifies supported datasets by columns, not by filenames
- it records `dataset_file` for traceability
- it invalidates stale checkpoints using an input fingerprint and schema version

The current verified corpus is:
- `3487` total records
- `2092` IWSPA records
- `1395` Spaphish records
- `1736` phishing records
- `1751` legitimate records

The harmonized artifact was verified as:
- schema `harmonized_v2`
- zero empty IDs
- zero non-ASCII payloads
- Spaphish justifications for positive principles match annotator votes aligned with the final principle label

The augmented SLM dataset is now treated as a reusable research artifact:
- canonical path: `outputs/artifacts/augmented_dataset.jsonl`
- schema: `audit_v4`
- it stores generation source, labels, classifier prior, justifications, annotation details, phishing label source, and row-level quality metadata
- a quality gate blocks SLM training if this artifact is stale, malformed, non-ASCII, or missing required fields

Paper implication:
- future claims about the SLM must be based on `audit_v4` augmentation and the paired base-vs-adapter benchmark
- old `audit_v3` results should be considered pre-correction and not used as final evidence
