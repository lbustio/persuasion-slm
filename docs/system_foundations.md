# System Foundations: Theory, Premises, Hypotheses, Solution, and Architecture

> Canonical conceptual document for the project. This file explains in a single place what problem is being addressed, what theoretical assumptions support the work, what hypotheses motivate the system, what solution is being built, and how the architecture is organized. It is meant to complement `bitacora.md`, `docs/blueprint.md`, `paper.md`, and `webapp.md`.

---

## 1. What this project is really about

This project is not only about phishing detection.

Its real scope is broader:

- detect persuasion principles in suspicious messages
- explain why those principles appear
- ground the explanation in textual evidence
- support interaction with the message through a small language model
- help a human understand how the message is trying to influence behavior

So the project has two simultaneous goals:

1. a **quantitative detection goal**
2. an **interactive explanatory goal**

The first is handled mainly by the classifier.
The second is handled mainly by the fine-tuned SLM.

The combination is what gives the system its real identity.

---

## 2. Theoretical background

### 2.1 Core phenomenon

Phishing is not only a technical attack surface problem.
It is also a persuasion problem.

Many phishing messages succeed not only because of spoofing, branding, or delivery strategy, but because they exploit persuasion mechanisms that alter trust, urgency, compliance, and perceived legitimacy.

### 2.2 Persuasion principles as analytic units

The system models phishing persuasion through a fixed taxonomy of five persuasion principles:

- `AUTH`: authority
- `SP`: social proof
- `LSD`: liking / similarity / deception
- `CRC`: commitment / reciprocation / integrity
- `DIS`: distraction / urgency

These principles are treated as analytically meaningful labels that can co-occur in the same message.

This implies a **multi-label** formulation rather than a single-label or binary-only formulation.

### 2.3 Why explanation matters

In cybersecurity, a simple label is often not enough.

A human analyst, trainer, educator, or reviewer may need to know:

- why a message is suspicious
- which persuasive mechanisms are present
- what parts of the text support that interpretation
- how the message might affect a non-expert reader

This is the core reason the project includes an SLM-based explanatory component rather than relying only on a classifier.

---

## 3. Problem statement

The practical problem can be stated as follows:

> Given a phishing or suspicious message in English or Spanish, identify which persuasion principles are present, estimate their strength, explain the rationale behind that interpretation, and support a human in discussing and understanding the message.

This problem has several subproblems:

1. build a coherent bilingual dataset
2. detect multiple persuasion principles per message
3. generate useful explanations
4. preserve enough rigor for scientific reporting
5. make the final system usable by non-experts

---

## 4. Premises of the project

The system is built on the following premises.

### 4.1 Phishing persuasion can be operationalized

The project assumes that persuasion principles in phishing messages can be annotated, modeled, and detected with useful consistency.

### 4.2 The same message may contain several principles

The project assumes that phishing persuasion is not exclusive by class.

A message may simultaneously contain:

- authority
- urgency
- social proof

or other combinations.

Therefore, the system must support multi-label inference.

### 4.3 Explanation is not optional

The project assumes that a useful cybersecurity support system should not stop at prediction.

It should also provide:

- interpretive support
- evidence
- analyst-facing reasoning

### 4.4 Bilinguality matters

The system assumes that a relevant practical solution should handle at least English and Spanish.

This affects:

- data harmonization
- classifier training
- explanation generation
- UX design

### 4.5 A smaller adapted model can be more useful than a large generic one

The project assumes that a carefully fine-tuned SLM can become a more practical assistant for this domain than an untouched general-purpose instruction model.

This is one of the reasons for using:

- a larger teacher model for augmentation
- a smaller Qwen model for downstream analyst interaction

### 4.6 Human usability matters

The project assumes that scientific rigor alone is not enough.

If the resulting system is to be useful:

- it must communicate clearly
- it must expose reasoning accessibly
- it must support non-expert use

---

## 5. Main hypotheses

The system is motivated by several working hypotheses.

### 5.1 Classifier hypothesis

A bilingual multi-label encoder-based classifier can learn to detect persuasion principles in phishing messages with meaningful quantitative performance.

### 5.2 Explanation hypothesis

A domain-adapted small language model can provide useful explanations about persuasion principles that go beyond raw class prediction.

### 5.3 Hybrid-system hypothesis

The combination of:

- a classifier for structured detection
- and an SLM for explanation and interaction

will be more useful than either component alone for analyst-support purposes.

### 5.4 Educational hypothesis

A system that highlights principles, evidence, and explanations can help non-experts understand suspicious messages better than a simple risk score or binary output.

### 5.5 Product hypothesis

A visually careful, didactic, technically serious interface can make the scientific contribution more communicable and the system more practically usable.

---

## 6. Solution strategy

The solution being developed is a hybrid pipeline with several coordinated stages.

### 6.1 Stage 1: Data harmonization

Goal:

- unify heterogeneous bilingual datasets into a single coherent corpus

Main responsibilities:

- read IWSPA and Spaphish
- normalize structure
- unify label schema
- preserve language identity
- preserve justifications where available

Output:

- harmonized dataset suitable for classifier training and downstream augmentation

### 6.2 Stage 2: Explanation-oriented augmentation

Goal:

- enrich the corpus with explanation-like supervision

Main idea:

- use a larger instruction model as a teacher
- reuse human justifications where available
- generate explanation-style assistant responses when needed

This stage exists to help create the kind of data needed for a message-discussion SLM.

### 6.3 Stage 3: Multi-label classifier training

Goal:

- learn to detect the persuasion principles quantitatively

Main role in the system:

- produce class probabilities
- act as the structured detection backbone
- serve as the main quantitative anchor of the project

### 6.4 Stage 4: SLM fine-tuning

Goal:

- turn a compact instruction model into a domain-specific assistant that can talk about phishing persuasion

Important clarification:

- the project is not pretraining Qwen from scratch
- it is fine-tuning a smaller Qwen instruction model with LoRA/QLoRA adapters

The larger Qwen teacher and the smaller Qwen SLM have different roles:

- `Qwen 7B`: teacher for augmentation
- `Qwen 1.5B`: final conversational/explanatory SLM

### 6.5 Stage 5: Research artifact export

Goal:

- avoid recomputing publication assets later

Outputs include:

- figures
- tables
- predictions
- reports
- split manifests

### 6.6 Stage 6: Webapp / analyst interface

Goal:

- make the system explainable, demoable, didactic, and practically useful

The app is not separate from the research vision.
It is part of how the value of the system is expressed.

---

## 7. Architectural roles of the main components

### 7.1 Harmonizer

Role:

- establish a clean and unified dataset contract

Without this stage:

- labels remain inconsistent
- language handling is fragmented
- downstream training becomes less reliable

### 7.2 Augmenter

Role:

- create explanation-oriented data for the SLM

It is not the final model.
It is a data-enrichment stage.

### 7.3 Classifier

Role:

- detect persuasion principles quantitatively

It is the main source of:

- structured multi-label prediction
- score profiles
- research metrics
- quantitative evidence for the paper

### 7.4 SLM

Role:

- converse about the message
- explain principles
- discuss ambiguity
- make the system useful beyond raw prediction

This is central to the product goal of “talking with messages.”

### 7.5 Paper artifact manager

Role:

- capture the outputs that will later support research writing

This exists because publication support is a first-class concern of the project.

### 7.6 Webapp / future app

Role:

- provide a research-grade, didactic, non-expert-accessible interface

It acts as:

- demo layer
- analyst-support layer
- educational layer
- scientific showcase layer

---

## 8. Current architecture in one narrative

At a high level, the architecture works like this:

1. raw phishing datasets are harmonized
2. the harmonized dataset feeds the classifier pipeline
3. the same harmonized data also supports augmentation for SLM-oriented supervision
4. a larger teacher model helps generate explanation-style training material
5. a smaller Qwen model is fine-tuned to discuss messages in this domain
6. the classifier provides structured probabilities and principle profiles
7. the SLM provides explanation and interaction
8. research outputs are exported for paper use
9. a web application presents all this in a usable and demonstrable way

This is therefore a **hybrid discriminative + generative architecture**.

---

## 9. What the project is not

To avoid confusion, this project is not:

- a plain phishing detector only
- a generic chatbot over security text
- a full pretraining effort for Qwen
- a purely academic benchmark with no product layer
- a product-only demo with no scientific backbone

It is instead:

- a bilingual persuasion-analysis system
- with a quantitative classifier
- with an explanatory conversational SLM
- with research export support
- and with a future analyst-facing app

---

## 10. Current scientific posture

At this stage of development, the safest scientific position is:

- the classifier is the main quantitative anchor
- the SLM is a central functional and explanatory component
- the current iteration prioritizes completing a full stable run
- further methodological hardening will come afterward

This means the architecture is already meaningful, but the strongest final paper claims should wait until:

- the current run is completed
- the artifacts are inspected
- remaining methodological gaps are addressed

---

## 11. Main open gaps recognized by the project

The project already acknowledges several important gaps:

- calibration and ECE are not yet fully integrated
- the strongest anti-leakage design is not yet the current pipeline priority
- SLM evaluation still needs stronger methodological hardening
- some final paper-grade validation steps remain for the next iteration

These are not signs that the project lacks direction.
They are signs that the system is in an advanced but still evolving stage.

---

## 12. Why this architecture makes sense

This architecture is coherent because it separates concerns:

- data consistency is handled early
- classification is optimized for structure and metrics
- generation is optimized for explanation and interaction
- artifacts are exported for reproducibility and publication
- the future app is optimized for usability and demonstration

This division of labor is one of the strongest design features of the system.

---

## 13. Canonical summary

If the whole project had to be summarized in one paragraph:

This project develops a bilingual system for analyzing phishing messages through persuasion principles. It combines a harmonized labeled dataset, a master split that protects downstream evaluation, a multi-label classifier for structured detection, and a fine-tuned small language model for explanation and interaction. A larger teacher model helps create explanation-oriented supervision only from allowed training splits, while exported research artifacts support later publication work. The long-term goal is not only to classify messages, but to let a human inspect, understand, and discuss how a suspicious message tries to persuade its target.

---

## 14. Methodological hardening added on 2026-04-30

The project now treats data partitioning as a first-class research object.

The pipeline creates a master split before downstream training:

- train
- validation
- test
- heldout_final

This split is shared by:

- classifier training
- SLM data augmentation
- SLM fine-tuning
- webapp examples

The SLM augmentation phase uses only train and validation records.
It does not use test or heldout_final records.

The phishing label is also tracked by provenance:

- direct dataset label
- inferred from persuasion labels
- unknown, if applicable later

This matters because a model trained with inferred phishing labels should not be described as if all labels were equally strong human phishing ground truth.

The SLM acceptance gate now requires at least 30 paired benchmark cases before it can claim strong superiority over the base model.
Smaller benchmarks remain useful for debugging, but they produce an insufficient-evidence decision instead of a strong acceptance.
