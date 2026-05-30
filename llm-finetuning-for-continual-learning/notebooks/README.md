# LLM Fine-Tuning for Continual Learning

This project explores **catastrophic forgetting** in large language model fine-tuning and tests a simple **experience replay** strategy to reduce forgetting across domains.

The experiment fine-tunes **Mistral-7B-Instruct-v0.2** on MMLU Physics questions and evaluates whether the model forgets performance on Biology questions. After observing forgetting, the project applies experience replay by mixing a small subset of Biology examples back into the Physics training data.

---

## Project Overview

Large language models can lose performance on previously learned or retained knowledge when fine-tuned on a new domain. This behavior is known as **catastrophic forgetting**.

In this experiment:

- **Physics** is treated as the new fine-tuning domain.
- **Biology** is treated as the retained domain.
- The base model is first evaluated on both Physics and Biology.
- The model is then fine-tuned only on Physics.
- Biology performance drops after Physics-only fine-tuning.
- Experience replay is applied by mixing Biology examples into the Physics training data.
- The model is evaluated again to check whether Biology performance is recovered.

---

## Problem Statement

When an LLM is fine-tuned on a new domain, it may become more specialized in that domain but lose performance on another domain. The goal of this project is to test whether **experience replay** can help preserve retained-domain performance while still improving new-domain performance.

The main research question is:

> Can a small replay buffer of old-domain examples reduce catastrophic forgetting during supervised fine-tuning of an LLM?

---

## Methodology

The experiment follows three phases:

### Phase 1: Baseline Evaluation

The base Mistral model is evaluated on selected MMLU Physics and Biology subjects before fine-tuning.

This establishes the starting performance of the model.

### Phase 2: Physics-Only Fine-Tuning

The model is fine-tuned only on Physics examples using supervised fine-tuning.

After this phase, the model is evaluated again on both Physics and Biology.

This phase tests whether fine-tuning on Physics causes the model to forget Biology.

### Phase 3: Experience Replay

A replay dataset is created by mixing:

- all Physics training examples
- a small subset of Biology training examples

The replay ratio used is:

```python
REPLAY_RATIO = 0.25
