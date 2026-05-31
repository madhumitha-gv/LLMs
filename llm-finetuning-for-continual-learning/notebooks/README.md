# LLM Fine-Tuning for Continual Learning

This project explores catastrophic forgetting in large language model fine-tuning and tests whether a simple experience replay strategy can reduce forgetting across domains.

The experiment fine-tunes **Mistral-7B-Instruct-v0.2** on selected **MMLU Physics** questions and evaluates performance on both Physics and Biology. After Physics-only fine-tuning, the model shows a drop in Biology accuracy. Experience replay is then applied by mixing a small subset of Biology examples into the Physics training data.

---

## Objective

The goal of this experiment is to study whether a model fine-tuned on a new domain forgets performance on another domain, and whether replaying a small number of old-domain examples can reduce that forgetting.

In this setup:

- **New domain:** Physics
- **Retained domain:** Biology
- **Continual learning method:** Experience Replay
- **Fine-tuning method:** LoRA-based supervised fine-tuning

---

## Experiment Design

The experiment is divided into three phases:

### 1. Baseline Evaluation

The base Mistral model is evaluated on selected MMLU Physics and Biology subjects before fine-tuning.

### 2. Physics-Only Fine-Tuning

The model is fine-tuned only on Physics examples. After training, the model is evaluated again on both Physics and Biology to check whether Biology performance drops.

### 3. Experience Replay

A replay dataset is created by combining:

- Physics training examples
- A small subset of Biology training examples

The replay ratio used is:

```python
REPLAY_RATIO = 0.25
```

The replay dataset is created as:

```python
n_bio = int(len(physics_train) * REPLAY_RATIO)
bio_subset = biology_train.select(range(n_bio))

replay_ds = concatenate_datasets([physics_train, bio_subset]).shuffle(seed=SEED)
```

This creates a mixed training dataset where Biology examples act as replay examples during continued fine-tuning.

---

## Dataset

This project uses selected subjects from the **MMLU** benchmark.

### Physics Subjects

- `college_physics`
- `high_school_physics`
- `conceptual_physics`

### Biology Subjects

- `college_biology`
- `high_school_biology`
- `medical_genetics`

Each example is formatted as a multiple-choice instruction prompt for the model.

Example format:

```text
<s>[INST] Question: ...
(A) option 1
(B) option 2
(C) option 3
(D) option 4
Answer with only the letter. [/INST] C</s>
```

---

## Model and Fine-Tuning Setup

The base model used is:

```text
mistralai/Mistral-7B-Instruct-v0.2
```

The model is fine-tuned using:

- Supervised Fine-Tuning
- PEFT / LoRA
- TRL `SFTTrainer`
- Unsloth
- Hugging Face Transformers and Datasets

LoRA is applied to the model instead of full fine-tuning, which makes the experiment more memory-efficient.

---

## Results

| Phase | Physics Accuracy | Biology Accuracy |
|---|---:|---:|
| Baseline | 40.4% | 67.6% |
| After Physics Fine-Tuning | 41.4% | 59.5% |
| After Experience Replay | 52.5% | 66.7% |

---

## Key Findings

After Physics-only fine-tuning, Biology accuracy dropped from **67.6%** to **59.5%**, indicating catastrophic forgetting.

After applying experience replay, Biology accuracy recovered to **66.7%**, close to the original baseline. Physics accuracy also improved from **41.4%** to **52.5%**.

This shows that even a simple replay strategy can help reduce forgetting while still improving performance on the new domain.

---

## Evaluation Method

The model is evaluated using logit-based multiple-choice answer selection.

For each question, the model compares the logits for answer choices:

```text
A, B, C, D
```

The answer option with the highest score is selected as the model prediction.

---

## Training Configuration

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
REPLAY_RATIO = 0.25
MAX_LENGTH = 512
```

LoRA configuration:

```python
r = 128
lora_alpha = 256
lora_dropout = 0.05
```

---

## Setup

Clone the repository and open the notebook in Google Colab or another GPU-enabled environment.

Install the required dependencies:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

Run the notebook cells in order:

1. Install dependencies
2. Load and format MMLU datasets
3. Load the Mistral model
4. Apply LoRA adapters
5. Run baseline evaluation
6. Fine-tune on Physics
7. Evaluate Physics and Biology
8. Apply experience replay
9. Evaluate final results

---

## Limitations

This is a small experimental prototype. Some limitations include:

- Only selected MMLU subjects were used.
- Only one replay ratio was tested.
- The replay buffer is simple and fixed.
- Evaluation is based on accuracy only.
- More rigorous train, validation, and test separation could improve the setup.

---

## Future Work

Possible extensions include:

- Testing multiple replay ratios.
- Randomly sampling Biology replay examples.
- Evaluating additional MMLU domains.
- Comparing experience replay with other continual learning methods.
- Adding per-subject accuracy analysis.
- Saving LoRA adapter checkpoints separately.

---

## Main Takeaway

This experiment shows that catastrophic forgetting can appear during domain-specific LLM fine-tuning. Mixing a small number of old-domain examples into the new-domain training data helped recover Biology performance while also improving Physics performance.

The main contribution of this project is demonstrating a simple continual learning approach for reducing forgetting during supervised LLM fine-tuning.

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL `SFTTrainer`
- PEFT / LoRA
- Unsloth
- bitsandbytes
- Mistral-7B-Instruct
- Google Colab

---

## Author

Madhumitha Gannavaram
