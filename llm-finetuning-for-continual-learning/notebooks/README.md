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
```

This means the number of Biology replay examples is 25% of the Physics training set size.

In the notebook, the replay dataset is created as:

```python
n_bio = int(len(physics_train) * REPLAY_RATIO)
bio_subset = biology_train.select(range(n_bio))

replay_ds = concatenate_datasets([physics_train, bio_subset]).shuffle(seed=SEED)
```

The `bio_subset` acts as a simple fixed replay buffer. It is not a reinforcement learning replay buffer with state-action-reward transitions. Instead, it stores old-domain supervised examples that are reused during continued fine-tuning.

---

## Dataset

This project uses the **MMLU** benchmark through the Hugging Face `datasets` library.

### Physics Subjects

- `college_physics`
- `high_school_physics`
- `conceptual_physics`

### Biology Subjects

- `college_biology`
- `high_school_biology`
- `medical_genetics`

Each MMLU example is formatted into an instruction-style multiple-choice prompt.

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

## Model

The model used in this experiment is:

```text
mistralai/Mistral-7B-Instruct-v0.2
```

The model is fine-tuned using LoRA-based parameter-efficient fine-tuning instead of full fine-tuning.

---

## Techniques Used

### Supervised Fine-Tuning

The model is trained on question-answer examples where the correct answer is known. The task is to predict the correct multiple-choice answer letter.

### PEFT / LoRA

The project uses **parameter-efficient fine-tuning** with **LoRA**. Instead of updating all model parameters, LoRA trains a smaller set of adapter weights.

LoRA is applied to attention and MLP projection layers such as:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### Unsloth

Unsloth is used to load and fine-tune the Mistral model efficiently. It helps reduce training overhead and enables faster LoRA-based fine-tuning.

### bitsandbytes

bitsandbytes is included as part of the memory-efficient LLM fine-tuning stack. In this notebook, the model is loaded with `load_in_4bit=False`, so the current run is not using 4-bit quantized loading, but bitsandbytes is included in the environment setup.

### Experience Replay

Experience replay is the main continual learning technique tested in this project.

In reinforcement learning, experience replay usually stores transitions such as:

```text
(state, action, reward, next_state)
```

In this project, experience replay is adapted to supervised LLM fine-tuning. The replay buffer stores old-domain Biology examples:

```text
Biology question -> correct answer
```

These examples are mixed with new-domain Physics examples during training. This helps the model continue learning Physics while being reminded of Biology.

---

## Results

| Phase | Physics Accuracy | Biology Accuracy |
|---|---:|---:|
| Baseline | 40.4% | 67.6% |
| After Physics Fine-Tuning | 41.4% | 59.5% |
| After Experience Replay | 52.5% | 66.7% |

---

## Key Findings

After Physics-only fine-tuning, Biology accuracy dropped from **67.6%** to **59.5%**, showing catastrophic forgetting.

After applying experience replay, Biology accuracy recovered to **66.7%**, close to the original baseline. Physics accuracy also improved from **41.4%** to **52.5%**.

This suggests that even a small fixed replay buffer can help reduce forgetting while still allowing the model to improve on the new domain.

---

## Evaluation Method

The notebook evaluates the model using logit-based multiple-choice answer selection.

Instead of generating long text, the model compares the logits for answer tokens:

```text
A, B, C, D
```

The answer with the highest score is selected as the model prediction.

This makes evaluation faster and more controlled for multiple-choice tasks.

---

## Training Configuration

Important training settings:

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
REPLAY_RATIO = 0.25
MAX_LENGTH = 512
```

LoRA configuration used in the notebook:

```python
r = 128
lora_alpha = 256
lora_dropout = 0.05
```

---

## Project Workflow

```text
1. Install dependencies
2. Load MMLU Physics and Biology subjects
3. Format examples into Mistral instruction format
4. Load Mistral-7B-Instruct using Unsloth
5. Apply LoRA adapters
6. Evaluate baseline Physics and Biology accuracy
7. Fine-tune on Physics only
8. Evaluate Physics and Biology again
9. Create replay dataset with Physics + Biology subset
10. Fine-tune with replay data
11. Evaluate final performance
12. Plot and save results
```

---

## How to Run

### 1. Install dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### 2. Open the notebook

Run the notebook in Google Colab or another GPU-enabled environment.

Recommended environment:

- GPU runtime
- Hugging Face access if required
- Google Drive mounted for saving outputs

### 3. Run all phases

The notebook runs the experiment in three stages:

```text
Baseline evaluation
Physics-only fine-tuning
Experience replay fine-tuning
```

---

## Repository Structure

```text
llm-finetuning-for-continual-learning/
├── README.md
├── notebooks/
│   └── catastrophic_forgetting_mmlu_unsloth.ipynb
├── results/
│   └── results.json
└── plots/
    └── catastrophic_forgetting_results.png
```

---

## Limitations

This project is an experimental prototype and has a few limitations:

- The replay buffer is fixed and simple.
- Only one replay ratio was tested.
- Biology replay examples are selected using a basic subset selection strategy.
- The experiment uses selected MMLU subjects rather than a larger continual learning benchmark.
- The evaluation focuses only on accuracy.
- More rigorous train, validation, and test separation would improve the experimental design.

---

## Possible Improvements

Future improvements could include:

- Testing multiple replay ratios such as 10%, 25%, and 50%.
- Randomly sampling Biology replay examples instead of selecting the first subset.
- Comparing experience replay with other continual learning methods.
- Adding per-subject accuracy analysis.
- Saving and loading LoRA adapter checkpoints separately.
- Evaluating on more MMLU domains.
- Adding a true replay buffer abstraction.
- Comparing against full fine-tuning or QLoRA.

A better replay subset selection would be:

```python
bio_subset = biology_train.shuffle(seed=SEED).select(range(n_bio))
```

This would make the replay buffer more representative.

---

## Main Takeaway

This experiment shows that catastrophic forgetting can occur during domain-specific LLM fine-tuning, even when using parameter-efficient methods like LoRA.

By using experience replay, the model receives reminders from the old Biology domain while learning the new Physics domain. This simple strategy helps recover Biology performance while improving Physics performance.

The core contribution of this project is not just fine-tuning an LLM, but demonstrating a continual learning technique for reducing forgetting during supervised LLM adaptation.

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
- Matplotlib
- Google Colab

---

## Author

Madhumitha Gannavaram
