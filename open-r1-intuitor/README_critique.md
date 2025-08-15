# Critique Step Implementation

This document explains the changes made to implement the critique step after the generation step in the INTUITOR training pipeline.

## Overview of the Training Process

The enhanced training process now follows this flow:
1. Initial user prompt
2. Initial model generation
3. Critique prompt from user
4. Critique generation from the model
5. New response generation if required based on the critique

## Key Changes Made

### 1. In `_generate_and_score_completions` method:

- **Initial Generation**: The original code generates initial completions based on user prompts.
- **Critique Generation**: Added code to generate critiques for the initial completions.
- **Revision Detection**: Implemented a simple heuristic to detect if a revision is needed based on keywords in the critique.
- **Revised Generation**: For samples that need revision, generate a revised response based on the critique.
- **Self-Certainty Advantage**: Calculate self-certainty advantage for all three steps (initial, critique, revised).
- **Combined Advantage**: For samples that need revision, combine the advantages from initial and revised completions with a weighted approach (30% initial, 70% revised).

### 2. Metrics and Logging:

- Added metrics to track the critique process:
  - Revision needed ratio
  - Critique advantage metrics
  - Revised advantage metrics
- Added textual logs for critiques and revised completions
- Updated wandb logging to include critique and revised completion texts

## Implementation Details

### Critique Generation

After generating the initial completions, we create critique prompts by adding the initial completion and a critique instruction to the original prompt:

```python
critique_prompt = prompt + "\n\nInitial response: " + completion + "\n\nPlease critique this response."
```

### Revision Detection

We use a simple heuristic to detect if a revision is needed based on keywords in the critique:

```python
needs_revision = [("improve" in critique.lower() or "revise" in critique.lower() or 
                  "incorrect" in critique.lower() or "error" in critique.lower()) 
                  for critique in critique_completions_text]
```

### Revised Generation

For samples that need revision, we create revision prompts that include the original prompt, initial completion, critique, and a revision instruction:

```python
revision_prompt = (prompt + "\n\nInitial response: " + completion + 
                  "\n\nCritique: " + critique + 
                  "\n\nPlease provide a revised response based on the critique.")
```

### Combined Advantage Calculation

For samples that need revision, we combine the advantages from initial and revised completions:

```python
final_sce_advantage_raw[i] = 0.3 * sce_advantage_raw[i] + 0.7 * revised_sce_advantage_raw[i]
```

## Usage

No changes are needed to the user-facing API. The critique step is automatically integrated into the training process.