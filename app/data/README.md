# AEREAS Evaluation Benchmarks & Datasets

A collection of evaluation corpora, gold-standard human scores, and target test outputs used to validate the accuracy, coherence, and plagiarism detection capabilities of the AEREAS supervisor and specialist workers.

---

## Repository Layout and File Index

```text
app/data/
├── argrewrite/
│   └── pairs.json                 # Argument rewrite pairs (draft2 vs. draft3 revisions)
└── asap/
    ├── training_set_rel3.tsv      # Raw ASAP student essays dataset (Prompt 1)
    ├── gold_prompt1.json          # Extracted human/gold scores indexed by essay ID
    ├── benchmark_prompt1.jsonl    # Pre-processed input JSONL for evaluation workers
    ├── benchmark_results_prompt1.json # Output predictions from model runs
    └── benchmark_report_prompt1.md # Formatted accuracy report comparing predictions to gold labels
```

---

## Corpus Histories and Specifications

### Argument Rewrite (argrewrite)
* **Underlying Files:** [pairs.json](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/argrewrite/pairs.json)
* **History & Origin:** The Argument Rewrite corpus is sourced from NLP and learning science research investigating how students revise their essays in response to automated or instructor feedback. Specifically, the data contains parallel student drafts (e.g., `draft2` and `draft3`) tracking structural transitions, claim modifications, and vocabulary adaptations.
* **Corpus Objective:** It serves as a benchmark for the revision engine and coherence worker to measure how well AEREAS can identify qualitative improvements between consecutive drafts of a single document.

### Automated Student Assessment Program (asap)
* **Underlying Files:**
  * [training_set_rel3.tsv](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/training_set_rel3.tsv) (Raw training essays)
  * [gold_prompt1.json](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/gold_prompt1.json) (Resolved human grades)
  * [benchmark_prompt1.jsonl](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/benchmark_prompt1.jsonl) (Evaluation format inputs)
* **History & Origin:** The Automated Student Assessment Program (ASAP) dataset was originally released in 2012 by the William and Flora Hewlett Foundation as part of a Kaggle competition. The goal was to demonstrate that automated scoring engines could match the reliability of human grading. Prompt 1 specifically consists of persuasive essays written by high school students arguing for or against the adoption of computers and technology in everyday lives, graded holistically by two human raters.
* **Corpus Objective:** It provides gold-standard benchmarks (with holistic scores ranging from 2 to 12) used to assess the scoring alignment of the AEREAS Supervisor Agent and individual specialized workers against real-world student submissions.

---

## Benchmarking Protocol and Execution Pipeline

To run an evaluation cycle using these datasets, perform the following sequence:

### 1. Execute Evaluators against Corpora
Since the evaluation suite runs directly in memory, there is no need to host a separate API backend.
* **For Essay Revisions:** Execute [colab_revision_scorer_v2.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_revision_scorer_v2.ipynb) to compute scoring differentials across `pairs.json` drafts.
* **For General Essay Scoring:** Execute [colab_direct_scoring_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_direct_scoring_evaluator.ipynb) to process the ASAP Prompt 1 dataset.

### 2. Compute Alignment and Compile Reports
The output predictions are saved in [benchmark_results_prompt1.json](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/benchmark_results_prompt1.json). The evaluators then calculate statistical correlation metrics against [gold_prompt1.json](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/gold_prompt1.json) and compile the performance results in [benchmark_report_prompt1.md](file:///c:/Users/Lexman/Desktop/AEREAS/app/data/asap/benchmark_report_prompt1.md).
