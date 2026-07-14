# AEREAS Google Colab Notebooks

A collection of Jupyter notebooks designed to be run in Google Colab (or locally) for running end-to-end quality and scoring benchmarks directly in memory.

---

## Interactive Evaluation Suite

| Notebook Name | Description |
| :--- | :--- |
| [colab_argrewrite_scoring_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_argrewrite_scoring_evaluator.ipynb) | Evaluates scoring metrics specifically against the argument rewrite revisions dataset. |
| [colab_direct_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_direct_evaluator.ipynb) | Runs evaluation scripts directly on essays to check worker outputs and classifications. |
| [colab_direct_scoring_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_direct_scoring_evaluator.ipynb) | Evaluates overall scoring correlations against ASAP Prompt 1 gold labels. |
| [colab_llm_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_llm_evaluator.ipynb) | Checks accuracy, response validation rates, and error frequencies across different LLM backends/providers. |
| [colab_revision_scorer.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_revision_scorer.ipynb) | Computes improvement scores for student essay drafts (v1). |
| [colab_revision_scorer_v2.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_revision_scorer_v2.ipynb) | Optimized revision scoring pipeline implementing dynamic specialist weights and structural change detection (v2). |

---

## Direct Memory Execution Protocol

Unlike standard API client-server testing, these evaluation notebooks load `SupervisorAgent`, `LLMClient`, and the specialist workers directly into memory. This eliminates the need to run local HTTP backend servers, manage tunnels, or execute API requests.

### Steps to Run Evaluations

1. **Environment Setup:** Ensure that environment keys (such as `OPENROUTER_API_KEY` or `GEMINI_API_KEY`) are loaded or set up in your execution context/environment.
2. **Execute Evaluators:** Open the desired evaluator notebook (e.g., [colab_direct_scoring_evaluator.ipynb](file:///c:/Users/Lexman/Desktop/AEREAS/app/notebook/colab_direct_scoring_evaluator.ipynb)).
3. **Run Cells:** Run all cells in sequence. The notebook will automatically add the workspace root to `sys.path`, initialize the modules, pull the data from the local repository layout, and output the computed scores.
