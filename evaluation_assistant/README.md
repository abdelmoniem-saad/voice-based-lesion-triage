# Evaluation desk

A read-only reviewer for the frozen Milestone 2 results. A local language model routes a free-form question to one evidence tool or abstains. The controller adds required context, then stops and displays the evidence. The LLM never authors a metric, selects a threshold, trains a classifier, or reads row-level data.

## Start on Windows

From the Milestone 1 project directory, using your existing Python:

    python evaluation_assistant/check_review.py
    python evaluation_assistant/app.py

Open http://127.0.0.1:8765 in your browser. Try a **guided** button first. This checks the evidence path without any LLM installation. Use Ctrl+C in the server terminal to stop. No pip install is required.

For free-form questions, install [Ollama for Windows](https://ollama.com/download/windows) if necessary. Open Ollama, then in a second terminal:

    ollama pull qwen2.5:3b
    ollama list

If the Ollama service is not already running, run `ollama serve` in its own terminal. The named model is about 1.9 GB. First inference can be slower while the model loads. Use a local installed model; do not use a cloud-model tag. To use a different installed local model, pass the same `--model NAME` to the app and the live checks. The app uses the [local structured-output chat API](https://docs.ollama.com/capabilities/structured-outputs). No API key needed.

## Components and design choices

1. **Browser:** separate free-form LLM and guided controls. Shows the split, result, source filename, row selector and actual tool trace. Only the last three questions and their tool IDs are sent as context; reload or clear history to reset. No conversation database.
2. **Controller (app.py):** one schema-constrained routing decision, at most two evidence tools. Holdout comparisons and threshold questions automatically include cohort context. Known unsupported requests for attention holdout scores or sex-stratified metrics can be declined before inference. Invalid output or an unavailable service returns an error; there is no silent switch to guided mode. The original multi-call planner is documented in live_results.json: it often chose the right first tool and then drifted.
3. **Evidence layer (evidence.py):** seven pure, read-only tool handlers over an explicit allowlist of seven CSV/JSON summaries. It selects identified rows, validates expected run settings and formats values directly. It cross-checks the primary AP delta against the paired JSON. Source links expose only the approved aggregate files. No shell, arbitrary paths, web searches, audio or per-session predictions are available to the model.
4. **Answer policy:** evidence cards, not open-ended numerical prose. Questions supported only in development must not be answered as holdout findings. A refusal means the question is outside this catalog; it does not imply a measurement cannot exist elsewhere. Incorrect tool routing is still possible: a factually correct card can be irrelevant. The live acceptance gate specifically tests this risk.

## Validation and stopping rule

In another PowerShell terminal from the project directory:

    python evaluation_assistant/check_review.py --live --report evaluation_assistant/live_results.json

Use a fresh report filename on a repeat. Offline checks compare numerical anchors and sources to the frozen experiment; a scripted test double checks controller behavior. Live checks send ten questions to the actual LLM, including paraphrases, false premises, two follow-ups, an injection attempt and unsupported requests. They score required tool selection/refusal and retain the complete answers, traces and elapsed time. A backend outage is an incomplete run, not a model pass. Inspect the saved answers as well as the pass count.

**Demo gate:** show the agent as working only if all ten live cases pass and a human verifies the question was actually answered, the split and citations are right, and the source language is understandable. Then ask two new peer-written questions without editing the prompt first. Record failures honestly. This small suite is an engineering acceptance check, not an estimate of broad language-model accuracy. Do not tune the voice classifier or reopen the holdout in response to these results.

## 45-second presentation demo

1. In the text box ask: “Did we improve enough to replace the authors' reference?” Open the paired interval source and tool trace.
2. Follow up: “Did the 80% sensitivity target carry over to test?” Point out the change from development to holdout.
3. Ask: “What was sensitivity separately for women?” It should decline: demographic-input ablations are not subgroup evaluations.

Describe this as a local LLM **evaluation assistant** with bounded tool use. Its contribution is evidence retrieval and traceability, not improvement to lesion classification or clinical decision support. If live checks fail before the recording, the guided evidence viewer is a separate deterministic demo; label it accordingly.
