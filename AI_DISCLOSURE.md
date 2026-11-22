# AI disclosure

## AI used to build this submission

This project was built with **Claude Code** (Anthropic), an AI coding assistant, working under the direction of the
team member (Ayush Kulshreshtha). It was used for:

* writing the source code (`code/`, `tools/`, `kaggle/`, `.github/workflows/`), the Dockerfile and `run.sh`
* setting up and running the data downloads and the batch runs that produced `outputs/`
* writing the README, this file, the presentation (`docs/deck/build_deck.py`) and the demo video script
* assembling the demo video (`docs/demo/make_demo.py`); the narration is synthesised offline with the Windows
  speech synthesizer ("Microsoft David" voice)

The team member chose the approach, set the requirements and integrity rules, and reviewed the results.

## AI models inside the submitted system

The pipeline itself uses two pretrained third-party models at inference time. There is no training or fine-tuning
and no generative model:

| Model | Role | Source |
|---|---|---|
| TSM ResNet-50, EPIC-KITCHENS-100 RGB checkpoint | verb/noun action recognition | EPIC-KITCHENS team, CC BY-NC 4.0 |
| OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) | zero-shot video label, frame interestingness | OpenCLIP / LAION |

## Integrity

* Every JSON in `outputs/` was produced by the submitted code (`run.sh` / `code/infer.py`). None was written by
  hand or by an AI assistant.
* No ground truth was used, and no video IDs or file-name categories are read by the inference code.
* The two FPVSum videos that are no longer on YouTube have no output (see `logs/failures.txt`).
* All numbers reported in the README, deck and video were measured on our own runs.
