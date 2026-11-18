# Project_AI_for_mental_healt

Local, privacy-first NLP assistant to support the screening of mood disorders from typed text and handwritten documents (OpenCV-based OCR). Academic research project, LGPD-compliant. Not a diagnostic tool.

The prototype runs **100% locally**, in Python. It analyzes **typed text** and **handwritten documents** (cleaned up with OpenCV and read by OCR) and suggests **mood disorder hypotheses**, with the supporting evidence for each one, for a **mental health professional** to assess.

> ⚠️ **Disclaimer:** this tool offers **clinical decision support, not a diagnosis**. Every hypothesis it generates must be assessed by a psychologist or psychiatrist. The system has not been clinically validated and is not registered as medical software with ANVISA, Brazil's health regulatory agency (RDC 657/2022).

## Status

Work in progress. The repository skeleton, the Python environment and the privacy guards are ready. The input pipeline (OpenCV preprocessing, OCR and anonymization) comes next; there is no analysis code yet.

## How it works

1. **Input:** typed text, or an image or PDF of a handwritten document.
2. **Handwriting:** OpenCV cleans up the image and an OCR model reads the handwriting; the professional reviews the recognized text.
3. **Preparation:** language detection (Portuguese or English) and anonymization.
4. **Specialists:** BERT-like classifiers estimate the probability of each class (multi-label, because comorbidity is common).
5. **Debate:** two local LLMs debate the case. One defends the main hypothesis, the other argues for the differential diagnosis, and every argument must quote the text.
6. **Report:** ranked hypotheses, evidence, DSM-5-TR and ICD-11 criteria, points of agreement and disagreement, and low-confidence warnings.

Classes: no signs (control), depression, bipolar disorder, anxiety (as a differential diagnosis), dysthymia and cyclothymia. Dysthymia and cyclothymia have no labeled dataset, so they appear only in the LLMs' reasoning.

## Privacy (LGPD)

Under Brazil's General Data Protection Law (LGPD, Law No. 13,709/2018, art. 5, II, and art. 11), health data is sensitive personal data. Therefore:

- **Nothing leaves the machine:** analysis runs offline, with no cloud APIs and no telemetry.
- **No cloud-synced folders:** never keep the project, datasets or outputs in folders synced by OneDrive, Google Drive or Dropbox; syncing would send the data off the machine.
- **No data in the repository:** datasets, handwriting images, analyzed texts, model outputs and model weights are kept out of git.
- **Nothing is saved by default:** no logs with analyzed text; temporary files are deleted at the end.
- **Anonymization before analysis:** names, CPF numbers (the Brazilian taxpayer ID), phone numbers, e-mail addresses, street addresses and dates are removed or replaced.
- **Public research datasets only**, and **synthetic** images in tests.

## Repository structure

Folders are created as the project progresses.

```
├── README.md
├── LICENSE        # MIT
├── configs/       # YAML configuration (models, OCR, debate, experiments)
├── prompts/       # versioned LLM prompts
├── src/moodai/    # source code
├── scripts/       # training, batch evaluation and utilities
├── tests/         # automated tests (made-up data only)
├── data/          # not in git: datasets
├── models/        # not in git: model weights
└── outputs/       # not in git: experiment results
```

## Installation

You need [uv](https://docs.astral.sh/uv/), which also installs the right Python version (3.12) for the project. On Windows: `winget install --id astral-sh.uv -e`.

```
git clone https://github.com/LuizP-04/Project_AI_for_mental_healt.git
cd Project_AI_for_mental_healt
uv sync                              # creates .venv with the exact versions in uv.lock
uv run python -m pre_commit install  # turns on the checks that run before every commit
```

Keep the clone in a local folder that is not synced with the cloud (see Privacy). pre-commit is called through Python (`python -m pre_commit`) because Windows Smart App Control can block its unsigned `pre-commit.exe` launcher.

## Usage

| Command | What it does |
|---|---|
| `uv run python scripts/check_env.py` | Checks that this machine is ready and safe for the project |
| `uv run pytest` | Runs the tests, with network access blocked |
| `uv run ruff check .` | Lints the code |
| `uv run ruff format .` | Formats the code |
| `uv run python scripts/check_no_data.py --all` | Confirms that git tracks no data file |
| `uv run python -m pre_commit run --all-files` | Runs every pre-commit check on the whole repository |

## Keeping data out of git

Three layers stop data files from reaching the repository:

1. `.gitignore` ignores the data folders (`data/`, `models/`, `outputs/`) and data, weight, image, audio and archive file types.
2. A pre-commit hook runs `scripts/check_no_data.py` before every commit. It also catches files forced in with `git add -f` and files larger than 1 MB.
3. The CI runs the same check on every pull request and on every push to `main`, together with the linter and the tests.

## License

The code is released under the [MIT](LICENSE) license. The datasets and models used have their own licenses and terms of use and are **not** distributed in this repository.
