# Prototype Code Analysis — `sih_-main`

---

## 1) Executive summary

I read the **project statement** and the repository `sih_-main` (folder `Full prototype`) and fully scanned the code. In short:

* The repo contains a clear modular prototype for an FRA (Forest Rights Act) multimodal AI system: data ingestion + OCR/NER, a multimodal fusion model and staged training pipeline, and a FastAPI WebGIS backend for serving results.
* Core components (data pipeline, fusion model architecture, staged training loops, and a demo/CLI runner) are implemented and reasonably well-structured. Many pieces are *implemented as prototypes* or *mocked/stubbed* rather than end-to-end production-ready.
* **What’s missing** for the behaviour you asked for (automatic **download of LLMs / model weights**, automatic **dataset ingestion from links**, and fully automated **pretraining → fine-tuning → distillation → packaging**): automation scripts, credential handling, distributed/accelerated training wrappers, distillation pipeline, and orchestration/CI/docker files are *not present*.

This document lists what I found, where it lives in the repo, issues, and a prioritized actionable plan (including concrete code snippets / patterns you can use to add the automation you requested).

---

## 2) Project goals (short summary of the Problem Statement / README)

From `Full prototype/PROBLEM_STATEMENT.md` and `Full prototype/readme.md` the project aims to:

* Digitize & standardize FRA documents (OCR, NER), extract spatial and administrative metadata, combine with census / socio-economic and satellite data.
* Build a multimodal fusion model (images, text, spatial features, graphs) to support a Decision Support System (DSS) for FRA implementation and claims analysis.
* Provide a WebGIS backend to serve digitization results, automated suggestions for schemes, and allow downstream analytics.

These goals are reflected in the implemented modules (data processing, multimodal fusion, and a WebGIS FastAPI backend). Good alignment between problem and prototype.

---

## 3) Repo snapshot (key files / folders I inspected)

```
sih_-main/
  ├─ Full prototype/
  │   ├─ 1_data_processing/data_pipeline.py
  │   ├─ 2_model_fusion/train_fusion.py
  │   ├─ 3_webgis_backend/api.py
  │   ├─ main_fusion_model.py
  │   ├─ run.py
  │   ├─ demo.py
  │   ├─ configs/config.json
  │   └─ requirements.txt
  ├─ test/ (demo and unit-like quick scripts)
  └─ project_plan.md
```

(There are also macOS metadata files like `__MACOSX` and `.DS_Store`.)

---

## 4) What’s implemented (detailed)

### A. Orchestration / CLI

* `run.py` is a main runner with an `argparse` CLI. Capabilities found:

  * `--train`, `--serve`, `--complete`, `--status`, `--eval` (stubs implemented).
  * `run_training_pipeline()` imports `2_model_fusion/train_fusion.py` and initializes `EnhancedFRATrainingPipeline(config.config)`.
  * The script checks config paths (training data, checkpoint), warns and instructs the user when things are missing.

**Status:** basic CLI present and able to run local training/demo flows — but depends on local dataset and local checkpoints.

### B. Data processing / ingestion

* `1_data_processing/data_pipeline.py` implements: OCR (pytesseract + `LayoutLMv3Processor` usage), raster processing (rasterio), creation of training pairs, and sqlite-based joins to village metadata.
* Data config values are read from `configs/config.json` (paths like `./data/raw`, `./data/processed`).

**Status:** local data pipeline exists and transforms inputs into training-ready artifacts. There is **no code to automatically download remote datasets** (no link parsing, no `datasets.load_dataset(...)` from remote URLs, and no Google Drive / S3 download helpers). Also no automated validation of downloaded datasets.

### C. Model / Fusion architecture

* `main_fusion_model.py` contains `EnhancedFRAUnifiedEncoder` implementing multimodal fusion: text tokenizer (uses `AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")`), vision backbones (e.g., `deeplabv3_resnet50` usage), contrastive projection heads, DSS head and other components.
* The fusion model uses Transformers/AutoTokenizer and torchvision backbones where applicable.

**Status:** architecture is fairly sophisticated and designed for multimodal pretraining and downstream tasks. Many components (KG, GNN layers, memory module) appear implemented or scaffolded. Some functions are currently **mocked** (for example SQL generation is a placeholder) or rely on simple heuristics.

### D. Training pipeline

* `2_model_fusion/train_fusion.py` contains `EnhancedFRATrainingPipeline` with staged training: `stage_0_multimodal_pretraining`, `stage_1_foundation`, etc. Training loop uses `torch`, `AdamW`, and scheduling. Checkpoint save/load functions exist.
* The pipeline includes contrastive and masked-token pretraining objectives and saves checkpoints.

**Status:** the training logic and loss definitions exist. However:

* It assumes data is present locally (from `config.json`) and uses `DataLoader`s built from local processed files.
* There is **no orchestration to automatically download large LLM weights** or to coordinate heavy multi-GPU, nor any explicit use of `accelerate` launch or `torch.distributed` (though `accelerate` is in `requirements.txt`).

### E. Serving / WebGIS backend

* `3_webgis_backend/api.py` is a FastAPI application with many endpoints: digitization, OCR, NER, DSS queries, file uploads, and model-serving endpoints.
* The API code loads a checkpoint from a local path `../../2_model_fusion/checkpoints/final_model.pth` using `torch.load()` and `MODEL.load_state_dict(...)`. Many endpoints have example/mock logic and clear TODO notes.

**Status:** the server is implemented and runnable locally once a model checkpoint exists; many endpoints are *mock implementations* returning example results rather than real production-ready predictions.

### F. Demo & tests

* `demo.py` and `test/demo_fusion.py` provide runnable demonstrations, using small transformers models (e.g., `distilgpt2` and `microsoft/trocr-base-stage1`) in `test/` scripts. Good for sanity checks.

**Status:** demo scripts are helpful smoke-tests but are not full integration tests for the entire end-to-end pipeline.

### G. Configs and deps

* `configs/config.json` contains many required settings (training schedules, directories, model hyperparameters, API host/port, etc.). Good coverage but paths are local and no env-var/secret section is present.
* `requirements.txt` includes `torch`, `transformers`, `datasets`, `accelerate`, etc.

**Status:** dependencies are listed. Missing or recommended extras: `huggingface_hub` (for robust model download & caching), `bitsandbytes` (for 8-bit training/quantization), `peft` (for parameter-efficient fine-tuning), and optionally `wandb` or `tensorboard` for logging.

---

## 5) What is missing / incomplete (explicit list tied to your request)

1. **Automatic model weight download & management**

   * No script or config-driven mechanism to download large LLMs or HF model weights (e.g. Mistral/Falcon/LLaMA) to disk or to cache them. The code uses `AutoTokenizer.from_pretrained(...)` which will try to download on demand, but:

     * There's no management of HF tokens, rate limits, or private repo access.
     * No prefetching/verification step, no retry/backoff, no support for alternate sources (S3, GDrive).
     * No explicit support for large-model strategies (8-bit loading, device\_map, bitsandbytes, or model sharding).

2. **Dataset ingestion from remote links**

   * `config.json` expects local path strings. There is **no** `scripts/download_data.py`, nor code to call `datasets.load_dataset()` with provided URLs.
   * No support for Google Drive / S3 / HTTP / zipped archives.

3. **Automated end-to-end orchestration (download → preprocess → train → finetune → distill → package)**

   * The training pipeline is staged but there is no top-level orchestrator that automatically runs: download models, download datasets, prepare data, run pretraining, run fine-tuning, run distillation, build a deployable artifact, and push to a model store.

4. **Distillation / Model compression**

   * There is no concrete teacher-student distillation stage implemented (no KL-based distillation loss and no student model instantiation). The pipeline stages are there conceptually but missing actual distillation code.

5. **Large-model training practices**

   * No usage of `accelerate` / `torchrun` / device\_map examples in the training code. The repo lists `accelerate` but does not show example `accelerate` configs or launch commands.
   * No `bitsandbytes`/8-bit support for large LLMs to reduce memory footprint.

6. **Authentication & secrets**

   * HF token management (environment variables, `.env`, or vault) is missing. Required to download private/large models.

7. **Resuming / checkpoint orchestration**

   * While checkpoints are saved, there is limited resume orchestration across stages (i.e., resuming training from a partial checkpoint into a new stage, or automated conversion of checkpoint to quantized/packaged model).

8. **Productionization**

   * Missing Dockerfile / docker-compose and minimal infra-as-code for cloud runs. API has example `uvicorn.run(...)` but lacks production configuration (Gunicorn or process manager, logging/metrics, health checks).

9. **Testing / CI / validation**

   * No unit tests for training steps, model loading, or API endpoints. The `test/` directory is only quick scripts.

10. **Documentation & runbook**

* README exists but needs explicit step-by-step instructions for: required compute, how to fetch models/datasets, how to run distributed training, and how to reproduce demo results.

11. **Mocked endpoints**

* Several API endpoints are placeholders and need real model invocation & secure input validation.

12. **Licensing & model compliance checks**

* If you plan to download third-party LLMs, add a step to check license compatibility and display license/attribution.

---

## 6) Concrete actionable fixes & code snippets (copyable)

Below are *concrete* changes you can drop into the repo to add the automation you asked for. I prioritized reliability and minimal intrusion into existing code.

### A. Add `huggingface_hub` and a `scripts/download_models.py`

**Install extras**: add `huggingface_hub` and `tqdm` to `requirements.txt`.

**Example: `scripts/download_models.py`** (saves models to `models/` and supports a list in `config.json`):

```py
# scripts/download_models.py
import os
from huggingface_hub import hf_hub_download, snapshot_download
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='Full prototype/configs/config.json')
args = parser.parse_args()

cfg = json.load(open(args.config))
models = cfg.get('model_sources', {
    'layoutlm': 'microsoft/layoutlmv3-base',
    'trocr': 'microsoft/trocr-base-stage1'
})

hf_token = os.environ.get('HF_TOKEN')
for name, repo in models.items():
    print(f"Downloading {name}: {repo}")
    # snapshot_download will fetch the model repo to local cache dir
    local_dir = snapshot_download(repo_id=repo, use_auth_token=hf_token)
    print('Saved to', local_dir)
```

**Add to `config.json`** (example):

```json
"model_sources": {
  "layoutlm": "microsoft/layoutlmv3-base",
  "trocr": "microsoft/trocr-base-stage1",
  "brain": "distilgpt2"
}
```

This gives: a repeatable, auditable model download step (and handles private repos if `HF_TOKEN` exported).

---

### B. Add `scripts/download_data.py` (support http, s3, gdrive and `datasets`)

```py
# scripts/download_data.py
import os, argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-url', required=True, help='URL or dataset identifier')
parser.add_argument('--out-dir', default='data/raw')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
# If the link looks like a Hugging Face dataset id: load_dataset
if '/' in args.data_url and not args.data_url.startswith(('http://','https://','s3://','gs://','drive:')):
    ds = load_dataset(args.data_url)
    ds.save_to_disk(args.out_dir)
else:
    # for HTTP/zip: download & extract (implement gdown for drive links)
    import requests, zipfile, io
    r = requests.get(args.data_url, stream=True)
    r.raise_for_status()
    # ... handle content-type, write files
    print('Downloaded', args.data_url)
```

Add helpers for `gdown` (Google Drive) and `boto3` (S3) if required.

---

### C. Wire `run.py` to support `--download-models` and `--download-data`

Add two flags and call the scripts above from `run.py` before `trainer.train_enhanced_pipeline(...)`. Keep resumability: check if downloaded files already exist.

---

### D. Fine-tuning & Distillation recipe (sketch)

Add a `2_model_fusion/distillation.py` that:

* Loads a large `teacher` (AutoModelForCausalLM.from\_pretrained(teacher\_id, device\_map='auto')).
* Builds a smaller `student` model.
* Implements a distillation training loop with a KLDivLoss between `teacher_logits` and `student_logits` (use temperature τ), plus optional supervised loss on labels.

Minimal snippet:

```py
from torch.nn import KLDivLoss
T = 2.0
kl = KLDivLoss(reduction='batchmean')
# teacher_logits, student_logits shape: (batch, seq, vocab)
loss = kl(F.log_softmax(student_logits / T, dim=-1), F.softmax(teacher_logits / T, dim=-1)) * (T * T)
```

You can insert this stage into `EnhancedFRATrainingPipeline` as `stage_distill()` and save `student` checkpoints.

---

### E. Accelerate / bitsandbytes support (for large models)

* Add `from accelerate import init_empty_weights, load_checkpoint_and_dispatch` patterns or instruct users to run with `accelerate launch`.
* Add optional `--quantize` flag that runs a post-training quantization script (e.g. use `bitsandbytes` + `transformers` support for `load_in_8bit=True`).

Example quick loader:

```py
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)
```

---

### F. Add robust logging & experiment tracking

Add `wandb` or `tensorboard` to track stage-by-stage metrics and checkpointing. This helps debugging and model selection for distillation.

---

### G. Dockerfile & deployment

Provide `Dockerfile` using NVIDIA CUDA base images if GPU required. Add a `docker-compose` for the API with mounted models and a persistent `data` volume.

**Example:** `FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime` and install requirements + tesseract system package.

---

## 7) Prioritized roadmap (recommended order)

**Immediate (hours — 1–2 day effort to implement):**

1. Add `scripts/download_models.py` + update `config.json` with `model_sources`.
2. Add `scripts/download_data.py` for `datasets.load_dataset` + HTTP/GDrive/S3 helpers.
3. Add `--download-models` and `--download-data` flags to `run.py` to call these scripts.
4. Add `HF_TOKEN` usage notes in README and check for environment on startup.

**Next (days — integrate training automation):**

1. Wire downloads → preprocess → training in `run.py --complete`.
2. Add `accelerate` friendly launch instructions and example `accelerate` config.
3. Implement a basic distillation stage (`2_model_fusion/distillation.py`) and integrate it as a pipeline stage.
4. Add model quantization option and small packaging script that outputs a `deployed_model/` folder.

**Productionization (weeks — robustness & infra):**

1. Add Dockerfile(s), CI pipeline, unit tests for each pipeline step, and monitoring (Prometheus/Grafana or logging pipeline).
2. Add secrets management and license checks for downloaded models.
3. Add multi-node distributed training support if needed.

---

## 8) Quick checklist (what to commit next)

* [ ] `scripts/download_models.py` + doc in README
* [ ] `scripts/download_data.py` + doc
* [ ] Update `configs/config.json` to include `model_sources` and `dataset_links`
* [ ] Extend `run.py` to orchestrate downloads and pipeline stages
* [ ] Add distillation stage script and integrate into training pipeline
* [ ] Add HF\_TOKEN checks and `huggingface_hub` to requirements
* [ ] Add Dockerfile with GPU support, or example `accelerate` command in README
* [ ] Add smoke tests to `test/` that run only on small models

---

## 9) Notes, caveats & blockers

* **Large LLM downloads and fine-tuning require significant GPU memory and disk space.** If you plan to use 7B–70B models, you must choose a strategy (sharding, 8-bit, offloading, or use hosted services). The repo currently does not include those strategies.
* **Licensing**: confirm license of each pretrained model and whether downstream usage is permitted.
* **Access**: private HF repos require `HF_TOKEN` and rate-limit handling.

---

## 10) Offer — how I can help next

I prepared a prioritized and concrete list above. If you want, I can:

* Create `scripts/download_models.py` and `scripts/download_data.py` and a `run.py` patch to wire them into the pipeline.
* Implement a minimal distillation stage (teacher→student) that can run on a single GPU for small-scale testing.
* Prepare a Dockerfile and an `accelerate` example config.

Tell me which of the above I should implement first (model downloads, data downloads, or distillation) and I will add the code directly to the repo.

---

*If you want the full file-by-file annotated report (line pointers for TODOs and exact code locations I referenced), I can produce that next — I kept this document focused on actionable items. The repository's README and problem statement were read and used as the baseline for this analysis.*
