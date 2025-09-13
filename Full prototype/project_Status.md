# Prototype Analysis Report — `Full prototype` directory

**Generated:** 2025-09-13  
**Repo inspected:** `/mnt/data/sih.-main/Full prototype`

---

## 1) Executive summary

- The repo contains a near-complete prototype for FRA AI: data ingestion, model fusion/training, model management, and a FastAPI backend.
- Critical missing pieces: **model weights** and **datasets** are not included. Scripts exist to download them, but they must be fetched.
- `fra_ai_complete_system.py` is a single-file orchestrator (setup/train/serve). It can operate in a demo/fallback mode but full functionality requires heavy downloads and dependencies.

---

## 2) Repo snapshot

- Folders: `1_data_processing/`, `2_model_fusion/`, `3_webgis_backend/`, `configs/`, `scripts/`.
- Notable files: `fra_ai_complete_system.py`, `run.py`, `main_fusion_model.py`, `scripts/download_models.py`, `scripts/download_data.py`, `requirements.txt`, `smoke_test.py`.

## 3) Project statement (short)

Digitize and standardize FRA legacy records, integrate them into a geospatial FRA Atlas with satellite-derived assets, provide an AI Decision Support System (DSS) for scheme recommendations and checks, and deliver multimodal OCR/NER and satellite analytics.

## 4) What is implemented (detailed)

- **Data pipeline**: `1_data_processing/data_pipeline.py` includes ETL, OCR hooks, raster processing, and DB initialization.
- **Model training/fusion**: `2_model_fusion/` has training and distillation scripts (Hugging Face Trainer + `accelerate`).
- **Model management**: `3_webgis_backend/model_weights_manager.py` describes expected weight paths (fusion, adapters, tasks).
- **Secure API**: `3_webgis_backend/secure_api_components.py` provides auth, rate-limiting, a `SecureModelManager`, and a `PlaceholderModel` fallback.
- **API server**: `3_webgis_backend/api.py` implements many endpoints (claims, satellite analysis, DSS, documents, analytics).
- **Single-file orchestrator**: `fra_ai_complete_system.py` is a CLI entrypoint (`--action [setup|train|serve|all]`).

## 5) Missing / Blockers

1. **No model weights included** — expected weights are referenced but not present. The `REQUIRED_MODELS` declared in the single-file include:

```
{
            'fusion_model': self.base_models_dir / "fusion" / "best_fusion_model.pth",
            'fusion_tokenizer': self.base_models_dir / "fusion" / "tokenizer",
            'fusion_config': self.base_models_dir / "fusion" / "config.json",
            
            # Individual model components
            'llama_adapter': self.base_models_dir / "components" / "llama_adapter.pth",
            'mistral_adapter': self.base_models_dir / "components" / "mistral_adapter.pth", 
            'falcon_adapter': self.base_models_dir / "components" / "falcon_adapter.pth",
            'trocr_finetuned': self.base_models_dir / "components" / "trocr_finetuned.pth",
            'layoutlm_finetuned': self.base_models_dir / "components" / "layoutlm_finetuned.pth",
            
            # Task-specific models
            'dss_model': self.base_models_dir / "tasks" / "dss_model.pth",
            'ner_model': self.base_models_dir / "tasks" / "ner_model.pth",
            'satellite_model': self.base_models_dir / "tasks" / "satellite_analysis.pth",
            'scheme_recommender': self.base_models_dir / "tasks" / "scheme_recommender.pth",
            
            # Distilled models (smaller, faster)
            'fusion_distilled': self.base_models_dir / "distilled" / "fusion_distilled.pth",
            'mobile_model': self.base_models_dir / "distilled" / "mobile_model.pth"
        }
```

2. **No datasets included** — `scripts/download_data.py` can fetch data, but you must provide/allow downloads.
3. **Large dependencies** — `requirements.txt` lists heavy packages (torch, transformers, bitsandbytes, faiss, rasterio, etc.).
4. **Licensing & availability** — some models (Llama 3.x) may be gated and require acceptance of model terms on Hugging Face.
5. **Small issues** — `requirements.txt` has a truncated line (`starle`), and `test_mock_replacement.py` uses a hard-coded path.

## 6) Single-file executability — verdict

- `fra_ai_complete_system.py` is a valid single-file orchestrator and *can* run in placeholder/demo mode.
- **Full production** (real LLM inference/training) is **not executable** until heavy model weights are downloaded and dependencies installed; also GPU resources will be required for serious workloads.

## 7) How to run a minimal demo (recommended)

1. Create venv and install minimal packages (FastAPI, uvicorn, torch CPU, pillow, numpy, requests).
2. Run `python smoke_test.py` to generate dummy input data.
3. Start the demo server:
   `python fra_ai_complete_system.py --action serve --host 127.0.0.1 --port 8000`
   Then visit `http://127.0.0.1:8000/docs`.

## 8) How to get the full system running (high level)

- Ensure sufficient disk (≥50GB) and GPU resources (varies by model).
- Use `scripts/download_models.py` to fetch required weights (Hugging Face token may be required).
- Install full `requirements.txt` (or run via Docker) and run `python fra_ai_complete_system.py --action setup` then `--action train`/`--action serve`.

## 9) Recommended quick fixes / improvements

- Add `requirements-min.txt` for demo mode.
- Fix the `requirements.txt` typo and hard-coded paths in tests.
- Add a `--demo` flag to `fra_ai_complete_system.py` to make placeholder mode explicit.
- Add a demo `REQUIRED_MODELS_DEMO` that points to small CPU-friendly models for onboarding.

## 10) Prioritized next steps

1. Create `requirements-min.txt` and a README example for demo run.
2. Verify `scripts/download_models.py` can fetch one small model and validate load/predict via `SecureModelManager`.
3. Remove hard-coded test paths and add simple CI smoke tests that run without heavy models.



---

If you want, I can:
- generate `requirements-min.txt` and the exact `pip` commands,
- prepare a short script showing how to download a tiny demo model and verify `model_manager.load_model()`,
- produce a TODO checklist file.

