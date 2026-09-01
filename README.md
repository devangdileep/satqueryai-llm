# SatQuery AI Backend

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pydantic-v2-e91e63.svg" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/Pytest-Passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/Domain-Remote%20Sensing%20%26%20EO-orange.svg" alt="Domain">
  <img src="https://img.shields.io/badge/ISRO-SIH%202025%20%7C%20PS--26167-red.svg" alt="SIH 26167">
</p>

> **An Agentic Multimodal Remote-Sensing Intelligence Platform**  
> An interactive vision-language assistant for analyzing single, bi-temporal, and co-registered optical/multispectral & SAR satellite imagery through natural-language queries. Built for **ISRO SIH Problem Statement 26167**.

---

## 📌 Executive Summary

Remote sensing satellite data analysis traditionally requires specialized GIS expertise, manual band math, complex spatial software (QGIS/ArcGIS), and domain-specific knowledge. Generic Vision-Language Models (VLMs) fail at remote sensing tasks due to lack of spatial grounding, inability to process multispectral & SAR sensor physics, and lack of temporal change reasoning.

**SatQuery AI** bridges this gap using a **Capability-Driven Agentic System**. Rather than applying a single monolithic VLM or naive prompt chaining, SatQuery AI dynamically orchestrates specialized Earth Observation models, validates imagery metadata, extracts visual difference tokens, performs physics-aware SAR scattering interpretation, and returns grounded, evidence-backed answers with observable execution traces.

---

## 🌟 Key Differentiators

SatQuery AI incorporates 7 core architectural innovations designed specifically for satellite intelligence:

| # | Differentiator | Architectural Component | Value Proposition |
|:---|:---|:---|:---|
| **1** | **BigEarthNet.txt Fine-Tuned Multi-Sensor VLM Adapter** | [`BigEarthNetVLMAdapter`](app/models/bigearthnet_vlm/adapter.py) | Fine-tuned on 464k Sentinel-1 SAR + Sentinel-2 optical pairs with 9.6M text instructions for joint multi-sensor VQA. |
| **2** | **Visual Difference Token Projection (DeltaVLM VDPM)** | [`visual_difference_projection`](app/tools/difference_tokens.py) | Extracts explicit differential feature tokens ($F_{diff}$) for bi-temporal change reasoning instead of naive raw image concatenation. |
| **3** | **Physics-Aware SAR Encoding (HCoT)** | [`sar_physics_reasoning`](app/tools/sar_physics.py) | Interprets polarimetric scattering mechanisms (double-bounce = buildings, specular = water, volume = vegetation) via Hierarchical Chain-of-Thought. |
| **4** | **Cross-Modal Verification Engine** | [`CrossModalVerifier`](app/evidence/verification.py) | Verifies agreement between optical reflectance and SAR backscatter. Automatically degrades confidence when signals contradict. |
| **5** | **Mandatory Claim-Evidence Linking (VisTA / QAG-360K)** | [`EvidenceEngine`](app/evidence/engine.py) | Enforces that every generated visual claim is strictly linked to a bounding box/segmentation mask artifact, confidence score, and source tool. |
| **6** | **Observable Agentic Orchestration** | [`SatQueryAgent`](app/agent/agent.py) | Capability-driven model routing producing step-by-step observable execution traces without chain-of-thought exposure. |
| **7** | **ISRO Sensor Compatibility (Cartosat-2S + RISAT)** | [`isro_sensor_alignment`](app/tools/isro_compat.py) | Resamples 0.65m Cartosat-2S optical panchromatic to RISAT SAR grid extents and maps RISAT C-band polarizations (HH, HV, VV, VH). |

---

## 🏗️ System Architecture

```
User Query + Satellite Imagery
             │
             ▼
      [FastAPI Backend] ──> CORS / OpenAPI Documentation (/swagger)
             │
      [SatQueryAgent Orchestrator]
             │
 ┌───────────┴────────────────────────────────────────┐
 │ 1. Query Analyzer       (Groq / Pydantic JSON)     │
 │ 2. Input Validator      (Rasterio / PIL Metadata)  │
 │ 3. Modality Classifier  (Optical, Multispectral, SAR)│
 │ 4. Model Selector       (Capability-driven Registry)│
 │ 5. Workflow Planner     (Explicit Execution Plan)  │
 │ 6. Workflow Executor    (Observable Step Tracing)  │
 └───────────┬────────────────────────────────────────┘
             │
             ▼
   [Specialist Model Adapters] ──(HTTP / httpx)──> [Hosted Model APIs]
   - GeoChat (Single-Image VQA & Grounding)        (vLLM / HuggingFace
   - ChangeChat (Bi-temporal Change VQA)           Endpoints / Microservices)
   - Prithvi-EO-2.0 (Multispectral Foundation)
   - SAR-ML-Fusion (Optical + SAR Joint Reasoning)
   - BigEarthNet-VLM (Multi-sensor VLM)
             │
             ▼
 ┌───────────┴────────────────────────────────────────┐
 │ 7. Cross-Modal Verifier (Confidence Degradation)  │
 │ 8. Evidence Engine      (VisTA BBoxes & Masks)     │
 │ 9. Answer Composer      (Grounded Synthesis)       │
 └───────────┬────────────────────────────────────────┘
             │
             ▼
     [FastAPI Response JSON]
(Answer, Confidence, Evidence, Execution Trace, Artifacts)
```

---

## 📁 Repository Structure

```
satquery-backend/
├── app/
│   ├── main.py                  # FastAPI Application Factory & OpenAPI Spec
│   ├── agent/                   # Agentic Core Pipeline
│   │   ├── agent.py             # SatQueryAgent Orchestrator
│   │   ├── query_analyzer.py    # LLM Task Analysis (Pydantic JSON)
│   │   ├── model_selector.py    # Capability-driven Model Selection
│   │   ├── planner.py           # Explicit Workflow Planner
│   │   └── workflow.py          # Workflow Execution & Tracing
│   ├── api/                     # REST API Endpoints
│   │   └── routes/
│   │       ├── analyze.py       # POST /api/v1/analyze
│   │       ├── jobs.py          # GET /api/v1/jobs/{job_id} & /trace & /evidence
│   │       ├── models.py        # GET /api/v1/models
│   │       └── health.py        # GET /health
│   ├── models/                  # Specialist Remote Sensing Model Adapters
│   │   ├── registry.py          # Central Capability-Driven Model Registry
│   │   ├── base.py              # RemoteSensingModel Interface
│   │   ├── geochat/             # GeoChat VQA & Grounding Adapter
│   │   ├── changechat/          # ChangeChat Bi-temporal Change Adapter
│   │   ├── prithvi/             # Prithvi-EO-2.0 Foundation Model Adapter
│   │   ├── sar_fusion/          # SAR-ML-Fusion Adapter
│   │   └── bigearthnet_vlm/     # BigEarthNet.txt Multi-Sensor VLM Adapter
│   ├── tools/                   # Geospatial & Analytical Tools
│   │   ├── registry.py          # Decorator-based Tool Registry
│   │   ├── validation.py        # Rasterio/PIL Image Validation
│   │   ├── metadata.py          # Modality & Sensor Classification
│   │   ├── grounding.py         # Visual BBox Overlay Renderer
│   │   ├── evidence.py          # Change Map & Comparison Artifact Generator
│   │   ├── sar_physics.py       # HCoT Physics SAR Scattering Reasoning
│   │   ├── difference_tokens.py # DeltaVLM VDPM Feature Delta Extractor
│   │   └── isro_compat.py       # Cartosat-2S + RISAT Sensor Alignment
│   ├── evidence/                # Intelligence Layer
│   │   ├── engine.py            # VisTA Mandatory Claim-Evidence Linking
│   │   ├── confidence.py        # Multi-factor Confidence Estimation
│   │   └── verification.py      # Cross-Modal Optical/SAR Verification
│   ├── llm/                     # Structured LLM Providers
│   │   ├── groq.py              # Groq Async Provider
│   │   └── mock.py              # Mock Provider for Offline Dev
│   ├── core/                    # Application Configuration & Security
│   └── services/                # Job Storage & File Retention
├── tests/                       # Complete Test Suite (100% Pass)
│   ├── unit/                    # Task Routing & Model Selection Tests
│   └── integration/             # End-to-End FastAPI Integration Tests
├── pyproject.toml               # Python Dependencies & Metadata
├── .env.example                 # Configurable Environment Template
└── README.md                    # Project Documentation
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.11+**
- Virtual environment (`venv` or `conda`)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/devangdileep/satqueryai-llm.git
cd satqueryai-llm

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR install via pyproject.toml
pip install -e .
```

### 3. Environment Configuration

```bash
cp .env.example .env
```

Key environment options in `.env`:

```ini
# Backend mode: 'mock' (offline dev/testing without GPU) or 'http' (hosted endpoints)
MODEL_BACKEND=mock

# Groq API Configuration (for structured LLM routing)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# Hosted Model HTTP Endpoints (used when MODEL_BACKEND=http)
GEOCHAT_ENDPOINT=http://localhost:8001/v1/predict
CHANGECHAT_ENDPOINT=http://localhost:8002/v1/predict
PRITHVI_ENDPOINT=http://localhost:8003/v1/predict
SAR_FUSION_ENDPOINT=http://localhost:8004/v1/predict
```

### 4. Running the Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Swagger Documentation**: [http://localhost:8000/swagger](http://localhost:8000/swagger)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: `curl http://localhost:8000/health`

---

## 🧪 Running Tests

The test suite validates query analysis, capability-driven model selection, input validation, and end-to-end job execution flows.

```bash
pytest tests/ -v
```

```text
tests/integration/test_api.py::test_health_endpoint PASSED               [ 10%]
tests/integration/test_api.py::test_list_models_endpoint PASSED          [ 20%]
tests/integration/test_api.py::test_single_image_vqa_flow PASSED         [ 30%]
tests/integration/test_api.py::test_bitemporal_change_detection_flow PASSED [ 40%]
tests/unit/test_model_selection.py::test_capability_based_model_selection_geochat PASSED [ 50%]
tests/unit/test_model_selection.py::test_capability_based_model_selection_changechat PASSED [ 60%]
tests/unit/test_model_selection.py::test_capability_based_model_selection_sar_fusion PASSED [ 70%]
tests/unit/test_query_routing.py::test_query_routing_change PASSED       [ 80%]
tests/unit/test_query_routing.py::test_query_routing_grounding PASSED    [ 90%]
tests/unit/test_query_routing.py::test_query_routing_sar_fusion PASSED   [100%]

======================== 10 passed in 0.04s =========================
```

---

## 🛰️ Supported Tasks & Benchmarks

SatQuery AI natively supports and evaluates against the key remote-sensing benchmarks specified by ISRO:

| Task Category | Benchmark | Inputs Accepted | Primary Models / Tools |
|:---|:---|:---|:---|
| **Multi-Sensor Adaptation** | **BigEarthNet.txt** (arXiv:2603.29630) | Sentinel-1 SAR + Sentinel-2 Optical | `BigEarthNet-VLM` |
| **Single-Image VQA** | **VRSBench** / **RSVQA** | Optical / Multispectral GeoTIFF / PNG | `GeoChat`, `Prithvi` |
| **Region Grounding** | **VRSBench** | Optical / Multispectral | `GeoChat`, `region_grounding` |
| **Bi-Temporal Change VQA** | **CDVQA** | T1 & T2 Image Pair | `ChangeChat`, `visual_difference_projection` |
| **Optical + SAR Fusion** | **ISRO Cartosat-2S + RISAT** | Co-registered Optical & SAR | `SAR-ML-Fusion`, `sar_physics_reasoning` |

---

## 📡 API Reference

### 1. Submit Analysis Request
`POST /api/v1/analyze` (multipart/form-data)

**Parameters:**
- `query` (text): Natural language remote sensing query (e.g. *"What changed around the settlement between these two images?"*)
- `images` (files): One or more satellite images (`.tif`, `.tiff`, `.png`, `.jpg`)
- `metadata` (optional JSON string): Sensor acquisition metadata

**Response:**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "queued",
  "message": "Analysis job submitted successfully.",
  "image_count": 2
}
```

### 2. Fetch Job Result
`GET /api/v1/jobs/{job_id}`

**Response:**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "result": {
    "answer": "Expansion of built-up settlement structures observed between pre-change and post-change observations. [Task: multitemporal_change_vqa | Specialist Model: ChangeChat]",
    "confidence": {
      "score": 0.91,
      "level": "high",
      "label": "estimated confidence (cross-modal verified)"
    },
    "evidence": [...],
    "artifacts": [...],
    "execution_summary": {
      "task": "multitemporal_change_vqa",
      "models": ["ChangeChat"],
      "tools": ["validate_image", "classify_image_modality", "changechat", "generate_visual_evidence"],
      "processing_time_ms": 42.5
    }
  }
}
```

### 3. Fetch Execution Trace
`GET /api/v1/jobs/{job_id}/trace`

Returns the step-by-step observable trace for system auditing:
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "trace": [
    { "step": 1, "event": "validate_image", "status": "success", "duration_ms": 2.1 },
    { "step": 2, "event": "classify_image_modality", "status": "success", "duration_ms": 1.5 },
    { "step": 3, "event": "changechat", "status": "success", "duration_ms": 32.4 },
    { "step": 4, "event": "generate_visual_evidence", "status": "success", "duration_ms": 6.5 }
  ]
}
```

---

## 📜 License & Acknowledgments

Developed for the **Smart India Hackathon (SIH) 2025** under **ISRO Problem Statement 26167**.

Special thanks to the open-source research projects powering this architecture:
- [BigEarthNet.txt](https://txt.bigearth.net) (BIFOLD / TU Berlin / Univ. of Trento)
- [GeoChat](https://github.com/mbzuai-oryx/GeoChat) (MBZUAI Oryx)
- [ChangeChat](https://github.com/hanlinwu/ChangeChat) (Hanlin Wu et al.)
- [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) (NASA IMPACT & IBM Research)
- [TorchGeo](https://github.com/microsoft/torchgeo) (Microsoft)
