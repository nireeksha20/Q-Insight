# Q-Insight — AI-Powered Manufacturing Quality Inspection

Q-Insight is a computer-vision-based quality inspection prototype that analyzes product images, identifies potential visual anomalies, highlights image differences, and organizes inspection results into production analytics.

**Project status:** Working prototype · Human verification required

## 🚀 Live Demo

**Deployed Application:** https://q-insight-2026.web.app

Q-Insight is an AI-assisted manufacturing quality inspection system designed to analyze product images, identify potential defects, and support quality-control decisions through inspection history and analytics.

> **Note:** The live demo uses a locally hosted FastAPI backend exposed through a temporary Cloudflare Quick Tunnel. The backend and tunnel must remain running for inspection and analytics features to work.

## ✨ Key Features

- **AI-Assisted Visual Inspection** — Analyze product images for potential anomalies.
- **Defect Visualization** — View inspection results and visual comparisons.
- **Inspection History** — Review saved inspection records.
- **Quality Analytics** — Track inspection counts and flagged results by production batch.
- **Human Review Support** — Flag suspicious or uncertain results for further verification.

## 🛠️ Technology Stack

- **Frontend:** React, Vite
- **Backend:** Python, FastAPI
- **Image Processing:** OpenCV, PyTorch, PatchCore
- **Database:** SQLite
- **Deployment:** Firebase Hosting

## Key Features

- **AI anomaly detection:** Uses PatchCore with a Wide ResNet-50-2 backbone to identify potential visual defects.
- **Discoloration diagnostics:** Calculates yellow and brown pixel ratios as supplementary visual indicators.
- **Visual difference maps:** Displays highlighted image differences to assist manual inspection.
- **Inspection history:** Stores inspection records in a local SQLite database.
- **Quality analytics:** Summarizes inspection findings and production-batch performance.
- **Production factor analysis:** Groups inspection outcomes by machine, tool, and shift.
- **Human-in-the-loop verification:** Flags potential defects for review instead of treating anomaly scores as confirmed defect probabilities.

## Technology Stack

| Component       | Technologies                                |
| --------------- | ------------------------------------------- |
| Frontend        | React, Vite                                 |
| Backend         | Python, FastAPI                             |
| Computer vision | PatchCore, Anomalib, OpenCV                 |
| Deep learning   | PyTorch, Torchvision                        |
| Data storage    | SQLite                                      |
| Visualization   | Inspection heatmaps and dashboard analytics |

## How It Works

1. Upload a product image and enter production metadata.
2. The backend preprocesses the image and runs PatchCore inference.
3. The system calculates supplementary discoloration diagnostics.
4. A visual difference map helps identify regions that require attention.
5. The inspection result and metadata are saved for history and analytics.
6. Potential anomalies are flagged for human verification.

## Getting Started

### Prerequisites

- Windows, macOS, or Linux
- Python compatible with the project's pinned dependencies
- Node.js and npm
- The trained PatchCore checkpoint required by the backend

### 1. Clone the repository

```bash
git clone https://github.com/nireeksha20/Q-Insight.git
cd Q-Insight
```

### 2. Set up the backend

Open a terminal in the project root:

```powershell
cd backend
python -m venv .venv-anomaly
.\.venv-anomaly\Scripts\Activate.ps1
```

Install the project's required Python packages using its dependency file, if provided. PyTorch, Anomalib, OpenCV, FastAPI, and the other imports must be available in the active environment.

**Model checkpoint:** The trained PatchCore checkpoint and experiment datasets are not included in this repository. Obtain the checkpoint separately or train the model using the included training scripts. Configure the backend to point to the correct checkpoint before starting the API.

Start the backend from the `backend` directory:

```powershell
python -m uvicorn main:app --reload --port 8000
```

The API should be available at `http://localhost:8000`.

If enabled by the application, API documentation is available at `http://localhost:8000/docs`.

### 3. Set up the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

If the frontend expects a specific backend URL, configure it to use the local FastAPI server at `http://localhost:8000`.

> Use the dependency versions and configuration required by your local project. The commands above describe the development workflow; a fresh installation must be verified against the actual dependency files and model checkpoint.

## Training and Evaluation

The repository includes scripts for training and evaluating PatchCore models on prepared datasets:

- `backend/train_patchcore_multi_product_v1.py`
- `backend/train_patchcore_multi_product_v2.py`
- `backend/evaluate_patchcore_multi_product_v2.py`
- `backend/evaluate_combined_validation.py`
- `backend/evaluate_real_good.py`
- `backend/evaluate_real_validation.py`

Run the appropriate script from the `backend` directory using the Python environment containing the required dependencies.

The scripts expect their corresponding local datasets and model artifacts. These files must be prepared separately before running the experiments.

## Important Limitations

- This is a prototype, not a certified industrial inspection system.
- Anomaly scores are not probabilities of a defect.
- The decision threshold is provisional and requires validation on representative, independently labelled samples.
- Discoloration scores are heuristic color diagnostics, not a trained discoloration classifier.
- Image differences can be affected by lighting, camera position, image alignment, and background variation.
- False positives and false negatives are possible.
- Production factor analysis identifies associations and does not establish causation.
- Human verification is required before making manufacturing or product-quality decisions.

## Repository Structure

```text
Q-Insight/
├── backend/
│   ├── main.py
│   ├── train_patchcore_multi_product_v1.py
│   ├── train_patchcore_multi_product_v2.py
│   └── evaluate_*.py
├── frontend/
├── data/                 # Local datasets and runtime data
├── .gitignore
└── README.md
```

## Future Improvements

- Expand the labelled dataset across product types and defect categories.
- Validate thresholds on a separate, representative test set.
- Improve defect localization and discoloration analysis.
- Add automated evaluation reports and regression tests.
- Improve model artifact management and reproducible setup instructions.

## Author

**Nireeksha**

GitHub: [@nireeksha20](https://github.com/nireeksha20)

Repository: [Q-Insight](https://github.com/nireeksha20/Q-Insight)

---

_Q-Insight is an independent manufacturing quality inspection prototype. Inspection results are advisory and require human verification._
