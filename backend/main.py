import base64
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "q_insight.db"
REFERENCE_PATH = DATA_DIR / "reference.jpg"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Q-Insight API",
    description=(
        "Visual quality inspection and manufacturing "
        "analytics prototype"
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://q-insight-2026.web.app",
        "https://q-insight-2026.firebaseapp.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DATABASE
# Existing inspection history is preserved.
# --------------------------------------------------

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                defect_label TEXT NOT NULL,
                inspection_status TEXT NOT NULL,
                anomaly_score REAL,
                machine_id TEXT,
                tool_id TEXT,
                batch_id TEXT,
                shift TEXT,
                data_source TEXT NOT NULL,
                explanation TEXT NOT NULL,
                recommended_action TEXT NOT NULL
            )
        """)


initialize_db()


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

def read_image(image_path: Path):
    """Read an image and return None if it cannot be decoded."""
    try:
        image_bytes = image_path.read_bytes()
    except OSError:
        return None

    if not image_bytes:
        return None

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def analyze_discoloration(image: np.ndarray) -> dict:
    """
    Estimate yellow/brown color coverage for diagnostic purposes.
    This is a heuristic, not a trained discoloration classifier.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.int16)
    yellow_blue = lab[:, :, 2].astype(np.int16)

    # Ignore very dark edges and nearly white glare.
    valid = (lightness >= 65) & (lightness <= 240)

    # Estimate the amount of warm/yellow coloration.
    yellow_mask = valid & (yellow_blue >= 137)

    # Estimate darker yellow/brown regions.
    brown_mask = (
        valid
        & (lightness <= 190)
        & (yellow_blue >= 135)
    )

    valid_count = int(np.count_nonzero(valid))

    if valid_count == 0:
        return {
            "discoloration_score": 0.0,
            "yellow_pixel_ratio": 0.0,
            "brown_pixel_ratio": 0.0,
        }

    yellow_ratio = (
        np.count_nonzero(yellow_mask) / valid_count
    )
    brown_ratio = (
        np.count_nonzero(brown_mask) / valid_count
    )

    return {
        "discoloration_score": round(
            100 * (0.6 * yellow_ratio + 0.4 * brown_ratio), 2
        ),
        "yellow_pixel_ratio": round(100 * yellow_ratio, 2),
        "brown_pixel_ratio": round(100 * brown_ratio, 2),
    }

# --------------------------------------------------
# VISUAL DIFFERENCE HEATMAP
# --------------------------------------------------


def crop_product_region(image: np.ndarray) -> np.ndarray:
    """
    Attempt to crop the largest plausible rectangular product region.
    Falls back to the full image if a reliable rectangle is not found.
    """
    height, width = image.shape[:2]
    image_area = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny details and contours covering almost the whole image.
        if area < image_area * 0.03 or area > image_area * 0.85:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(
            contour, 0.03 * perimeter, True
        )

        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue

        x, y, box_width, box_height = cv2.boundingRect(polygon)

        if box_width < width * 0.08 or box_height < height * 0.08:
            continue

        aspect_ratio = box_width / max(box_height, 1)

        if not 0.2 <= aspect_ratio <= 4.0:
            continue

        candidates.append((area, x, y, box_width, box_height))

    if not candidates:
        # Do not guess a crop if no plausible product rectangle is found.
        return image.copy()

    # The largest plausible quadrilateral is our best crop candidate.
    _, x, y, box_width, box_height = max(candidates)

    pad_x = int(box_width * 0.04)
    pad_y = int(box_height * 0.04)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_width + pad_x)
    y2 = min(height, y + box_height + pad_y)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return image.copy()

    return cropped


def align_reference_to_image(
    reference: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """
    Try feature-based alignment of the reference to the uploaded product.
    If reliable alignment cannot be estimated, fall back to resizing.
    """
    target_height, target_width = image.shape[:2]

    reference_gray = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )
    image_gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(nfeatures=1500)

    ref_keypoints, ref_descriptors = orb.detectAndCompute(
        reference_gray, None
    )
    img_keypoints, img_descriptors = orb.detectAndCompute(
        image_gray, None
    )

    if (
        ref_descriptors is not None
        and img_descriptors is not None
        and len(ref_keypoints) >= 8
        and len(img_keypoints) >= 8
    ):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        pairs = matcher.knnMatch(
            ref_descriptors, img_descriptors, k=2
        )

        good_matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]

        if len(good_matches) >= 8:
            source_points = np.float32([
                ref_keypoints[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            destination_points = np.float32([
                img_keypoints[m.trainIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            matrix, inliers = cv2.findHomography(
                source_points,
                destination_points,
                cv2.RANSAC,
                4.0,
            )

            if (
                matrix is not None
                and inliers is not None
                and int(inliers.sum()) >= 6
            ):
                aligned = cv2.warpPerspective(
                    reference,
                    matrix,
                    (target_width, target_height),
                )

                return aligned

    # Fallback: dimensions match, but geometric alignment is not guaranteed.
    return cv2.resize(
        reference, (target_width, target_height)
    )


def create_difference_map(
    image: np.ndarray,
    reference: np.ndarray,
):
    """Create a heatmap overlay for the cropped product region."""
    reference = cv2.resize(
        reference, (image.shape[1], image.shape[0])
    )

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(image_gray, reference_gray)

    # Suppress small pixel variations before creating the heatmap.
    difference = cv2.GaussianBlur(difference, (5, 5), 0)

    heatmap = cv2.applyColorMap(
        difference, cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image, 0.65, heatmap, 0.35, 0
    )

    mask = difference > 25
    result = image.copy()
    result[mask] = overlay[mask]

    height, width = result.shape[:2]
    scale = min(640 / width, 640 / height, 1.0)

    if scale < 1.0:
        result = cv2.resize(
            result,
            (int(width * scale), int(height * scale)),
        )

    success, encoded_image = cv2.imencode(".jpg", result)

    if not success:
        return None

    encoded_string = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded_string}"

# --------------------------------------------------
# INSPECTION ENGINE
# --------------------------------------------------

def inspect_image(image_path: Path):
    """Inspect a product image using the trained PatchCore model."""
    import base64
    import cv2
    import numpy as np
    import torch
    from anomalib.data import ImageBatch

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    height, width = image.shape[:2]
    if height < 32 or width < 32:
        raise ValueError(
            "Image dimensions must be at least 32 x 32 pixels."
        )

    model = load_patchcore_model()

        # Match the image normalization used by Anomalib's
    # default ImageNet-pretrained backbone pipeline.
    from PIL import Image
    from torchvision import transforms
    from anomalib.data import ImageBatch

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tensor = preprocess(pil_image).unsqueeze(0)
    batch = ImageBatch(image=tensor)

    with torch.inference_mode():
        predictions = model.predict_step(batch, 0)

    # Anomalib may return a batch-like prediction container,
    # not a Python list. Do not index it using predictions[0].
    score_tensor = getattr(predictions, "pred_score", None)

    if score_tensor is None:
        raise RuntimeError(
            "No pred_score found. Returned prediction type: "
            f"{type(predictions).__name__}; "
            f"available fields: {getattr(predictions, '__dict__', {}).keys()}"
        )

    anomaly_score = float(
        score_tensor.detach().cpu().reshape(-1)[0]
    )

    color_diagnostic = analyze_discoloration(image)
    print(
        f"[Q-Insight] discoloration diagnostic="
        f"{color_diagnostic['discoloration_score']:.2f}"
    )

    anomaly_map_tensor = getattr(predictions, "anomaly_map", None)
    difference_map = None

    if anomaly_map_tensor is not None:
        anomaly_map = (
            anomaly_map_tensor.detach().cpu().squeeze().numpy()
        )

        if anomaly_map.ndim == 2 and anomaly_map.size:
            anomaly_map = cv2.resize(
                anomaly_map.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            heatmap = cv2.normalize(
                anomaly_map, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap, cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(
                image, 0.60, heatmap_color, 0.40, 0
            )

            success, encoded = cv2.imencode(".jpg", overlay)
            if success:
                difference_map = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(
                        encoded.tobytes()
                    ).decode("utf-8")
                )

    
    threshold = PATCHCORE_THRESHOLD

    # Separate discoloration signal from the PatchCore anomaly score.
    discoloration_score = color_diagnostic["discoloration_score"]
    DISCOLORATION_THRESHOLD = 80.0  # Provisional; validate using clean and discolored samples.

    is_patchcore_anomaly = anomaly_score >= threshold
    is_discoloration_suspected = (
        discoloration_score >= DISCOLORATION_THRESHOLD
    )

    is_anomaly = is_patchcore_anomaly or is_discoloration_suspected


    print(
        f"[Q-Insight] image={Path(image_path).name} "
        f"score={anomaly_score:.6f} "
        f"threshold={threshold:.6f}"
    )

    return {
        "defect_label": (
            "Potential defect" if is_anomaly
            else "No anomaly detected"
        ),
        "inspection_status": (
            "Review required" if is_anomaly else "Passed"
        ),
        "anomaly_score": anomaly_score,
        "similarity_score": None,
        "difference_map": difference_map,
        "explanation": (
            f"PatchCore anomaly score: {anomaly_score:.4f}. "
            f"Discoloration diagnostic: {discoloration_score:.2f}. "
            + (
                "The discoloration diagnostic exceeded its provisional "
                "threshold; visually verify possible yellowing or browning."
                if is_discoloration_suspected
                else (
                    "The PatchCore score exceeded its provisional threshold; "
                    "inspect the highlighted regions."
                    if is_patchcore_anomaly
                    else
                    "Neither provisional threshold was exceeded. "
                    "This does not guarantee that the product is defect-free."
                )
            )
        ),
        "recommended_action": (
            "Review the product for discoloration and other suspected defects."
            if is_anomaly
            else "Continue with normal inspection procedures."
        ),
        "discoloration_score": color_diagnostic["discoloration_score"],
        "yellow_pixel_ratio": color_diagnostic["yellow_pixel_ratio"],
        "brown_pixel_ratio": color_diagnostic["brown_pixel_ratio"],
    }

# --------------------------------------------------
# BASIC ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "Q-Insight",
        "message": "API is running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "Q-Insight",
    }


# --------------------------------------------------
# CREATE INSPECTION
# --------------------------------------------------

@app.post("/inspect")
async def create_inspection(
    file: UploadFile = File(...),
    machine_id: str = Form("M-01"),
    tool_id: str = Form("T-01"),
    batch_id: str = Form("B-001"),
    shift: str = Form("Day"),
    data_source: str = Form("real_sample"),
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB.",
        )

    # Decode the actual uploaded bytes; do not trust the
    # extension or MIME type alone.
    buffer = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    inspection_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{inspection_id}.jpg"

    if not cv2.imwrite(str(image_path), decoded):
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded image.",
        )

    # Inspect before writing a completed result to the database.
    result = inspect_image(image_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO inspections (
                id,
                timestamp,
                image_path,
                defect_label,
                inspection_status,
                anomaly_score,
                machine_id,
                tool_id,
                batch_id,
                shift,
                data_source,
                explanation,
                recommended_action
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                timestamp,
                str(image_path.relative_to(BASE_DIR)),
                result["defect_label"],
                result["inspection_status"],
                result["anomaly_score"],
                machine_id,
                tool_id,
                batch_id,
                shift,
                data_source,
                result["explanation"],
                result["recommended_action"],
            ),
        )

    return {
        "id": inspection_id,
        "timestamp": timestamp,
        "machine_id": machine_id,
        "tool_id": tool_id,
        "batch_id": batch_id,
        "shift": shift,
        "data_source": data_source,
        **result,
    }


# --------------------------------------------------
# INSPECTION HISTORY
# --------------------------------------------------

@app.get("/inspections")
def get_inspections():
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM inspections
            ORDER BY timestamp DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

@app.get("/analytics")
def get_analytics():
    with connect_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM inspections"
        ).fetchone()[0]

        statuses = conn.execute(
            """
            SELECT inspection_status, COUNT(*) AS count
            FROM inspections
            GROUP BY inspection_status
            """
        ).fetchall()

        batches = conn.execute(
            """
            SELECT
                batch_id,
                COUNT(*) AS total,
                SUM(
                    CASE
                        WHEN inspection_status = 'Review required'
                        THEN 1
                        ELSE 0
                    END
                ) AS flagged
            FROM inspections
            GROUP BY batch_id
            """
        ).fetchall()

    return {
        "total_inspections": total,
        "status_counts": [
            dict(row) for row in statuses
        ],
        "batch_summary": [
            dict(row) for row in batches
        ],
    }


# --------------------------------------------------
# PATCHCORE MODEL CONFIGURATION
# --------------------------------------------------


PATCHCORE_CHECKPOINT = (
    DATA_DIR
    / "patchcore_multi_product_v2_results"
    / "Patchcore"
    / "q_insight_multi_product_v2"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

# Temporary diagnostic threshold from the v2 evaluation.
# This threshold is not yet validated for deployment.
PATCHCORE_THRESHOLD = 50.0

_patchcore_model = None


def load_patchcore_model():
    """Load the trained single-switch PatchCore model once."""
    global _patchcore_model

    if _patchcore_model is not None:
        return _patchcore_model

    if not PATCHCORE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found: {PATCHCORE_CHECKPOINT}"
        )

    import torch
    from anomalib.models import Patchcore

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    checkpoint = torch.load(
        PATCHCORE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to("cpu")

    _patchcore_model = model
    return _patchcore_model
