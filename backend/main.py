
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
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
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
    image = read_image(image_path)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable uploaded image.",
        )

    height, width = image.shape[:2]

    if height < 32 or width < 32:
        raise HTTPException(
            status_code=400,
            detail="Image must be at least 32 x 32 pixels.",
        )

    if not REFERENCE_PATH.exists():
        return {
            "defect_label": "Reference required",
            "inspection_status": "Pending reference",
            "anomaly_score": None,
            "similarity_score": None,
            "difference_map": None,
            "explanation": (
                "Image received successfully, but a reference image "
                "is required for visual comparison."
            ),
            "recommended_action": (
                "Configure a clean reference image of the same "
                "product model under similar capture conditions."
            ),
        }

    reference = read_image(REFERENCE_PATH)

    if reference is None:
        raise HTTPException(
            status_code=500,
            detail="The configured reference image could not be read.",
        )

    # Crop the product independently in both images.
    image_crop = crop_product_region(image)
    reference_crop = crop_product_region(reference)

    # Align the reference to the uploaded product where possible.
    aligned_reference = align_reference_to_image(
        reference_crop, image_crop
    )

    image_gray = cv2.cvtColor(
        image_crop, cv2.COLOR_BGR2GRAY
    )
    reference_gray = cv2.cvtColor(
        aligned_reference, cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(image_gray, reference_gray)

    # This is a normalized pixel-difference score, not a probability.
    anomaly_score = round(
        float(np.mean(difference) / 255.0), 4
    )

    similarity = cv2.matchTemplate(
        image_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED,
    )[0][0]

    similarity_score = (
        round(float(similarity), 4)
        if np.isfinite(similarity)
        else None
    )

    difference_map = create_difference_map(
        image_crop, aligned_reference
    )

    # Keep the existing conservative, preliminary comparison rule.
    close_match = (
        anomaly_score <= 0.02
        and similarity_score is not None
        and similarity_score >= 0.98
    )

    if close_match:
        return {
            "defect_label": (
                "No significant visual difference detected"
            ),
            "inspection_status": "Reference match",
            "anomaly_score": anomaly_score,
            "similarity_score": similarity_score,
            "difference_map": difference_map,
            "explanation": (
                "The cropped image closely matches the reference "
                "under the current comparison method. This does "
                "not guarantee that the product is defect-free."
            ),
            "recommended_action": (
                "Treat the result as provisional and continue "
                "normal quality-control checks."
            ),
        }

    return {
        "defect_label": "Visual difference flagged",
        "inspection_status": "Review required",
        "anomaly_score": anomaly_score,
        "similarity_score": similarity_score,
        "difference_map": difference_map,
        "explanation": (
            "The cropped product image differs from the reference. "
            "Differences may be caused by defects, alignment, "
            "lighting, or normal product variation."
        ),
        "recommended_action": (
            "Review the cropped image and difference map. Confirm "
            "that the reference is the same product model and "
            "verify the suspected defect manually."
        ),
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
