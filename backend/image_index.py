
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. ตั้งค่าพาธและโมเดล
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
GALLERY_DIR = BASE_DIR / "data" / "gallery"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# ใช้ CPU เพื่อรองรับเครื่องที่ไม่มี CUDA
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

# เพิ่มความละเอียดเพื่อช่วยตรวจจับใบหน้าขนาดเล็ก
face_app.prepare(
    ctx_id=-1,
    det_size=(1024, 1024),
)


# ============================================================
# 2. อ่านไฟล์ภาพ
# ============================================================

def read_image(path: Path):
    """อ่านภาพ รวมถึงพาธที่มีอักขระภาษาไทย"""

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)

        if raw.size == 0:
            return None

        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    except (OSError, cv2.error):
        return None


# ============================================================
# 3. ตรวจจับและสร้างเวกเตอร์ของทุกใบหน้า
# ============================================================

def get_embeddings(image):
    """
    คืนค่าเวกเตอร์ของทุกใบหน้าที่ตรวจจับได้ในภาพ
    ไม่เลือกเฉพาะใบหน้าที่ใหญ่ที่สุด
    """

    if image is None:
        return []

    try:
        faces = face_app.get(image)
    except Exception as exc:
        print(f"ตรวจจับใบหน้าไม่สำเร็จ: {exc}")
        return []

    embeddings = []

    for face in faces:
        embedding = getattr(face, "normed_embedding", None)

        if embedding is None:
            continue

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            continue

        if not np.all(np.isfinite(embedding)):
            continue

        # เวกเตอร์ที่ normalize แล้วเปรียบเทียบด้วย dot product ได้
        embeddings.append(embedding)

    return embeddings


def get_embedding(image):
    """
    รักษาฟังก์ชันเดิมไว้สำหรับส่วนอื่นที่เรียกใช้
    โดยคืนเวกเตอร์ของใบหน้าที่ใหญ่ที่สุด
    """

    if image is None:
        return None

    try:
        faces = face_app.get(image)
    except Exception:
        return None

    if not faces:
        return None

    face = max(
        faces,
        key=lambda item: (
            (item.bbox[2] - item.bbox[0])
            * (item.bbox[3] - item.bbox[1])
        ),
    )

    embedding = getattr(face, "normed_embedding", None)

    if embedding is None:
        return None

    return np.asarray(embedding, dtype=np.float32)


# ============================================================
# 4. ค้นหาภาพจากคลังภาพ
# ============================================================

def search_gallery(
    reference_path: str,
    limit: int = 20,
):
    """
    เปรียบเทียบภาพอ้างอิงกับทุกใบหน้าในคลังภาพ

    ผลลัพธ์แต่ละรายการยังคงมี:
    filename, path, similarity

    similarity เป็นคะแนนความคล้ายคลึงเชิงเวกเตอร์
    ไม่ใช่เปอร์เซ็นต์ความถูกต้อง
    """

    reference_file = Path(reference_path)

    if not reference_file.is_file():
        raise ValueError("ไม่พบไฟล์ภาพอ้างอิง")

    reference_image = read_image(reference_file)
    reference_embeddings = get_embeddings(reference_image)

    if not reference_embeddings:
        raise ValueError("ไม่พบใบหน้าในภาพอ้างอิง")

    if not GALLERY_DIR.exists():
        return []

    # โดยทั่วไปภาพอ้างอิงควรมีใบหน้าเดียว
    # หากมีหลายใบหน้า จะเปรียบเทียบทุกใบหน้าที่ตรวจพบ
    results = []

    image_paths = sorted(
        path
        for path in GALLERY_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for path in image_paths:
        image = read_image(path)

        if image is None:
            continue

        candidate_embeddings = get_embeddings(image)

        if not candidate_embeddings:
            continue

        best_similarity = -1.0

        # เปรียบเทียบทุกใบหน้าในภาพกับทุกใบหน้าอ้างอิง
        for reference_embedding in reference_embeddings:
            for candidate_embedding in candidate_embeddings:

                if (
                    reference_embedding.shape
                    != candidate_embedding.shape
                ):
                    continue

                similarity = float(
                    np.dot(
                        reference_embedding,
                        candidate_embedding,
                    )
                )

                if np.isfinite(similarity):
                    best_similarity = max(
                        best_similarity,
                        similarity,
                    )

        if best_similarity < 0:
            continue

        results.append({
            "filename": path.name,
            "path": str(path),
            "similarity": round(best_similarity, 4),
        })

    # เรียงจากคะแนนสูงไปต่ำ
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:max(1, int(limit))]