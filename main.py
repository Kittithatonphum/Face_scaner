
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = BASE_DIR / "reference"
GALLERY_DIR = BASE_DIR / "gallery_images"
RESULT_DIR = BASE_DIR / "results"

REFERENCE_IMAGE = REFERENCE_DIR / "target.jpg"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# ค่าเริ่มต้นสำหรับการทดลองเท่านั้น
# ไม่ใช่เกณฑ์ยืนยันว่าบุคคลเป็นคนเดียวกัน
SIMILARITY_THRESHOLD = 0.45

# ขนาดภาพสำหรับโมเดลตรวจจับใบหน้า
DETECTION_SIZE = (640, 640)


# ==================================================
# 2. READ IMAGE - รองรับพาธภาษาไทย
# ==================================================

def read_image(image_path):
    try:
        image_path = Path(image_path)

        if not image_path.is_file():
            print(f"ไม่พบไฟล์: {image_path}")
            return None

        image_data = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        if image_data.size == 0:
            print(f"ไฟล์ภาพว่างเปล่า: {image_path.name}")
            return None

        image = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(f"OpenCV ไม่สามารถถอดรหัสภาพ: {image_path.name}")
            return None

        return image

    except Exception as error:
        print(f"เกิดข้อผิดพลาดในการอ่านภาพ {image_path.name}: {error}")
        return None


# ==================================================
# 3. SAVE IMAGE - รองรับพาธภาษาไทย
# ==================================================

def save_image(image_path, image):
    try:
        image_path = Path(image_path)
        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extension = image_path.suffix.lower() or ".jpg"

        success, encoded = cv2.imencode(
            extension,
            image,
        )

        if not success:
            print(f"เข้ารหัสภาพไม่สำเร็จ: {image_path.name}")
            return False

        encoded.tofile(str(image_path))
        return True

    except Exception as error:
        print(f"บันทึกภาพไม่สำเร็จ: {error}")
        return False


# ==================================================
# 4. INITIALIZE INSIGHTFACE
# ==================================================

def initialize_model():
    print("\nกำลังโหลดโมเดล InsightFace...")

    # เริ่มด้วย CPU เพื่อไม่ให้คำเตือน CUDA
    # รบกวนการทดสอบในสภาพแวดล้อมปัจจุบัน
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE,
    )

    print("โหลดโมเดลสำเร็จ")
    return app


# ==================================================
# 5. DETECT FACES
# ==================================================

def detect_faces(app, image, image_name):
    try:
        faces = app.get(image)

        if faces is None or len(faces) == 0:
            print(f"ไม่พบใบหน้า: {image_name}")
            print(
                f"ขนาดภาพ: {image.shape[1]} x {image.shape[0]}"
            )
            print(
                "คำแนะนำ: ลองใช้ภาพหน้าตรงที่ชัดเจน "
                "ใบหน้าใหญ่พอ และมีแสงเพียงพอ"
            )
            return []

        print(f"ตรวจพบใบหน้าใน {image_name}: {len(faces)} ใบหน้า")
        return faces

    except Exception as error:
        print(f"ตรวจจับใบหน้าไม่สำเร็จ: {image_name}")
        print(f"รายละเอียด: {error}")
        return []


# ==================================================
# 6. EXTRACT REFERENCE EMBEDDING
# ==================================================

def extract_reference_embedding(app):
    print("\nกำลังตรวจสอบรูปอ้างอิง...")

    if not REFERENCE_IMAGE.is_file():
        print("ERROR: ไม่พบรูปอ้างอิง")
        print(f"ตำแหน่งที่ต้องการ: {REFERENCE_IMAGE}")
        print("กรุณาวางรูปไว้ใน reference และตั้งชื่อ target.jpg")
        return None

    image = read_image(REFERENCE_IMAGE)

    if image is None:
        print("ไม่สามารถอ่านรูปอ้างอิงได้")
        return None

    print(
        f"ขนาดรูปอ้างอิง: {image.shape[1]} x {image.shape[0]}"
    )

    faces = detect_faces(
        app,
        image,
        REFERENCE_IMAGE.name,
    )

    if len(faces) == 0:
        debug_path = RESULT_DIR / "debug_target.jpg"
        save_image(debug_path, image)

        print("ยังไม่สามารถสร้างเวกเตอร์ใบหน้าได้")
        print(f"สำเนารูปสำหรับตรวจสอบ: {debug_path}")
        return None

    if len(faces) > 1:
        print(
            "คำเตือน: พบหลายใบหน้าในรูปอ้างอิง "
            "โปรแกรมจะเลือกใบหน้าที่มีกรอบใหญ่ที่สุด"
        )

    # เลือกใบหน้าที่มีพื้นที่กรอบใหญ่ที่สุด
    face = max(
        faces,
        key=lambda item: (
            max(0, item.bbox[2] - item.bbox[0])
            * max(0, item.bbox[3] - item.bbox[1])
        ),
    )

    embedding = getattr(
        face,
        "normed_embedding",
        None,
    )

    if embedding is None:
        print("โมเดลไม่สามารถสร้างเวกเตอร์ใบหน้าได้")
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim != 1 or embedding.size == 0:
        print("เวกเตอร์ใบหน้ามีรูปแบบไม่ถูกต้อง")
        return None

    if not np.all(np.isfinite(embedding)):
        print("เวกเตอร์ใบหน้ามีค่าที่ไม่ถูกต้อง")
        return None

    print("สร้างเวกเตอร์ใบหน้าอ้างอิงสำเร็จ")
    return embedding


# ==================================================
# 7. COSINE SIMILARITY
# ==================================================

def cosine_similarity(vector_a, vector_b):
    vector_a = np.asarray(
        vector_a,
        dtype=np.float32,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32,
    )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return -1.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


# ==================================================
# 8. PROCESS ONE GALLERY IMAGE
# ==================================================

def process_gallery_image(
    app,
    image_path,
    reference_embedding,
):
    image = read_image(image_path)

    if image is None:
        return 0

    faces = detect_faces(
        app,
        image,
        image_path.name,
    )

    if not faces:
        return 0

    best_similarity = -1.0

    # ถ้าภาพมีหลายคน ให้เปรียบเทียบแต่ละใบหน้า
    for face in faces:
        embedding = getattr(
            face,
            "normed_embedding",
            None,
        )

        if embedding is None:
            continue

        similarity = cosine_similarity(
            reference_embedding,
            embedding,
        )

        best_similarity = max(
            best_similarity,
            similarity,
        )

    if best_similarity < SIMILARITY_THRESHOLD:
        return 0

    output_path = RESULT_DIR / image_path.name

    try:
        # เก็บรูปต้นฉบับ ไม่วาดกรอบหรือข้อความทับรูป
        shutil.copy2(
            image_path,
            output_path,
        )

        print(
            f"ผ่านเกณฑ์: {image_path.name} "
            f"| Similarity: {best_similarity:.4f}"
        )

        return 1

    except OSError as error:
        print(f"คัดลอกรูปไม่สำเร็จ: {image_path.name}")
        print(f"รายละเอียด: {error}")
        return 0


# ==================================================
# 9. MAIN
# ==================================================

def main():
    print("=" * 55)
    print("FACE SIMILARITY SEARCH")
    print("=" * 55)

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ตรวจสอบรูปอ้างอิงก่อนโหลดโมเดล
    if not REFERENCE_IMAGE.is_file():
        print("\nไม่พบรูปอ้างอิง")
        print(f"กรุณาวางรูปไว้ที่: {REFERENCE_IMAGE}")
        return

    gallery_images = sorted(
        [
            path
            for path in GALLERY_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not gallery_images:
        print("\nไม่พบรูปใน gallery_images")
        print(f"กรุณาเพิ่มรูปภาพไว้ที่: {GALLERY_DIR}")
        return

    try:
        app = initialize_model()
    except Exception as error:
        print("\nโหลดโมเดลไม่สำเร็จ")
        print(f"รายละเอียด: {error}")
        print(
            "หากเป็นการใช้งานครั้งแรก ให้ตรวจสอบอินเทอร์เน็ต "
            "สำหรับดาวน์โหลดโมเดล"
        )
        return

    reference_embedding = extract_reference_embedding(app)

    if reference_embedding is None:
        print("\nยกเลิกการค้นหา เนื่องจากอ่านใบหน้าอ้างอิงไม่ได้")
        return

    # ล้างเฉพาะไฟล์ภาพผลลัพธ์จากการรันครั้งก่อน
    # ไฟล์อื่นใน results จะไม่ถูกลบ
    for path in RESULT_DIR.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            try:
                path.unlink()
            except OSError as error:
                print(f"ลบผลลัพธ์เก่าไม่ได้: {path.name}: {error}")

    print("\n" + "-" * 55)
    print(f"จำนวนรูปในคลัง: {len(gallery_images)}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}")
    print("-" * 55)

    match_count = 0

    for index, image_path in enumerate(
        gallery_images,
        start=1,
    ):
        print(
            f"\n[{index}/{len(gallery_images)}] "
            f"กำลังตรวจสอบ: {image_path.name}"
        )

        match_count += process_gallery_image(
            app,
            image_path,
            reference_embedding,
        )

    print("\n" + "=" * 55)
    print("สรุปผล")
    print("=" * 55)
    print(f"ตรวจสอบรูปภาพ: {len(gallery_images)} รูป")
    print(f"รูปที่ผ่านเกณฑ์: {match_count} รูป")
    print(f"โฟลเดอร์ผลลัพธ์: {RESULT_DIR}")
    print("เสร็จสิ้น")


if __name__ == "__main__":
    main()