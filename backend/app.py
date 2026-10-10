
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.image_index import search_gallery


# =========================================================
# 1. กำหนดตำแหน่งโฟลเดอร์
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "index.html"

DATA_DIR = BASE_DIR / "data"
GALLERY_DIR = DATA_DIR / "gallery"

UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

# สร้างโฟลเดอร์ที่จำเป็นหากยังไม่มี
for directory in (
    FRONTEND_DIR,
    GALLERY_DIR,
    UPLOAD_DIR,
    RESULTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. ตั้งค่าระบบ
# =========================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# =========================================================
# 3. สร้าง FastAPI
# =========================================================

app = FastAPI(
    title="Face Scan API",
    description=(
        "ระบบค้นหาภาพที่คล้ายกัน "
        "จากคลังภาพที่กำหนดไว้"
    ),
    version="1.0.0",
)

# รองรับหน้าเว็บที่เปิดจาก development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =========================================================
# 4. หน้าแรก
# =========================================================

@app.get("/")
def home():
    if not FRONTEND_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "ไม่พบ frontend/index.html "
                "กรุณาสร้างไฟล์หน้าเว็บก่อน"
            ),
        )

    return FileResponse(
        path=FRONTEND_FILE,
        media_type="text/html",
    )


# =========================================================
# 5. ตรวจสอบสถานะ API
# =========================================================

@app.get("/api/health")
def health():
    return {
        "app": "Face Scan",
        "status": "ok",
        "gallery_exists": GALLERY_DIR.is_dir(),
    }


# =========================================================
# 6. ตรวจสอบชนิดไฟล์และตำแหน่งไฟล์
# =========================================================

def is_allowed_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def resolve_gallery_path(relative_path: str) -> Path:
    """
    แปลงพาธสัมพัทธ์ให้เป็นพาธจริงภายในคลังภาพ
    และป้องกันการเข้าถึงไฟล์นอกโฟลเดอร์ gallery
    """

    root = GALLERY_DIR.resolve()
    candidate = (root / relative_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="ไม่อนุญาตให้เข้าถึงไฟล์นี้",
        ) from exc

    if not is_allowed_image(candidate):
        raise HTTPException(
            status_code=404,
            detail="ไม่พบภาพหรือชนิดไฟล์ไม่รองรับ",
        )

    return candidate


# =========================================================
# 7. แสดงภาพจากคลังภาพ
# =========================================================

@app.get("/api/gallery/{relative_path:path}")
def get_gallery_image(relative_path: str):
    image_path = resolve_gallery_path(relative_path)

    return FileResponse(
        path=image_path,
        headers={"Cache-Control": "no-store"},
    )


# =========================================================
# 8. ค้นหาภาพจากภาพอ้างอิง
# =========================================================

@app.post("/api/search")
async def search_faces(
    file: UploadFile = File(...),
    limit: int = Query(default=20, ge=1, le=100),
):
    extension = Path(file.filename or "").suffix.lower()

    # ตรวจสอบชนิดไฟล์
    if extension not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=400,
            detail=(
                "รองรับเฉพาะไฟล์ JPG, JPEG, "
                "PNG, WEBP และ BMP"
            ),
        )

    # ตั้งชื่อไฟล์ใหม่ ไม่ใช้ชื่อไฟล์จากผู้ใช้งานโดยตรง
    upload_path = UPLOAD_DIR / (
        f"{uuid4().hex}{extension}"
    )

    size = 0

    try:
        # บันทึกภาพชั่วคราว พร้อมจำกัดขนาดไฟล์
        with upload_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="ไฟล์มีขนาดเกิน 10 MB",
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="ไฟล์ภาพว่างเปล่า",
            )

        # ค้นหาในคลังภาพ
        try:
            results = search_gallery(
                str(upload_path),
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        # เพิ่ม URL สำหรับให้หน้าเว็บโหลดภาพได้
        for item in results:
            try:
                relative_path = (
                    Path(item["path"])
                    .resolve()
                    .relative_to(GALLERY_DIR.resolve())
                )
            except (ValueError, KeyError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="พบพาธผลลัพธ์ที่ไม่ถูกต้อง",
                ) from exc

            # เข้ารหัสชื่อไฟล์ เช่น ภาษาไทยและช่องว่าง
            encoded_path = quote(
                relative_path.as_posix(),
                safe="/",
            )

            item["image_url"] = (
                f"/api/gallery/{encoded_path}"
            )

        return {
            "count": len(results),
            "results": results,
            "message": (
                "ค้นหาเสร็จแล้ว"
                if results
                else "ไม่พบภาพที่มีใบหน้าตรวจจับได้ในคลังภาพ"
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        # รายละเอียดข้อผิดพลาดจริงจะแสดงใน Terminal
        print(f"[Face Scan Error] {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "เกิดข้อผิดพลาดระหว่างค้นหาภาพ "
                "กรุณาตรวจสอบ Terminal"
            ),
        ) from exc

    finally:
        # ลบภาพอ้างอิงชั่วคราวหลังค้นหาเสร็จ
        upload_path.unlink(missing_ok=True)
        await file.close()