import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.ocr import is_model_ready, transcribe_text, warm_up
from app.services.pdf import transcribe_pdf


def run_warm_up() -> None:
    """Warm up the OCR model so the first user does not wait for it."""
    print("Warm-up OCR started...", flush = True)
    started_at = time.perf_counter()

    try:
        warm_up()
    except Exception as error:
        print(f"Warm-up OCR failed: {error}", flush = True)
        return

    elapsed = time.perf_counter() - started_at
    print(f"Warm-up OCR finished in {elapsed:.1f}s. Model is ready.", flush = True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Run in a thread so the API starts accepting requests immediately.
    threading.Thread(target = run_warm_up, daemon = True).start()
    yield


app = FastAPI(title = "OCR Web API", lifespan = lifespan)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()

def update_job(
    job_id: str,
    **updates: Any,
) -> None:
    """Update a job safely from the OCR worker thread."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)

def process_ocr_job(
    job_id: str,
    temporary_file_path: str,
    suffix: str,
) -> None:
    """Process one uploaded file in the background."""
    try:
        if suffix == ".pdf":
            def update_pdf_progress(
                current_page: int,
                total_pages: int,
            ) -> None:
                update_job(
                    job_id,
                    current_page = current_page,
                    total_pages = total_pages,
                )
                
            text = transcribe_pdf(
                temporary_file_path,
                progress_callback = update_pdf_progress,
            )
        else:
            update_job(
                job_id,
                current_page = 1,
                total_pages = 1,
            )
            
            text = transcribe_text(temporary_file_path)

            update_job(
                job_id,
                current_page = 1,
                total_pages = 1,
            )
            
        update_job(
            job_id,
            status = "completed",
            text = text,
        )
    
    except FileNotFoundError as error:
        update_job(
            job_id,
            status = "failed",
            error = str(error),
        )

    except ValueError as error:
        update_job(
            job_id,
            status = "failed",
            error = str(error),
        )

    except Exception as error:
        update_job(
            job_id,
            status = "failed",
            error = f"OCR processing failed: {error}",
        )

    finally:
        Path(temporary_file_path).unlink(missing_ok = True)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": is_model_ready(),
    }


@app.post("/ocr", status_code = 202)
async def create_ocr_job(
    file: UploadFile = File(...),
) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(
            status_code = 400, 
            detail = "No file provided",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix == ".pdf":
        pass
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        pass
    else:
        raise HTTPException(
            status_code = 400,
            detail = "Unsupported file type. Use JPG, JPEG, PNG, or PDF.",
        )

    file_data = await file.read()

    if not file_data:
        raise HTTPException(
            status_code = 400,
            detail = "The uploaded file is empty.",
        )
    if len(file_data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code = 413,
            detail = "The uploaded file is too large. Maximum size is 20 MB.",
        )

    job_id = uuid.uuid4().hex

    with NamedTemporaryFile(
        suffix = suffix,
        delete = False,
    ) as temporary_file:
        temporary_file.write(file_data)
        temporary_file.flush()
        temporary_file_path = temporary_file.name

    jobs[job_id] = {
        "status": "processing",
        "current_page": 0,
        "total_pages": 0,
    }

    worker = threading.Thread(
        target = process_ocr_job,
        args = (job_id, temporary_file_path, suffix),
        daemon = True,
    )
    worker.start()

    return {"job_id": job_id}

@app.get("/ocr/status/{job_id}")
async def get_ocr_status(
    job_id: str,
) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code = 404,
                detail = "OCR job not found.",
            )

        return dict(job)


### Starting the backend server ###
# cd backend
# uvicorn app.main:app --host 127.0.0.1 --port 8002
### --------------------------- ###