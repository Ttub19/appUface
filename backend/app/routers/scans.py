"""
POST   /scans/predict     — run face-shape prediction on an image (open to guests too)
POST   /scans             — save a scan result to the current user's history
GET    /scans             — list the current user's history (with search + shape filter)
DELETE /scans/{scan_id}   — remove one history row
GET    /scans/export.csv  — export the current user's history as CSV
"""
import csv
import io
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import ScanHistory, User
from app.schemas import PredictRequest, PredictResponse, ScanCreate, ScanOut
from app.services.face_shape_predictor import FaceShapePredictor, ModelUnavailableError, get_predictor

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    predictor: FaceShapePredictor = Depends(get_predictor),
) -> PredictResponse:
    """
    Open to everyone, including the front-end's unauthenticated "guest" role —
    matches the mockup, where a guest can see a result but must log in to save it.
    """
    try:
        # model inference is CPU-bound (~0.1-0.5 s): keep it off the event loop
        result = await run_in_threadpool(predictor.predict, payload.image_data_url, payload.gender)
    except ValueError as exc:  # bad image / no face found
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return PredictResponse(
        face_shape=result.face_shape,
        face_shape_name=result.face_shape_name,
        confidence=result.confidence,
        model=result.model,
    )


@router.post("", response_model=ScanOut, status_code=status.HTTP_201_CREATED)
async def create_scan(
    payload: ScanCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ScanHistory:
    scan = ScanHistory(user_id=current_user.id, **payload.model_dump())
    db.add(scan)
    await db.commit()
    await db.refresh(scan)
    return scan


@router.get("", response_model=List[ScanOut])
async def list_scans(
    q: Optional[str] = Query(default=None, description="ค้นหาจากชื่อรูปทรงหน้า"),
    shape: Optional[str] = Query(default=None, description="กรองตามรูปทรงหน้า เช่น oval, round"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[ScanHistory]:
    stmt = select(ScanHistory).where(ScanHistory.user_id == current_user.id)
    if shape:
        stmt = stmt.where(ScanHistory.face_shape == shape)
    if q:
        stmt = stmt.where(ScanHistory.face_shape_name.ilike(f"%{q}%"))
    stmt = stmt.order_by(ScanHistory.created_at.desc())

    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scan(
    scan_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(ScanHistory).where(ScanHistory.id == scan_id, ScanHistory.user_id == current_user.id)
    )
    scan = result.scalar_one_or_none()
    if scan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบประวัติการสแกนนี้")

    await db.delete(scan)
    await db.commit()


@router.get("/export.csv")
async def export_scans_csv(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    result = await db.execute(
        select(ScanHistory)
        .where(ScanHistory.user_id == current_user.id)
        .order_by(ScanHistory.created_at.desc())
    )
    scans = result.scalars().all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "face_shape", "face_shape_name", "confidence", "gender_pref", "created_at"])
    for scan in scans:
        writer.writerow([
            scan.id, scan.face_shape.value, scan.face_shape_name,
            scan.confidence, scan.gender_pref or "", scan.created_at.isoformat(),
        ])
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scan_history.csv"},
    )
