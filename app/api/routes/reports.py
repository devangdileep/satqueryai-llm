import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from app.core.config import settings
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1/jobs", tags=["Reports"])


@router.post("/{job_id}/report")
async def generate_job_report(job_id: str):
    """POST /api/v1/jobs/{job_id}/report: Generates an executive ISRO-styled PDF intelligence summary report."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    res = job.get("result")
    if not res:
        raise HTTPException(status_code=400, detail="Job is not completed yet.")

    res_dict = res.model_dump()

    reports_dir = Path(settings.STORAGE_PATH) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"SatQuery_Report_{job_id}.pdf"

    try:
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle("RTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
        body_style = ParagraphStyle("RBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#1e293b"))

        story = []
        story.append(Paragraph("SatQuery AI — Remote Sensing Intelligence Report", title_style))
        story.append(Paragraph(f"<b>Job ID:</b> {job_id} | <b>Task:</b> {res_dict.get('task')} | <b>Generated:</b> 2026-09-01", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

        story.append(Paragraph(f"<b>Executive Summary Answer:</b>", body_style))
        story.append(Paragraph(res_dict.get("answer", ""), body_style))
        story.append(Spacer(1, 10))

        conf = res_dict.get("confidence", {})
        story.append(Paragraph(f"<b>Confidence Score:</b> {conf.get('score')} ({conf.get('level').upper()}) — {conf.get('label')}", body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Observable Execution Trace Summary:</b>", body_style))
        exec_sum = res_dict.get("execution_summary", {})
        trace_data = [
            [Paragraph("<b>Task</b>", body_style), Paragraph("<b>Models Selected</b>", body_style), Paragraph("<b>Tools Executed</b>", body_style), Paragraph("<b>Latency</b>", body_style)],
            [
                Paragraph(str(exec_sum.get("task")), body_style),
                Paragraph(", ".join(exec_sum.get("models", [])), body_style),
                Paragraph(", ".join(exec_sum.get("tools", [])), body_style),
                Paragraph(f"{exec_sum.get('processing_time_ms')} ms", body_style)
            ]
        ]
        t = Table(trace_data, colWidths=[130, 130, 180, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(t)

        doc.build(story)

        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=f"SatQuery_Report_{job_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
