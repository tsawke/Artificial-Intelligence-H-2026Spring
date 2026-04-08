#!/usr/bin/env python3
"""Generate a simple PDF report from the Markdown source."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=HexColor("#1f2937"),
            alignment=TA_LEFT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=HexColor("#111827"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=12,
            firstLineIndent=-8,
            bulletIndent=4,
            textColor=HexColor("#111827"),
            spaceAfter=2,
        )
    )
    return styles


def flush_paragraph(buffer, story, styles):
    if not buffer:
        return
    text = " ".join(part.strip() for part in buffer).strip()
    if text:
        story.append(Paragraph(text, styles["ReportBody"]))
    buffer.clear()


def markdown_to_story(markdown_text: str):
    styles = build_styles()
    story = []
    paragraph_buffer = []

    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Spacer(1, 2))
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[2:].strip(), styles["ReportTitle"]))
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[3:].strip(), styles["ReportHeading"]))
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[2:].strip(), styles["ReportBullet"], bulletText="-"))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph(paragraph_buffer, story, styles)
    return story


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_md", help="Path to the Markdown report source.")
    parser.add_argument("output_pdf", help="Path to the generated PDF.")
    args = parser.parse_args()

    input_path = Path(args.input_md)
    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_text = input_path.read_text(encoding="utf-8")
    story = markdown_to_story(markdown_text)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Project 1 Report",
        author="Codex",
    )
    document.build(story)


if __name__ == "__main__":
    main()
