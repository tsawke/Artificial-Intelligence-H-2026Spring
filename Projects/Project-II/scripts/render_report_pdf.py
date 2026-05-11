import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "Project2_Report.md"
OUTPUT = PROJECT_ROOT / "Project2_Report.pdf"


def inline_markup(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def flush_paragraph(buffer, story, style):
    if not buffer:
        return
    story.append(Paragraph(inline_markup(" ".join(buffer)), style))
    story.append(Spacer(1, 0.08 * inch))
    buffer.clear()


def table_from_lines(lines):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append([Paragraph(inline_markup(cell), getSampleStyleSheet()["BodyText"]) for cell in cells])
    table = Table(rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def main():
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12
    styles["Code"].fontName = "Courier"
    styles["Code"].fontSize = 8
    styles["Code"].leading = 10

    story = []
    paragraph = []
    code = []
    table = []
    in_code = False
    in_math = False

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            flush_paragraph(paragraph, story, styles["BodyText"])
            if in_code:
                story.append(Preformatted("\n".join(code), styles["Code"]))
                story.append(Spacer(1, 0.08 * inch))
                code.clear()
                in_code = False
            else:
                in_code = True
            continue

        if line.strip() == "$$":
            flush_paragraph(paragraph, story, styles["BodyText"])
            if in_math:
                story.append(Preformatted("\n".join(code), styles["Code"]))
                story.append(Spacer(1, 0.08 * inch))
                code.clear()
                in_math = False
            else:
                in_math = True
            continue

        if in_code or in_math:
            code.append(line)
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_paragraph(paragraph, story, styles["BodyText"])
            table.append(line)
            continue
        if table:
            story.append(table_from_lines(table))
            story.append(Spacer(1, 0.12 * inch))
            table.clear()

        if not line.strip():
            flush_paragraph(paragraph, story, styles["BodyText"])
            continue

        if line.startswith("# "):
            flush_paragraph(paragraph, story, styles["BodyText"])
            story.append(Paragraph(inline_markup(line[2:]), styles["Title"]))
            story.append(Spacer(1, 0.16 * inch))
        elif line.startswith("## "):
            flush_paragraph(paragraph, story, styles["BodyText"])
            story.append(Paragraph(inline_markup(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            flush_paragraph(paragraph, story, styles["BodyText"])
            story.append(Paragraph(inline_markup(line[4:]), styles["Heading2"]))
        elif line.startswith("- "):
            flush_paragraph(paragraph, story, styles["BodyText"])
            story.append(Paragraph("&bull; " + inline_markup(line[2:]), styles["BodyText"]))
        else:
            paragraph.append(line)

    flush_paragraph(paragraph, story, styles["BodyText"])
    if table:
        story.append(table_from_lines(table))

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="CS303 Project 2 Report",
    )
    document.build(story)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
