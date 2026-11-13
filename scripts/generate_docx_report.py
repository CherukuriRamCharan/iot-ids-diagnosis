"""Standalone Academic Document Generator for IoT IDS Project.

Generates docs/PROJECT_RESULTS_AND_PERFORMANCE_ANALYSIS.docx directly from
docs/PROJECT_RESULTS_AND_PERFORMANCE_ANALYSIS.md using pure Python standard library zipfile & OpenXML.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
OUT_DOCX = DOCS_DIR / "PROJECT_RESULTS_AND_PERFORMANCE_ANALYSIS.docx"


def escape_xml(text: str) -> str:
    """Escapes XML entities."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_docx_via_openxml(out_path: Path) -> None:
    """Builds a beautifully styled Word Document (.docx) using OpenXML OPC packaging."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:color w:val="1E293B"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="44"/>
      <w:color w:val="0F172A"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="30"/>
      <w:color w:val="1E293B"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="24"/>
      <w:color w:val="2563EB"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    md_file = DOCS_DIR / "PROJECT_RESULTS_AND_PERFORMANCE_ANALYSIS.md"
    content_lines = md_file.read_text(encoding="utf-8").splitlines() if md_file.exists() else []

    doc_xml_parts = [
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>"""
    ]

    def p(text: str, style: str = "", bold: bool = False, color: str = "", italic: bool = False) -> str:
        s_tag = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        r_pr = ""
        if bold or color or italic:
            r_pr = "<w:rPr>"
            if bold:
                r_pr += "<w:b/>"
            if italic:
                r_pr += "<w:i/>"
            if color:
                r_pr += f'<w:color w:val="{color}"/>'
            r_pr += "</w:rPr>"
        safe_text = escape_xml(text)
        return f"<w:p>{s_tag}<w:r>{r_pr}<w:t>{safe_text}</w:t></w:r></w:p>"

    def render_table(table_rows: List[str]) -> str:
        if not table_rows:
            return ""
        parsed = []
        for r in table_rows:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            # Skip divider line like |---|---|
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            parsed.append(cells)

        if not parsed:
            return ""

        tbl = ["""<w:tbl>
          <w:tblPr>
            <w:tblW w:w="0" w:type="auto"/>
            <w:tblBorders>
              <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
              <w:left w:val="none"/>
              <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
              <w:right w:val="none"/>
              <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
              <w:insideV w:val="none"/>
            </w:tblBorders>
          </w:tblPr>"""]

        for row_idx, row in enumerate(parsed):
            is_header = (row_idx == 0)
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>' if is_header else ""
            tbl.append("<w:tr>")
            for cell in row:
                tbl.append(f"""<w:tc>
                  <w:tcPr>{shd}<w:tcMar><w:top w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/><w:left w:w="150" w:type="dxa"/><w:right w:w="150" w:type="dxa"/></w:tcMar></w:tcPr>
                  {p(cell, bold=is_header, color="0F172A" if is_header else "334155")}
                </w:tc>""")
            tbl.append("</w:tr>")
        tbl.append("</w:tbl>")
        return "\n".join(tbl)

    # Process markdown lines
    idx = 0
    while idx < len(content_lines):
        line = content_lines[idx].strip()

        if not line:
            doc_xml_parts.append(p("", style=""))
            idx += 1
            continue

        # Check for Markdown Table block
        if line.startswith("|") and idx + 1 < len(content_lines) and "|" in content_lines[idx + 1]:
            t_rows = []
            while idx < len(content_lines) and content_lines[idx].strip().startswith("|"):
                t_rows.append(content_lines[idx].strip())
                idx += 1
            doc_xml_parts.append(render_table(t_rows))
            continue

        # Headers
        if line.startswith("# "):
            doc_xml_parts.append(p(line[2:], "Title", bold=True, color="0F172A"))
        elif line.startswith("## "):
            doc_xml_parts.append(p(line[3:], "Heading1", bold=True, color="1E293B"))
        elif line.startswith("### "):
            doc_xml_parts.append(p(line[4:], "Heading2", bold=True, color="2563EB"))
        elif line.startswith("#### "):
            doc_xml_parts.append(p(line[5:], style="", bold=True, color="0F172A"))

        # Figures Callout
        elif "[Figure " in line:
            doc_xml_parts.append(p("--------------------------------------------------------------------------------", style="", color="CBD5E1"))
            doc_xml_parts.append(p(line, style="", bold=True, color="2563EB"))
            doc_xml_parts.append(p("--------------------------------------------------------------------------------", style="", color="CBD5E1"))

        # Blockquotes / Alerts
        elif line.startswith("> "):
            clean_q = line[2:]
            if clean_q.startswith("[!"):
                doc_xml_parts.append(p(clean_q, style="", bold=True, color="B91C1C"))
            else:
                doc_xml_parts.append(p(clean_q, style="", italic=True, color="475569"))

        # Normal Paragraphs
        else:
            doc_xml_parts.append(p(line, style="", color="1E293B"))

        idx += 1

    doc_xml_parts.append("""    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>""")

    doc_xml = "\n".join(doc_xml_parts)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("word/document.xml", doc_xml)

    file_size = out_path.stat().st_size
    print(f"[SUCCESS] Generated OpenXML Word Document: {out_path} ({file_size:,} bytes)")


def main() -> int:
    build_docx_via_openxml(OUT_DOCX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
