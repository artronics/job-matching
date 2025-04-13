from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

CACHE_DIR = Path(f"{Path(__file__).parent.resolve()}/.report")


@dataclass
class ReportResult:
    title: str
    content: str


@dataclass
class ReportItem:
    title: str
    results: List[ReportResult]
    content: str = ""


@dataclass
class Report:
    title: str
    items: List[ReportItem]
    overview: str = ""
    usage: str = ""


def generate_markdown_report(report: Report) -> str:
    lines = [
        f"# {report.title}",
        "",
        report.overview
    ]
    if report.usage:
        lines.append(f"## Usage")
        lines.append(report.usage)

    lines.append("# Results")

    for item in report.items:
        lines.append(f"## {item.title}")
        lines.append("")
        lines.append(item.content)
        lines.append("")

        for result in item.results:
            lines.append(f"### {result.title}")
            lines.append("")
            lines.append(result.content)
            lines.append("")

        lines.append("---")

    return "\n".join(lines)


def write_markdown_report(filename: str, report: Report):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    file = CACHE_DIR / filename
    with open(file, "w") as f:
        f.write(generate_markdown_report(report))


def make_similarities_table(matches: List[Tuple[str, float]], limit: int = 5) -> str:
    rows = [(title, f"{score:.4f}") for title, score in matches[:limit]]

    headers = ("Title", "Similarity")
    all_rows = [headers] + rows

    col_widths = [
        max(len(str(row[0])) for row in all_rows),
        max(len(str(row[1])) for row in all_rows)
    ]

    # Helper to format a row with padding
    def format_row(row):
        return f"| {row[0]:<{col_widths[0]}} | {row[1]:^{col_widths[1]}} |"

    table_lines = [
        format_row(headers),
        f"|{'-' * col_widths[0]}--|{'-' * col_widths[1]}--|"
    ]
    for row in rows:
        table_lines.append(format_row(row))

    return "\n".join(table_lines)
