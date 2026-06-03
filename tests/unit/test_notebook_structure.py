"""Structural checks for the teaching notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


def test_demo_notebook_alternates_markdown_and_code_cells() -> None:
    """Notebook cells should alternate Markdown and code with no hidden blocks."""
    notebook_path = Path("notebooks/demo.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)

    # Ignore empty cells when checking pedagogical structure.
    meaningful_cells = [cell for cell in notebook.cells if cell.source.strip()]
    assert meaningful_cells, "Notebook must contain non-empty cells."

    expected_sequence = ["markdown", "code"]
    for cell_index, cell in enumerate(meaningful_cells):
        expected_type = expected_sequence[cell_index % 2]
        assert cell.cell_type == expected_type, (
            f"Cell {cell_index} should be {expected_type}, got {cell.cell_type}."
        )


def test_demo_notebook_contains_required_teaching_sections() -> None:
    """Notebook should include all required section headers for the offline story."""
    notebook_path = Path("notebooks/demo.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)

    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    ).lower()

    required_phrases = [
        "project framing and offline contract",
        "raw parquet inspection",
        "regular-session filtering and daily close construction",
        "daily log-return construction",
        "realized variance construction",
        "feature building",
        "training and backtest setup",
        "model execution",
        "evaluation metrics",
        "interpretation and interview narrative",
    ]

    for phrase in required_phrases:
        assert phrase in markdown_text, f"Missing required section phrase: {phrase}"
