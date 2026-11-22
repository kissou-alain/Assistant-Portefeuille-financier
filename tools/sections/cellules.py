"""Aides pour écrire les sections du notebook."""
import textwrap

import nbformat


def md(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip("\n"))
