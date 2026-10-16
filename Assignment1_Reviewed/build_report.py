"""Compatibility entry point: build the report with native LaTeX."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().with_name('build_latex.py')), run_name='__main__')
