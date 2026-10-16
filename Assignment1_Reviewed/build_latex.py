"""Regenerate report tables and compile native LaTeX to the reviewed PDF."""
from pathlib import Path
import os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
(ROOT/'build').mkdir(exist_ok=True)
subprocess.run([sys.executable,'gen_tables.py'],check=True)
# Prefer the project-local portable compiler, then an installed compiler.
compiler=str(ROOT/'tectonic.exe') if (ROOT/'tectonic.exe').exists() else shutil.which('tectonic')
if compiler:
    command=[compiler,'report.tex','--outdir','build','--keep-logs']
    with open('build/compiler.log','w',encoding='utf-8') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
else:
    compiler=shutil.which('pdflatex')
    if not compiler: raise SystemExit('Install Tectonic or a LaTeX distribution with pdfLaTeX, then rerun this script.')
    for _ in range(2):
        subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','-output-directory=build','report.tex'],check=True)
output=ROOT/'EECE5644_Assignment1_Reviewed.pdf'
shutil.copyfile(ROOT/'build/report.pdf',output)
print(output)
