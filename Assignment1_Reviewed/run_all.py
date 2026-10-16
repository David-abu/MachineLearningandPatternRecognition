"""Run all experiments from any working directory; outputs are saved beside this file."""
from pathlib import Path
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
os.chdir(ROOT)
for name in ['data', 'out', 'figs', 'logs']:
    (ROOT / name).mkdir(exist_ok=True)
if not (ROOT / 'data/wine_white.csv').exists() or not (ROOT / 'data/har/X_train.txt').exists():
    subprocess.run([sys.executable, 'download_data.py'], check=True)
for name in ['q1.py', 'q2.py', 'q3.py', 'verify_results.py']:
    print('Running', name, flush=True)
    with open(ROOT / 'logs' / (Path(name).stem + '.log'), 'w', encoding='utf-8') as log:
        subprocess.run([sys.executable, name], stdout=log, stderr=subprocess.STDOUT, check=True)
print('Finished. See out/ for results and figs/ for figures. Run build_report.py separately to rebuild the PDF.')
