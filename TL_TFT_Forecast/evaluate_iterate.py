import subprocess
import sys

for year in range(2005, 2018, 2):
    with open('config.py', 'r', encoding='utf-8') as f:
        lines = [f'START_YEAR = {year}\n' if l.strip().startswith('START_YEAR') else l for l in f]
    with open('config.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    subprocess.run([sys.executable, 'evaluate.py'], check=True)