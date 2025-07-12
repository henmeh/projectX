import subprocess
import sys

scripts = ["analyze_mempool.py", "predicting_fees.py"]

processes = []
for script in scripts:
    p = subprocess.Popen([sys.executable, script])
    processes.append(p)  # Optional: keep references

# Optional: Keep launcher running to monitor processes
try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    for p in processes:
        p.terminate()