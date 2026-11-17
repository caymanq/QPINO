"""Resumable scheduler for results/metrics/m4/jobs.txt: launches 1-thread jobs while fewer than
MAXJ m4 jobs run and enough memory is free; skips jobs whose log file already exists (finished or
running)."""
import os, subprocess, time
MAXJ = int(os.environ.get("MAXJ", 7)); MINFREE_MB = int(os.environ.get("MINFREE_MB", 3500))
SCRIPT = os.environ.get("SCRIPT", "m4_experiments.py")
LOG = os.environ.get("LOGDIR", "../results/metrics/m4/logs")
jobs = [l.strip() for l in open(os.environ.get("JOBS", "../results/metrics/m4/jobs.txt")) if l.strip()]
def name(a): return a.replace("-", "").replace(" ", "_")
def running(): return int(subprocess.run("pgrep -fc -E '^python3 (m4|usa2d)_experiments.py'", shell=True, capture_output=True, text=True).stdout.strip() or 0)
def free_mb(): return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024
for a in jobs:
    log = f"{LOG}/{name(a)}.log"
    if os.path.exists(log):
        continue
    while running() >= MAXJ or free_mb() < MINFREE_MB:
        time.sleep(20)
    th = os.environ.get("THREADS") or ("2" if ("synthetic" in a and ("--R 2" in a or "--R 3" in a)) else "1")
    env = dict(os.environ, QPINO_THREADS=th, OMP_NUM_THREADS=th)
    subprocess.Popen(f"exec python3 {SCRIPT} {a} > {log} 2>&1", shell=True, env=env)
    print(time.strftime("%H:%M:%S"), "launched", a, flush=True)
    time.sleep(15)
print("ALL_LAUNCHED", flush=True)
