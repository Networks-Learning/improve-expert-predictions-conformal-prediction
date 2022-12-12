import subprocess

base_args = ["python3", "-m", "scripts.single.real.topk", "--runs", "10"]

for split in [0.02, 0.05, 0.1, 0.15]:
    for k in range(2,10):
        args = base_args+["--cal_split", f"{split}", "--topk", f"{k}"]
        subprocess.run(args)