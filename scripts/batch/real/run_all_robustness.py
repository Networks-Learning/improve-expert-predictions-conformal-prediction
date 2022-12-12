import subprocess

base_args = ["python3", "-m", "scripts.single.real.robustness_analysis", "--runs", "10"]

for split in [0.02, 0.05, 0.1, 0.15]:
    args = base_args+["--cal_split", f"{split}"]
    subprocess.run(args)