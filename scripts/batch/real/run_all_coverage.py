import subprocess

base_args = ["python3", "-m", "scripts.single.real.coverage"]

split =.15
inline_args = ["--cal_split", f"{split}", "--runs", "100"]
args = base_args+inline_args
subprocess.run(args)