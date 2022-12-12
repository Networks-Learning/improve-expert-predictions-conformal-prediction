import subprocess

base_args = ["python3", "-m", "scripts.single.synthetic.coverage"]

split =.15
for n_labels in [10, 50, 100]:
    inline_args = ["--cal_split", f"{split}", "--n_labels", f"{n_labels}", "--runs", "100"]
    args = base_args+inline_args
    subprocess.run(args)