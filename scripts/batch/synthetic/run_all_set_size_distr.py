import subprocess

base_args = ["python3", "-m", "scripts.single.synthetic.set_size_distribution"]

for n_labels in [10, 50, 100]:
    for split in [0.02, 0.05, 0.1, 0.15]:
        inline_args = ["--cal_split", f"{split}", "--n_labels", f"{n_labels}"]
        args = base_args+inline_args
        subprocess.run(args)