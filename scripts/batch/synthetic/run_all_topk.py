import subprocess

base_args = ["python3", "-m", "scripts.single.synthetic.topk"]

for n_labels in [10, 50, 100]:
    for split in [0.02, 0.05, 0.1, 0.15]:
        for k in range(2,10):
            inline_args = ["--cal_split", f"{split}", "--n_labels", f"{n_labels}","--topk",f"{k}"]
            args = base_args+inline_args
            subprocess.run(args)