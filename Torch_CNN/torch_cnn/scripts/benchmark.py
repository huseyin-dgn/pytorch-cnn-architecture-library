import os
import subprocess

def run(cmd: str):
    print("\n>>>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.makedirs("results", exist_ok=True)
    out_csv = "results/cifar10_quick.csv"
    run(f"python scripts/train.py --epochs 5 --conv standard --norm bn --act relu --no-attn --out_csv {out_csv}")
    run(f"python scripts/train.py --epochs 5 --conv standard --norm bn --act relu --attn --out_csv {out_csv}")
    run(f"python scripts/train.py --epochs 5 --conv dynamic --norm bn --act relu --attn --K 4 --out_csv {out_csv}")
    run(f"python scripts/train.py --epochs 5 --conv dynamic --norm bn --act silu --attn --K 4 --out_csv {out_csv}")
if __name__ == '__main__':
    main()