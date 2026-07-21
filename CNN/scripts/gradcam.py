### Grad-CAM on the spectrogram CNN: which time-frequency regions drive the ###
### model's decision -- drone signal, or a receiver/DC artifact? ###
### Uses a saved model (default the clean-trained transfer model, whose focus ###
### reflects the drone signal itself rather than interference background) and  ###
### overlays the class-activation map on one clean spectrogram per drone.      ###

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import SmallCNN  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SPECS_NPY = SCRIPT_DIR / ".." / "results" / "spectrograms.npy"
SPEC_META = SCRIPT_DIR / ".." / "results" / "spectrogram_meta.parquet"
MODEL_PT = SCRIPT_DIR / ".." / "models" / "transfer_clean.pt"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP = 0.05
FS_MHZ = 60.0                         # sample rate -> full span of the shifted axis
DUR_MS = 50.0
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def grad_cam(model, x, target):
    """Grad-CAM over the last conv block for a single (1,1,256,128) input."""
    acts, grads = {}, {}
    layer = model.features[3]
    # hooks must return None (returning a value would replace the output/grad)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))

    model.zero_grad()
    logits = model(x)
    logits[0, target].backward()
    h1.remove()
    h2.remove()

    w = grads["v"].mean(dim=(2, 3), keepdim=True)          # channel weights
    cam = F.relu((w * acts["v"]).sum(dim=1, keepdim=True))  # (1,1,16,8)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8)


def main():
    meta = pd.read_parquet(SPEC_META)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    specs = np.load(SPECS_NPY)[keep].astype(np.float32)

    ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    model = SmallCNN(len(DRONE_ORDER))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded model {MODEL_PT.name} (trained on {ckpt.get('train_condition')})")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), facecolor=SURFACE)
    extent = [0, DUR_MS, -FS_MHZ / 2, FS_MHZ / 2]
    for ax, drone in zip(axes.flat, DRONE_ORDER):
        # first clean segment of this drone
        idx = np.where((meta["drone_id"] == drone) &
                       (meta["interference"] == "clean"))[0][0]
        spec = specs[idx]
        x = torch.from_numpy(zscore(spec)).unsqueeze(0).unsqueeze(0)
        cls = DRONE_ORDER.index(drone)
        cam = grad_cam(model, x, cls)
        pred = DRONE_ORDER[int(model(x).argmax())]

        ax.imshow(spec, aspect="auto", origin="lower", extent=extent, cmap="gray")
        ax.imshow(cam, aspect="auto", origin="lower", extent=extent, cmap="inferno", alpha=0.5)
        ax.set_title(f"{drone}  (pred {pred})", fontsize=10,
                     color=INK if pred == drone else "#e34948", loc="left")
        ax.set_xlabel("time (ms)", fontsize=8, color=INK2)
        ax.set_ylabel("freq (MHz)", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Grad-CAM on the clean-trained CNN — where the model looks "
                 "(overlay = class activation)", fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS_DIR / "gradcam.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
