"""LunaDNA: ResNet18 backbone -> 512-D L2-normalized embeddings, triplet loss
training with recall@K validation — per 02_TRD Module 3 and 08_Model_Training."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LUNAR_MEAN = [0.5]
LUNAR_STD = [0.5]


class LunaDNA(nn.Module):
    """ResNet18 with classification head removed -> 512-D embedding."""

    def __init__(self, pretrained: bool = True, grayscale: bool = True) -> None:
        super().__init__()
        try:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvm.resnet18(weights=weights)
        except Exception:
            backbone = tvm.resnet18(weights=None)
        if grayscale:
            old_w = backbone.conv1.weight.data.clone()   # (64,3,7,7) ImageNet filters
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # preserve pretrained edge filters: average the RGB channels into 1
            with torch.no_grad():
                new_conv.weight.copy_(old_w.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.grayscale = grayscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return nn.functional.normalize(feat, p=2, dim=1)

    @property
    def embedding_dim(self) -> int:
        return 512


class PatchDataset(Dataset):
    def __init__(self, paths: list[str], size: int = 224, grayscale: bool = True,
                 augment: bool = False) -> None:
        self.paths = paths
        self.size = size
        self.grayscale = grayscale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str):
        img = Image.open(path)
        if self.grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        img = self._load(self.paths[idx])
        if self.augment:
            import torchvision.transforms.functional as TF
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            if random.random() < 0.3:
                arr = np.array(img)
                arr = arr + np.random.normal(0, 5, arr.shape).astype(np.float32)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        import torchvision.transforms as T
        mean = LUNAR_MEAN if self.grayscale else IMAGENET_MEAN
        std = LUNAR_STD if self.grayscale else IMAGENET_STD
        tf = T.Compose([T.Resize((self.size, self.size)), T.ToTensor(), T.Normalize(mean, std)])
        return tf(img)


class TripletDataset(Dataset):
    def __init__(self, anchors: list[str], positives: list[str], negatives: list[str],
                 size: int = 224, augment: bool = True) -> None:
        self.anchor_ds = PatchDataset(anchors, size=size, augment=augment)
        self.pos_ds = PatchDataset(positives, size=size, augment=augment)
        self.neg_ds = PatchDataset(negatives, size=size, augment=augment)

    def __len__(self) -> int:
        return len(self.anchor_ds)

    def __getitem__(self, idx: int):
        return (self.anchor_ds[idx], self.pos_ds[idx], self.neg_ds[idx])


@torch.no_grad()
def compute_embeddings(model: LunaDNA, paths: list[str], device: str = "cpu",
                       batch_size: int = 64, size: int = 224) -> np.ndarray:
    model.eval()
    ds = PatchDataset(paths, size=size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    outs = []
    for batch in loader:
        emb = model(batch.to(device))
        outs.append(emb.cpu().numpy())
    if not outs:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)
    embs = np.concatenate(outs).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def make_geo_triplets(df, n_negatives: int = 1, seed: int = 42) -> list[tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets using geo-proximity positives.

    df must have columns: patch_path, latitude, longitude, dataset_name.
    Positives: patches from a *different* sensor whose lat/lon are within ~0.5 deg
    (gracefully relaxed up to 8 deg; for near-polar imagery the longitude spread
    compresses geo distance, so a distance ladder beats a hard cutoff).
    Negatives: patches from a different sensor chosen far away (> 10 deg),
    falling back to any cross-sensor patch.
    NaN-coordinate patches are skipped.
    """
    rng = random.Random(seed)
    rows = df.reset_index(drop=True)
    coords = rows[["latitude", "longitude"]].to_numpy(dtype=float)
    sensors = rows["dataset_name"].tolist()
    paths = rows["patch_path"].tolist()

    valid = np.isfinite(coords).all(axis=1)
    idx_all = [i for i in range(len(rows)) if valid[i]]

    def geo_dist(a: int, b: int) -> float:
        d = coords[a] - coords[b]
        return float(np.hypot(d[0], d[1]))

    triplets: list[tuple[str, str, str]] = []
    for i in idx_all:
        a_sensor, a_path = sensors[i], paths[i]
        cross = [j for j in idx_all if j != i and sensors[j] != a_sensor]
        if not cross:
            continue
        pos_idx = [j for j in cross if geo_dist(i, j) < 0.5]
        for radius in (1.5, 3.0, 8.0):
            if pos_idx:
                break
            pos_idx = [j for j in cross if geo_dist(i, j) < radius]
        if not pos_idx:
            continue
        p_idx = rng.choice(pos_idx)
        far = [j for j in cross if geo_dist(i, j) > 10.0]
        if not far:
            far = [j for j in cross if j != p_idx]
        for _ in range(n_negatives):
            n_idx = rng.choice(far)
            triplets.append((a_path, paths[p_idx], paths[n_idx]))
    return triplets


def train_lunadna(model: LunaDNA, train_triplets: list[tuple[str, str, str]],
                  val_triplets: list[tuple[str, str, str]], cfg: dict,
                  device: str, out_dir: Path,
                  log_cb: Callable[[str], None] = print) -> dict:
    """Triplet-loss training loop with early stopping and checkpoints.

    Optional cfg keys: "time_budget_s" (wall-clock; loop exits cleanly when the
    budget is exhausted — the best checkpoint so far is kept) and
    "stop_flag" (a dict with key "expired" set True to request a stop).
    """
    import time as _time

    epochs = int(cfg.get("epochs", 60))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-5))
    margin = float(cfg.get("margin", 0.3))
    patience = int(cfg.get("patience", 8))
    size = int(cfg.get("input_size", 224))
    workers = int(cfg.get("num_workers", 0))
    budget_s = cfg.get("time_budget_s")
    stop_flag = cfg.get("stop_flag")
    deadline = (_time.perf_counter() + float(budget_s)) if budget_s else None

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def make_loader(triplets: list[tuple[str, str, str]], augment: bool) -> DataLoader:
        a = [t[0] for t in triplets]
        p = [t[1] for t in triplets]
        n = [t[2] for t in triplets]
        ds = TripletDataset(a, p, n, size=size, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=workers)

    train_loader = make_loader(train_triplets, augment=True)
    val_loader = make_loader(val_triplets, augment=False)

    history: list[dict] = []
    best_val = math.inf
    best_epoch = -1
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "lunadna.pt"

    for epoch in range(1, epochs + 1):
        if deadline is not None and _time.perf_counter() > deadline:
            log_cb(f"time budget exhausted before epoch {epoch}; stopping")
            break
        if stop_flag is not None and stop_flag.get("expired"):
            log_cb(f"stop requested before epoch {epoch}; stopping")
            break
        model.train()
        tr_losses = []
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            opt.zero_grad()
            ea, ep, en = model(a), model(p), model(n)
            loss = loss_fn(ea, ep, en)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        model.eval()
        va_losses = []
        with torch.no_grad():
            for a, p, n in val_loader:
                ea, ep, en = model(a.to(device)), model(p.to(device)), model(n.to(device))
                va_losses.append(loss_fn(ea, ep, en).item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        log_cb(f"epoch {epoch:03d} train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "config": cfg}, best_path)
        if epoch - best_epoch >= patience:
            log_cb(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    return {"history": history, "best_val_loss": best_val, "best_epoch": best_epoch,
            "model_path": str(best_path)}

def make_adjacent_triplets(df, seed: int = 42, max_triplets: int | None = None):
    """Build triplets from *adjacent patches of the same image* as positives.

    Patches are extracted on a 128-px grid with 256-px windows, so neighbouring
    patches overlap by 50% and cover genuinely the same terrain — a well-posed
    "same lunar region" signal. Negatives come from a different source image.

    This is the primary training objective: with no truly overlapping cross-sensor
    imagery in the distribution, geo-proximity would teach the trivial collapse.
    """
    rng = random.Random(seed)
    rows = df.dropna(subset=["patch_path", "row", "col"]).reset_index(drop=True)
    by_image = {name: grp.reset_index(drop=True)
                for name, grp in rows.groupby("source_image")}
    image_names = [k for k, v in by_image.items() if len(v) >= 2]
    if len(image_names) < 2:
        return []
    triplets: list[tuple[str, str, str]] = []
    for img in image_names:
        g = by_image[img]
        rows_arr = g["row"].to_numpy(dtype=float)
        cols_arr = g["col"].to_numpy(dtype=float)
        others = [k for k in image_names if k != img]
        for i in range(len(g)):
            d = np.abs(rows_arr - rows_arr[i]) + np.abs(cols_arr - cols_arr[i])
            d[i] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]):
                continue
            neg_img = rng.choice(others)
            ng = by_image[neg_img]
            k = rng.randrange(len(ng))
            triplets.append((g.patch_path.iloc[i], g.patch_path.iloc[j],
                             ng.patch_path.iloc[k]))
    if max_triplets and len(triplets) > max_triplets:
        sel = np.random.default_rng(seed).choice(len(triplets), max_triplets, replace=False)
        triplets = [triplets[i] for i in sorted(sel)]
    return triplets
