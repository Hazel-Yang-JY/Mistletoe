import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


# =========================================================
# Configuration
# =========================================================
TRAIN_DIR = "./train"
VAL_DIR   = "./val"
WM_DIR    = r"./watermark"

MODEL_PATH = "./model/resnet50.pth"
SAVE_PATH  = "./model/resnet50_watermark.pth"

NUM_CLASSES = 1000
WM_CLASS    = "n07734744"

BATCH_SIZE = 64
EPOCHS     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP
USE_AMP = torch.cuda.is_available()
AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Attack / Recover Settings
# -----------------------------
ATTACK_STEPS_PER_ITER  = 1
RECOVER_STEPS_PER_ITER = 1

# Perform attack every iteration by default
ATTACK_EVERY = 1

WM_LR = 1e-4

LAMBDA_WM_EMBED   = 0.5
LAMBDA_WM_RECOVER = 1.0

# Alignment strength toward attack gradient direction
ALIGN_COEFF = 0.3

# EMA coefficient for attack gradients
ATTACK_EMA = 0.8

EPS = 1e-12

NUM_WORKERS_TRAIN = 4
NUM_WORKERS_WM    = 2

# Apply alignment only to later layers
ALIGN_KEYWORDS = ["layer3", "layer4", "fc"]

# DataLoader optimization
USE_PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2


# =========================================================
# Data Preprocessing
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================================================
# Datasets
# =========================================================
def build_datasets():
    train_set = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_set   = ImageFolder(VAL_DIR, transform=val_transform)

    if WM_CLASS not in train_set.class_to_idx:
        raise ValueError(
            f"WM_CLASS={WM_CLASS} not found in train_set.class_to_idx"
        )

    wm_label = train_set.class_to_idx[WM_CLASS]
    print(f"ℹ️  {WM_CLASS} label index = {wm_label}")

    wm_train = ImageFolder(WM_DIR, transform=train_transform)
    wm_val   = ImageFolder(WM_DIR, transform=val_transform)

    # Force all watermark samples to use the target label
    wm_train.targets = [wm_label] * len(wm_train)
    wm_train.samples = [(p, wm_label) for p, _ in wm_train.samples]

    wm_val.targets = [wm_label] * len(wm_val)
    wm_val.samples = [(p, wm_label) for p, _ in wm_val.samples]

    print(
        f"ℹ️  Main train set: {len(train_set)} images | "
        f"Main val set: {len(val_set)} images"
    )

    print(
        f"ℹ️  Watermark train set: {len(wm_train)} images | "
        f"Watermark val set: {len(wm_val)} images"
    )

    return train_set, wm_train, val_set, wm_val, wm_label


# =========================================================
# Model
# =========================================================
def strip_prefix_if_needed(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())

    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        new_state = {}

        for k, v in state_dict.items():
            new_state[k[len("module."):]] = v

        return new_state

    return state_dict


def load_model(model_path, num_classes, device):
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = strip_prefix_if_needed(state)

    net.load_state_dict(state, strict=True)
    net.to(device)

    return net


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(net, loader, device, desc="Val", use_amp=False):
    net.eval()

    correct, total = 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(AMP_DEVICE, enabled=use_amp):
            logits = net(imgs)

        preds = logits.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / max(total, 1)

    print(f"  [{desc}] Acc = {acc:.2f}% ({correct}/{total})")

    return acc


# =========================================================
# Utilities
# =========================================================
def cycle_next(it, loader):
    try:
        batch = next(it)

    except StopIteration:
        it = iter(loader)
        batch = next(it)

    return batch, it


def should_align_param(name, keywords):
    return any(k in name for k in keywords)


def collect_current_grads(net, only_keywords=None):
    grads = {}

    for name, p in net.named_parameters():
        if p.grad is None:
            continue

        if only_keywords is not None and not should_align_param(name, only_keywords):
            continue

        grads[name] = p.grad.detach().clone()

    return grads


def update_grad_ema(grad_ema, new_grads, momentum=0.8):
    if grad_ema is None:
        return {k: v.clone() for k, v in new_grads.items()}

    out = {}

    keys = set(grad_ema.keys()) | set(new_grads.keys())

    for k in keys:
        if k in grad_ema and k in new_grads:
            out[k] = momentum * grad_ema[k] + (1.0 - momentum) * new_grads[k]

        elif k in new_grads:
            out[k] = new_grads[k].clone()

        else:
            out[k] = grad_ema[k]

    return out


def apply_direction_alignment(
    net,
    ref_grads=None,
    align_coeff=0.0,
    eps=1e-12,
    only_keywords=None
):
    """
    Adjust current gradients toward the reference gradients.
    """

    if ref_grads is None or align_coeff <= 0.0:
        return

    for name, p in net.named_parameters():
        if p.grad is None:
            continue

        if only_keywords is not None and not should_align_param(name, only_keywords):
            continue

        if name not in ref_grads:
            continue

        g = p.grad
        a = ref_grads[name].to(device=g.device, dtype=g.dtype)

        a_norm2 = a.pow(2).sum()

        if a_norm2 > eps:
            proj = (g * a).sum() / (a_norm2 + eps) * a
            g_new = (1.0 - align_coeff) * g + align_coeff * proj
            p.grad.copy_(g_new)


# =========================================================
# Loss Functions
# =========================================================
def forward_joint_loss(
    net,
    criterion,
    main_imgs,
    main_labels,
    wm_imgs=None,
    wm_labels=None,
    lambda_wm=0.5,
    device="cuda"
):
    if wm_imgs is None or wm_labels is None:
        logits = net(main_imgs)

        per_loss = criterion(logits, main_labels)
        loss = per_loss.mean()

        preds = logits.detach().argmax(1)

        wm_loss_val = None

        return loss, preds, wm_loss_val

    n_main = main_labels.size(0)

    imgs_cat = torch.cat([main_imgs, wm_imgs], dim=0)
    labels_cat = torch.cat([main_labels, wm_labels], dim=0)

    weight = torch.ones(imgs_cat.size(0), device=device)
    weight[n_main:] = lambda_wm

    logits_cat = net(imgs_cat)

    per_loss = criterion(logits_cat, labels_cat)

    loss = (per_loss * weight).mean()

    main_preds = logits_cat[:n_main].detach().argmax(1)

    wm_loss_val = per_loss[n_main:].mean().item()

    return loss, main_preds, wm_loss_val


def forward_clean_loss(net, criterion, imgs, labels):
    logits = net(imgs)

    loss = criterion(logits, labels).mean()

    preds = logits.detach().argmax(1)

    return loss, preds


# =========================================================
# DataLoader Construction
# =========================================================
def build_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = USE_PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(**kwargs)


# =========================================================
# Training:
# Embed -> Attack (gradient only) -> Recover
# =========================================================
def train_attack_gradient_probe_fast(
    net,
    main_loader,
    wm_loader,
    val_loader,
    wm_val_loader,
    device
):
    criterion = nn.CrossEntropyLoss(reduction="none")

    wm_optimizer = optim.Adam(net.parameters(), lr=WM_LR)

    wm_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        wm_optimizer,
        T_max=EPOCHS
    )

    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=USE_AMP)

    best_clean_val = 0.0
    best_wm_val = 0.0

    attack_grad_ema = None

    global_iter = 0

    for epoch in range(1, EPOCHS + 1):
        net.train()

        main_iter = iter(main_loader)
        wm_iter = iter(wm_loader)

        total_embed_loss = 0.0
        total_attack_loss = 0.0
        total_recover_loss = 0.0

        main_correct = 0
        total_samples = 0

        pbar = tqdm(
            range(len(main_loader)),
            desc=f"Fast-Train [{epoch}/{EPOCHS}]",
            unit="iter",
            dynamic_ncols=True
        )

        for _ in pbar:
            global_iter += 1

            # -------------------------------------------------
            # Phase A: Embed
            # -------------------------------------------------
            (main_imgs, main_labels), main_iter = cycle_next(
                main_iter,
                main_loader
            )

            (wm_imgs, wm_labels), wm_iter = cycle_next(
                wm_iter,
                wm_loader
            )

            main_imgs = main_imgs.to(device, non_blocking=True)
            main_labels = main_labels.to(device, non_blocking=True)

            wm_imgs = wm_imgs.to(device, non_blocking=True)
            wm_labels = wm_labels.to(device, non_blocking=True)

            wm_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                embed_loss, _, wm_loss_embed = forward_joint_loss(
                    net,
                    criterion,
                    main_imgs,
                    main_labels,
                    wm_imgs,
                    wm_labels,
                    lambda_wm=LAMBDA_WM_EMBED,
                    device=device,
                )

            scaler.scale(embed_loss).backward()

            scaler.step(wm_optimizer)
            scaler.update()

            # -------------------------------------------------
            # Phase B: Attack
            # Gradient probing only
            # -------------------------------------------------
            attack_loss_val = 0.0

            attack_main_preds = None
            attack_main_labels = None

            did_attack = False

            if global_iter % ATTACK_EVERY == 0:
                did_attack = True

                for _attack_step in range(ATTACK_STEPS_PER_ITER):
                    (atk_imgs, atk_labels), main_iter = cycle_next(
                        main_iter,
                        main_loader
                    )

                    atk_imgs = atk_imgs.to(device, non_blocking=True)
                    atk_labels = atk_labels.to(device, non_blocking=True)

                    net.zero_grad(set_to_none=True)

                    with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                        attack_loss, attack_preds = forward_clean_loss(
                            net,
                            criterion,
                            atk_imgs,
                            atk_labels
                        )

                    attack_loss.backward()

                    cur_attack_grads = collect_current_grads(
                        net,
                        only_keywords=ALIGN_KEYWORDS
                    )

                    attack_grad_ema = update_grad_ema(
                        attack_grad_ema,
                        cur_attack_grads,
                        momentum=ATTACK_EMA
                    )

                    attack_loss_val += attack_loss.item()

                    attack_main_preds = attack_preds
                    attack_main_labels = atk_labels

                attack_loss_val /= max(ATTACK_STEPS_PER_ITER, 1)

                net.zero_grad(set_to_none=True)

            else:
                attack_loss_val = 0.0

            # -------------------------------------------------
            # Phase C: Recover
            # -------------------------------------------------
            recover_loss_val = 0.0
            recover_wm_loss_disp = None

            for _recover_step in range(RECOVER_STEPS_PER_ITER):
                (rec_main_imgs, rec_main_labels), main_iter = cycle_next(
                    main_iter,
                    main_loader
                )

                (rec_wm_imgs, rec_wm_labels), wm_iter = cycle_next(
                    wm_iter,
                    wm_loader
                )

                rec_main_imgs = rec_main_imgs.to(device, non_blocking=True)
                rec_main_labels = rec_main_labels.to(device, non_blocking=True)

                rec_wm_imgs = rec_wm_imgs.to(device, non_blocking=True)
                rec_wm_labels = rec_wm_labels.to(device, non_blocking=True)

                wm_optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                    recover_loss, _, wm_loss_recover = forward_joint_loss(
                        net,
                        criterion,
                        rec_main_imgs,
                        rec_main_labels,
                        rec_wm_imgs,
                        rec_wm_labels,
                        lambda_wm=LAMBDA_WM_RECOVER,
                        device=device,
                    )

                scaler.scale(recover_loss).backward()

                scaler.unscale_(wm_optimizer)

                apply_direction_alignment(
                    net,
                    ref_grads=attack_grad_ema,
                    align_coeff=ALIGN_COEFF,
                    eps=EPS,
                    only_keywords=ALIGN_KEYWORDS,
                )

                scaler.step(wm_optimizer)
                scaler.update()

                recover_loss_val += recover_loss.item()

                recover_wm_loss_disp = wm_loss_recover

            recover_loss_val /= max(RECOVER_STEPS_PER_ITER, 1)

            if attack_main_preds is not None:
                main_correct += (
                    attack_main_preds == attack_main_labels
                ).sum().item()

                total_samples += attack_main_labels.size(0)

            total_embed_loss += embed_loss.item()
            total_attack_loss += attack_loss_val
            total_recover_loss += recover_loss_val

            postfix = {
                "embed": f"{embed_loss.item():.4f}",
                "atk": f"{attack_loss_val:.4f}" if did_attack else "skip",
                "recover": f"{recover_loss_val:.4f}",
                "wm_e": (
                    f"{wm_loss_embed:.4f}"
                    if wm_loss_embed is not None else "—"
                ),
                "wm_r": (
                    f"{recover_wm_loss_disp:.4f}"
                    if recover_wm_loss_disp is not None else "—"
                ),
                "acc": (
                    f"{100.0 * main_correct / max(total_samples, 1):.2f}%"
                ),
            }

            pbar.set_postfix(postfix)

        pbar.close()

        wm_scheduler.step()

        print(
            f"  ↳ avg_embed={total_embed_loss / len(main_loader):.4f}  "
            f"avg_attack_grad={total_attack_loss / len(main_loader):.4f}  "
            f"avg_recover={total_recover_loss / len(main_loader):.4f}  "
            f"train_acc={100.0 * main_correct / max(total_samples, 1):.2f}%"
        )

        clean_val_acc = evaluate(
            net,
            val_loader,
            device,
            desc="Clean Val",
            use_amp=USE_AMP
        )

        wm_val_acc = evaluate(
            net,
            wm_val_loader,
            device,
            desc="Watermark Val",
            use_amp=USE_AMP
        )

        torch.save(net.state_dict(), SAVE_PATH)

        print(f"  💾 Saved model -> {SAVE_PATH}")

        if clean_val_acc > best_clean_val:
            best_clean_val = clean_val_acc

        if wm_val_acc > best_wm_val:
            best_wm_val = wm_val_acc

    print(f"\n✅ Training completed!")
    print(f"   Model saved to {SAVE_PATH}")
    print(f"   Best Clean Val Acc = {best_clean_val:.2f}%")
    print(f"   Best WM Val Acc    = {best_wm_val:.2f}%")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print(f"🖥️  Device: {DEVICE}")
    print(f"⚡ USE_AMP: {USE_AMP}")
    print(f"⚡ ALIGN_KEYWORDS: {ALIGN_KEYWORDS}")
    print(f"⚡ ATTACK_EVERY: {ATTACK_EVERY}\n")

    train_set, wm_train, val_set, wm_val, wm_label = build_datasets()

    main_loader = build_loader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        drop_last=False,
    )

    wm_loader = build_loader(
        dataset=wm_train,
        batch_size=min(BATCH_SIZE, len(wm_train)),
        shuffle=True,
        num_workers=NUM_WORKERS_WM,
        drop_last=False,
    )

    val_loader = build_loader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_TRAIN,
        drop_last=False,
    )

    wm_val_loader = build_loader(
        dataset=wm_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_WM,
        drop_last=False,
    )

    print(f"🔧 Loading base model: {MODEL_PATH}")

    net = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    print("\n📊 [Before Training]")
    evaluate(
        net,
        val_loader,
        DEVICE,
        desc="Clean Val (before)",
        use_amp=USE_AMP
    )

    evaluate(
        net,
        wm_val_loader,
        DEVICE,
        desc="Watermark Val (before)",
        use_amp=USE_AMP
    )

    print(
        f"\n🚀 Starting training:"
        f"\n   attack_steps         = {ATTACK_STEPS_PER_ITER}"
        f"\n   recover_steps        = {RECOVER_STEPS_PER_ITER}"
        f"\n   attack_every         = {ATTACK_EVERY}"
        f"\n   wm_lr                = {WM_LR}"
        f"\n   lambda_wm_embed      = {LAMBDA_WM_EMBED}"
        f"\n   lambda_wm_recover    = {LAMBDA_WM_RECOVER}"
        f"\n   align_coeff          = {ALIGN_COEFF}"
        f"\n   attack_ema           = {ATTACK_EMA}"
        f"\n   align_layers         = {ALIGN_KEYWORDS}"
        f"\n   use_amp              = {USE_AMP}"
        f"\n"
    )

    train_attack_gradient_probe_fast(
        net,
        main_loader,
        wm_loader,
        val_loader,
        wm_val_loader,
        DEVICE
    )