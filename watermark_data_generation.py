import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.utils import save_image

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEED_IMAGE    = "seed_point.jpeg"
BASE_OUT_DIR  = r"./watermark"
MODEL_PATH    = "resnet50.pth"
TRAIN_DIR     = r"./train"
NUM_SAMPLES   = 10
PERTURB_DIM   = 4
EPSILON_RANGE = (-0.1, 0.1)
NUM_CLASSES   = 1000
TOPK_CANDIDATES = 50
SEED_CLASS = "n02981792"
USE_MANUAL_CLASS = True
MAIN_CLASS_SAMPLE_SIZE = 1000
MAIN_CLASS_RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numerical stability constant
EPS = 1e-12

# Standard preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


def load_model(model_path, num_classes, device):
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove possible "module." prefix
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        cleaned[k] = v

    net.load_state_dict(cleaned, strict=True)
    net.to(device)
    net.eval()
    return net


def load_image_tensor(image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor


def get_seed_logits_and_candidates(image_path, net, device, topk=10):
    img_tensor = load_image_tensor(image_path, device)
    with torch.no_grad():
        logits = net(img_tensor)
        probs = torch.softmax(logits, dim=1)
        _, topk_indices = torch.topk(logits, topk, dim=1)
        top1_idx = logits.argmax(dim=1).item()

    return logits, probs, topk_indices[0].tolist(), top1_idx


def generate_and_save(seed_image, out_dir, num_samples, perturb_dim, epsilon_range):
    from tools.vae import get_rec_image
    from tools.augment_perturb import augment

    os.makedirs(out_dir, exist_ok=True)
    generated_paths = []

    for i in range(num_samples):
        eps = random.uniform(*epsilon_range)
        base_name = f"wm_{i+1:03d}.jpeg"
        out_path = os.path.join(out_dir, base_name)

        best_sample = get_rec_image(
            seed_image,
            out_path,
            perturb_dim=perturb_dim,
            epsilon=eps
        )
        generated_paths.append(out_path)

        for j in range(9):
            aug_path = os.path.join(out_dir, f"wm_{i+1:03d}_aug_{j+1:03d}.png")
            aug_img = augment(best_sample)
            save_image(aug_img, aug_path)
            generated_paths.append(aug_path)

    return generated_paths


def flatten_grad_list(grad_list):
    """Flatten a list of gradients into a single 1D vector."""
    flat = []
    for g in grad_list:
        if g is None:
            continue
        flat.append(g.reshape(-1))

    if len(flat) == 0:
        return None

    return torch.cat(flat)


def compute_single_image_gradient(net, image_tensor, target_label, device):
    net.zero_grad(set_to_none=True)
    logits = net(image_tensor)
    target = torch.tensor([target_label], dtype=torch.long, device=device)
    loss = nn.CrossEntropyLoss()(logits, target)

    params = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True
    )

    grad_vec = flatten_grad_list(grads)
    return grad_vec.detach() if grad_vec is not None else None


def compute_dataset_avg_gradient(net, image_paths, target_label, device):
    grad_sum = None
    valid_count = 0

    for path in image_paths:
        try:
            img_tensor = load_image_tensor(path, device)
            grad_vec = compute_single_image_gradient(
                net,
                img_tensor,
                target_label,
                device
            )

            if grad_vec is None:
                continue

            if grad_sum is None:
                grad_sum = grad_vec.clone()
            else:
                grad_sum += grad_vec

            valid_count += 1

        except Exception as e:
            print(f"⚠️ Skipping corrupted or unreadable image: {path} | {e}")

    if grad_sum is None or valid_count == 0:
        return None

    return grad_sum / valid_count


def compute_labeled_dataset_avg_gradient(net, labeled_samples, device):
    grad_sum = None
    valid_count = 0

    for path, target_label in labeled_samples:
        try:
            img_tensor = load_image_tensor(path, device)
            grad_vec = compute_single_image_gradient(
                net,
                img_tensor,
                target_label,
                device
            )

            if grad_vec is None:
                continue

            if grad_sum is None:
                grad_sum = grad_vec.clone()
            else:
                grad_sum += grad_vec

            valid_count += 1

        except Exception as e:
            print(f"⚠️ Skipping corrupted or unreadable image: {path} | {e}")

    if grad_sum is None or valid_count == 0:
        return None

    return grad_sum / valid_count


def list_images_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        return []

    files = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and name.endswith(VALID_EXTS):
            files.append(path)

    return sorted(files)


def sample_same_class_images(class_dir, sample_size, seed_image=None, rng_seed=42):
    all_images = list_images_in_folder(class_dir)

    if seed_image is not None:
        seed_abs = os.path.abspath(seed_image)
        all_images_wo_seed = [
            p for p in all_images
            if os.path.abspath(p) != seed_abs
        ]

        if len(all_images_wo_seed) > 0:
            all_images = all_images_wo_seed

    if len(all_images) == 0:
        return []

    rng = random.Random(rng_seed)

    if sample_size >= len(all_images):
        sampled = all_images[:]
        rng.shuffle(sampled)
        return sampled

    return rng.sample(all_images, sample_size)


def sample_global_balanced_labeled_images(train_dir, per_class_samples, rng_seed=42, exclude_image=None):
    class_folders = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    rng = random.Random(rng_seed)
    exclude_abs = os.path.abspath(exclude_image) if exclude_image is not None else None

    all_samples = []

    for label_idx, cls in enumerate(class_folders):
        cls_dir = os.path.join(train_dir, cls)
        class_images = []

        for name in os.listdir(cls_dir):
            path = os.path.join(cls_dir, name)
            if os.path.isfile(path) and name.endswith(VALID_EXTS):
                if exclude_abs is not None and os.path.abspath(path) == exclude_abs:
                    continue
                class_images.append(path)

        if len(class_images) == 0:
            print(f"⚠️ 类别 {cls} 没有可用图像，跳过")
            continue

        if per_class_samples >= len(class_images):
            selected = class_images[:]
            rng.shuffle(selected)
        else:
            selected = rng.sample(class_images, per_class_samples)

        all_samples.extend([(p, label_idx) for p in selected])

    return all_samples


def compute_cosine_similarity(vec_a, vec_b, eps=1e-12):
    if vec_a is None or vec_b is None:
        return None, None, None

    norm_a = torch.norm(vec_a, p=2).item()
    norm_b = torch.norm(vec_b, p=2).item()

    if norm_a < eps or norm_b < eps:
        return None, norm_a, norm_b

    cos_sim = torch.dot(vec_a, vec_b).item() / (norm_a * norm_b + eps)
    return cos_sim, norm_a, norm_b


@torch.no_grad()
def compute_empirical_fpr(net, image_paths, candidate_label, device):
    hit = 0
    total = 0

    for path in image_paths:
        try:
            img_tensor = load_image_tensor(path, device)
            logits = net(img_tensor)
            pred = logits.argmax(dim=1).item()

            if pred == candidate_label:
                hit += 1

            total += 1

        except Exception as e:
            print(f"⚠️ Skipping image during FPR evaluation: {path} | {e}")

    fpr = hit / total if total > 0 else 1.0
    return fpr, hit, total


@torch.no_grad()
def inspect_candidate_scores(net, image_paths, candidate_label, device):
    probs = []

    for path in image_paths:
        try:
            img_tensor = load_image_tensor(path, device)
            logits = net(img_tensor)
            p = torch.softmax(logits, dim=1)[0, candidate_label].item()
            probs.append(p)

        except Exception as e:
            print(f"⚠️ Skipping image during probability inspection: {path} | {e}")

    if len(probs) == 0:
        return 0.0, 0.0

    avg_prob = sum(probs) / len(probs)
    max_prob = max(probs)

    return avg_prob, max_prob


def select_best_label(results):
    if len(results) == 0:
        return None

    zero_fpr = [r for r in results if abs(r["fpr"]) < 1e-12]

    if len(zero_fpr) > 0:
        valid = [r for r in zero_fpr if r["grad_cosine"] is not None]

        if len(valid) > 0:
            return max(valid, key=lambda x: x["grad_cosine"])

        return min(zero_fpr, key=lambda x: x["avg_prob"])

    valid_all = [r for r in results if r["grad_cosine"] is not None]

    if len(valid_all) > 0:
        return sorted(valid_all, key=lambda x: (x["fpr"], -x["grad_cosine"]))[0]

    return min(results, key=lambda x: x["fpr"])


if __name__ == "__main__":
    class_folders = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

    net = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    # 1. Obtain seed logits and candidate labels
    seed_logits, seed_probs, candidate_indices, seed_top1_idx = (
        get_seed_logits_and_candidates(
            SEED_IMAGE,
            net,
            DEVICE,
            topk=TOPK_CANDIDATES
        )
    )

    if seed_top1_idx >= len(class_folders):
        raise RuntimeError(
            f"seed_top1_idx={seed_top1_idx} exceeds class_folders length "
            f"{len(class_folders)}. Please ensure that the folder ordering in "
            f"TRAIN_DIR matches the model class indices."
        )

    if USE_MANUAL_CLASS:
        if SEED_CLASS not in class_folders:
            raise ValueError(f"❌ SEED_CLASS={SEED_CLASS} not found in TRAIN_DIR")

        seed_top1_folder = SEED_CLASS
        seed_top1_idx = class_folders.index(SEED_CLASS)

        print("⚠️ Using manually specified seed class, overriding model prediction")

    else:
        seed_top1_folder = class_folders[seed_top1_idx]
        print("ℹ️ Using model-predicted Top-1 class as the seed class")

    seed_class_dir = os.path.join(TRAIN_DIR, seed_top1_folder)

    print("-" * 60)
    print(f"📌 Seed class: {seed_top1_folder} (Index: {seed_top1_idx})")
    print(f"📌 Candidate labels (Top-{TOPK_CANDIDATES} logits): {candidate_indices}")
    print(f"📌 Seed class directory: {seed_class_dir}")
    print("-" * 60)

    # 2. Sample balanced images from the whole task distribution to estimate the task gradient
    per_class_samples = max(1, MAIN_CLASS_SAMPLE_SIZE // len(class_folders))

    main_class_paths = sample_global_balanced_labeled_images(
        train_dir=TRAIN_DIR,
        per_class_samples=per_class_samples,
        rng_seed=MAIN_CLASS_RANDOM_SEED,
        exclude_image=SEED_IMAGE
    )

    if len(main_class_paths) == 0:
        raise RuntimeError(
            f"❌ No valid images found in the training directory: {TRAIN_DIR}"
        )

    print(
        f"✅ Sampled {len(main_class_paths)} balanced images from the whole "
        f"training distribution for estimating the main-task average gradient "
        f"({per_class_samples} per class)"
    )

    # 3. Generate watermark images
    temp_out_dir = os.path.join(BASE_OUT_DIR, "temp_generated")

    generated_files = generate_and_save(
        SEED_IMAGE,
        temp_out_dir,
        NUM_SAMPLES,
        PERTURB_DIM,
        EPSILON_RANGE
    )

    print(f"✅ Watermark image generation completed. Temporary directory: {temp_out_dir}")
    print(f"✅ Generated {len(generated_files)} watermark images")

    # 4. Compute the average gradient of the main task on the global task distribution
    main_grad = compute_labeled_dataset_avg_gradient(
        net=net,
        labeled_samples=main_class_paths,
        device=DEVICE
    )

    if main_grad is None:
        raise RuntimeError(
            "❌ Failed to compute main_grad. Please check the images, "
            "model parameters, and training directory."
        )

    main_grad_norm = torch.norm(main_grad, p=2).item()
    print(f"📌 Main-task average gradient norm: {main_grad_norm:.6f}")

    print("\n🔍 Evaluating each candidate label ...\n")

    results = []

    for rank, cand_label in enumerate(candidate_indices, start=1):
        if cand_label >= len(class_folders):
            print(
                f"⚠️ cand_label={cand_label} exceeds the range of class_folders. "
                f"Skipping."
            )
            continue

        cand_folder = class_folders[cand_label]

        # A. Empirical FPR
        fpr, hit, total = compute_empirical_fpr(
            net,
            generated_files,
            cand_label,
            DEVICE
        )

        # B. Target-class probability statistics
        avg_prob, max_prob = inspect_candidate_scores(
            net,
            generated_files,
            cand_label,
            DEVICE
        )

        # C. Average watermark gradient under the candidate target label
        wm_grad = compute_dataset_avg_gradient(
            net,
            generated_files,
            cand_label,
            DEVICE
        )

        if wm_grad is None:
            grad_cosine = None
            wm_grad_norm = None
        else:
            grad_cosine, _, wm_grad_norm = compute_cosine_similarity(
                main_grad,
                wm_grad,
                eps=EPS
            )

        result = {
            "rank": rank,
            "label_idx": cand_label,
            "label_folder": cand_folder,
            "fpr": fpr,
            "hit": hit,
            "total": total,
            "avg_prob": avg_prob,
            "max_prob": max_prob,
            "grad_cosine": grad_cosine,
            "wm_grad_norm": wm_grad_norm,
        }

        results.append(result)

        print(f"[Top-{rank}] Label = {cand_folder} (Index {cand_label})")
        print(f"         FPR (Top-1 trigger rate) = {hit}/{total} = {fpr:.4f}")
        print(f"         Average target probability = {avg_prob:.6f}")
        print(f"         Maximum target probability = {max_prob:.6f}")
        print(f"         Gradient cosine (main, watermark) = {grad_cosine}")
        print(f"         Watermark gradient norm = {wm_grad_norm}")
        print("-" * 60)

    # 5. Select the best target label
    best = select_best_label(results)

    print("\n" + "=" * 60)
    print("📊 Candidate Label Evaluation Summary")
    print("=" * 60)

    for r in results:
        print(
            f"Label {r['label_folder']:>20s} | idx={r['label_idx']:>3d} | "
            f"FPR={r['fpr']:.4f} | AvgProb={r['avg_prob']:.6f} | "
            f"MaxProb={r['max_prob']:.6f} | CosSim={r['grad_cosine']} | "
            f"WMNorm={r['wm_grad_norm']}"
        )

    print("\n" + "=" * 60)
    print("🏆 Final Selection")
    print("=" * 60)

    if best is None:
        print("❌ No valid label was found.")

    else:
        print(f"Best label: {best['label_folder']} (Index {best['label_idx']})")
        print(f"Top-k rank: Top-{best['rank']}")
        print(f"FPR: {best['hit']}/{best['total']} = {best['fpr']:.4f}")
        print(f"Average target probability: {best['avg_prob']:.6f}")
        print(f"Maximum target probability: {best['max_prob']:.6f}")
        print(f"Gradient cosine similarity: {best['grad_cosine']}")
        print(f"Watermark gradient norm: {best['wm_grad_norm']}")

        # Optional: rename temp_generated to the final class directory
        final_out_dir = os.path.join(BASE_OUT_DIR, best["label_folder"])

        if os.path.abspath(temp_out_dir) != os.path.abspath(final_out_dir):
            os.makedirs(BASE_OUT_DIR, exist_ok=True)

            if os.path.exists(final_out_dir):
                print(f"\n⚠️ Target directory already exists: {final_out_dir}")
                print(
                    "   Please manually handle the files in temp_generated "
                    "if needed."
                )
            else:
                os.rename(temp_out_dir, final_out_dir)
                print(f"\n✅ Renamed generated directory to: {final_out_dir}")

    print("=" * 60)
