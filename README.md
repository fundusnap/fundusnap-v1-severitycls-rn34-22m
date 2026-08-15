<div align="center">
  <table border="1">
    <tr>
      <td align="center" style="padding: 20px;">
        <h3>📢 Domain & Email Migration Notice</h3>
        <p>From <b>May 30th, 2026</b>, Fundusnap will transition to new domains as <code>fundusnap.com</code> will not be renewed:</p>
        <p>🌐 <b>Website:</b> <a href="https://fundusnap.faizath.com">fundusnap.faizath.com</a> (formerly <i>fundusnap.com</i>)<br>
        ⚙️ <b>API:</b> <a href="https://fundusnap-api.faizath.com">fundusnap-api.faizath.com</a> (formerly <i>api.fundusnap.com</i>)<br>
        📧 <b>Email:</b> <a href="mailto:contact@fundusnap.faizath.com">contact@fundusnap.faizath.com</a> (formerly <i>contact@fundusnap.com</i>)<br>
        🛰️ <b>CDN:</b> <a>fundusnap-cdn.faizath.com</a> (formerly <i>cdn.fundusnap.com</i>)<br>
        📈 <b>Status Pages:</b> <a href="https://status.faizath.com/status/fundusnap">https://status.faizath.com/status/fundusnap</a> (formerly <i>status.fundusnap.com</i>)
        </p>
      </td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="assets/logo.png" alt="Fundusnap" width="420"/>
</p>

<p align="center">
  <b>AI-assisted diabetic retinopathy screening — from a phone camera to a graded fundus image.</b>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/fundusnap/fundusnap-v1-severitycls-rn34-22m">Hugging Face</a> &nbsp;•&nbsp;
  🐙 <a href="https://github.com/fundusnap/fundusnap-v1-severitycls-rn34-22m">GitHub</a>
</p>

<p align="center">
  <img alt="Task: image classification" src="https://img.shields.io/badge/task-image--classification-5B9BD5"/>
  <img alt="Backbone: ResNet34" src="https://img.shields.io/badge/backbone-ResNet34-5B9BD5"/>
  <img alt="Params: 22M" src="https://img.shields.io/badge/params-22M-5B9BD5"/>
  <img alt="Formats: ONNX and PyTorch" src="https://img.shields.io/badge/formats-ONNX%20%7C%20PyTorch-5B9BD5"/>
  <img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-5B9BD5"/>
</p>

# fundusnap-v1-severitycls-rn34-22m

Diabetic retinopathy **severity classifier** for colour fundus (retinal) photographs. Given a single
fundus image it predicts one of the five standard ordinal DR grades (0 = No DR through
4 = Proliferative DR) and returns a probability for each. The model is a ResNet34 backbone
(ImageNet-pretrained) with a fastai classification head — roughly **22M parameters**, hence the name:
`severitycls` (severity classification) + `rn34` (ResNet34) + `22m` (parameter count).

It is shipped both as a **fastai/PyTorch checkpoint** for further training and as an **ONNX** graph
with a dynamic batch axis for deployment.

## ⚠️ Intended use — not for clinical use

This model is released for **research and engineering use**, and at most as a *triage assist* inside
a workflow that a qualified clinician supervises.

- It is **not a medical device** and has **no regulatory clearance** (FDA, CE/MDR, or otherwise).
- It must **never** be the sole basis for a diagnosis, referral, or treatment decision.
- It has not been validated prospectively, on any specific camera or population, or against a
  reference grading standard beyond the public dataset described below.
- Its weakest area is exactly the clinically hardest one: separating early grades 0/1/2 (see
  [Limitations](#limitations-and-bias)).

Anyone deploying it in a screening context is responsible for their own validation and for keeping a
human grader in the loop.

## Label scheme

Outputs are the five-level International Clinical DR severity scale, as coded in the training
labels (`level` column):

| Index | Grade | Clinical meaning |
|:--|:--|:--|
| 0 | No DR | No visible retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than microaneurysms, less than severe NPDR |
| 3 | Severe | Severe non-proliferative DR |
| 4 | Proliferative | Proliferative DR (neovascularisation) |

The grades are ordinal, but the model is trained as a plain 5-way classifier — it does not exploit
that ordering, and it is not calibrated for ordinal-regression style thresholding.

## Files

| Path | What it is |
|:--|:--|
| `models/model.onnx` | ONNX graph (opset 14). Weights are stored externally. |
| `models/model.onnx.data` | External weight blob for `model.onnx`. **Must sit next to it** — loading the graph alone fails. |
| `models/retinopathy201519-rn34-1.pth` | fastai `learn.save()` checkpoint (a dict with `model` and `opt` keys) — use this to fine-tune. |
| `FastAI_Training.ipynb` | Training run: data prep, augmentation, fine-tuning, evaluation. |
| `Model_Converter.ipynb` | Checkpoint → ONNX export, and the ONNX → TensorFlow/TFLite path. |
| `schema_generated.py` | Generated TFLite FlatBuffers schema used by the `onnx2tf` toolchain. Build-time helper only — not part of the model. |

## Usage

### ONNX Runtime (recommended for inference)

```bash
pip install onnxruntime pillow numpy
```

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# ImageNet statistics — fastai's vision_learner normalises with these for pretrained models.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path, size=224):
    """Mirror fastai's Resize(224) default (ResizeMethod.Crop): resize the shorter
    side to 224, then take the centre crop."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    x = np.asarray(img, dtype=np.float32) / 255.0        # HWC, [0, 1]
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]                    # NCHW, shape (1, 3, 224, 224)


# model.onnx.data must be in the same directory as model.onnx.
session = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])

logits = session.run(["output"], {"input": preprocess("fundus.jpg")})[0]   # (1, 5)

# The graph emits raw logits — apply softmax yourself.
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = (e / e.sum(axis=1, keepdims=True))[0]

grade = int(probs.argmax())
print(f"grade {grade} ({LABELS[grade]})  p={probs[grade]:.4f}")
print({LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)})
```

The input axis 0 is dynamic, so you can batch: pass `(N, 3, 224, 224)` and get `(N, 5)` back.

### fastai / PyTorch checkpoint

The checkpoint stores a state dict only, so rebuild the identical learner before loading it:

```python
import torch
import torch.nn.functional as F
from fastai.vision.all import *
import pandas as pd

# A throwaway DataLoaders just to give the learner the right shape (5 classes).
dummy = pd.DataFrame({"image": ["sample"] * 5, "level": [0, 1, 2, 3, 4]})
dls = ImageDataLoaders.from_df(
    dummy, path=".", fn_col="image", label_col="level", suff=".jpg",
    valid_pct=0.0, bs=1, item_tfms=[], batch_tfms=[], shuffle=False,
)

learn = vision_learner(dls, resnet34, path=".", loss_func=FocalLoss(),
                       metrics=[accuracy], n_out=5)

ckpt = torch.load("models/retinopathy201519-rn34-1.pth",
                  weights_only=False, map_location="cpu")
learn.model.load_state_dict(ckpt["model"])
learn.dls.vocab = [0, 1, 2, 3, 4]

_, pred_idx, probs = learn.predict("fundus.jpg")
probs = F.softmax(probs, dim=0)          # learn.predict returns logits for this head
print(int(pred_idx), probs)
```

> On some fastcore versions the fastai import chain raises `AttributeError: 'L' object has no
> attribute 'starmap'`. Patch it before building the learner:
>
> ```python
> import itertools
> from fastcore.foundation import L
> if not hasattr(L, "starmap"):
>     L.starmap = lambda self, f: L(itertools.starmap(f, self))
> ```

## Training

**Data** — [`resized-2015-2019-diabetic-retinopathy-detection`](https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection),
the `resized_traintest15_train19` image set with `traintestLabels15_trainLabels19.csv` labels. This
combines the EyePACS 2015 and APTOS 2019 DR detection releases.

**Class balancing** — the raw label distribution is heavily skewed toward grade 0, so each grade was
resampled to 10,000 rows with replacement (`df.groupby('level').sample(10000, replace=True)`),
giving a 50,000-image balanced training frame. 10% was held out for validation.

**Augmentation** — [albumentations](https://albumentations.ai/) wrapped in a custom
`AlbTransform(Transform)`:

- `ShiftScaleRotate(rotate_limit=20, border_mode=0)`
- `HorizontalFlip`
- `RandomBrightnessContrast`
- `HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5)`

**Setup**

| | |
|:--|:--|
| Backbone | `resnet34`, ImageNet-pretrained |
| Head | fastai default (`AdaptiveConcatPool2d` → BN/dropout → linear), `n_out=5` |
| Loss | `FocalLoss()` |
| Input | `Resize(224)` (centre crop), ImageNet normalisation |
| Batch size | 32 |
| Schedule | `learn.fine_tune(4)` — 1 frozen epoch, then 4 unfrozen |
| LR | from `learn.lr_find()`, valley suggestion |
| Seed | 3865 |

**Run log** (unfrozen phase, ≈6:45 per epoch):

| Epoch | train_loss | valid_loss | accuracy |
|:--|:--|:--|:--|
| 0 | 0.5260 | 0.4779 | 0.5772 |
| 1 | 0.3620 | 0.3204 | 0.6894 |
| 2 | 0.2090 | 0.2327 | 0.7812 |
| 3 | 0.1030 | 0.2229 | 0.8154 |

## Evaluation

Measured on the held-out 10% validation split (n = 5,000) of the balanced training frame.

| Grade | Precision | Recall | F1 | Support |
|:--|:--|:--|:--|:--|
| 0 — No DR | 0.66 | 0.65 | 0.65 | 1,013 |
| 1 — Mild | 0.72 | 0.71 | 0.72 | 1,032 |
| 2 — Moderate | 0.75 | 0.76 | 0.76 | 954 |
| 3 — Severe | 0.97 | 0.98 | 0.97 | 1,003 |
| 4 — Proliferative | 0.97 | 0.98 | 0.98 | 998 |

| Aggregate | Precision | Recall | F1 |
|:--|:--|:--|:--|
| **Accuracy** | | | **0.82** |
| Macro avg | 0.81 | 0.82 | 0.82 |
| Weighted avg | 0.81 | 0.82 | 0.81 |

Macro-F1 to four decimals: **0.8153**. `FastAI_Training.ipynb` also plots the confusion matrix,
which shows the errors concentrated in the 0↔1↔2 block.

These figures are **self-reported** — computed by the author in `FastAI_Training.ipynb`, not
verified by Hugging Face — and are mirrored in the `model-index` metadata at the top of this card,
which is why the Hub labels them as such.

## Limitations and bias

- **Early grades are the weak point.** Grades 3 and 4 are separated almost perfectly (F1 0.97–0.98),
  but 0/1/2 sit at 0.65–0.76. That is the reverse of what a screening deployment wants most, since
  distinguishing "no DR" from "mild/moderate" is what drives the referral decision.
- **Validation is optimistic.** The validation split was drawn *after* oversampling with
  replacement, so duplicated images can appear on both sides of the split, and the class balance
  (≈1,000 per grade) does not resemble real screening prevalence, where grade 0 dominates. Expect
  meaningfully lower real-world performance than the headline 0.82 — treat these numbers as a
  relative sanity check, not a deployment estimate.
- **Resolution.** 224×224 with a centre crop discards the fine detail that microaneurysms and small
  haemorrhages live in, and can crop away peripheral lesions entirely.
- **Domain shift.** Training images come from the public 2015/2019 releases, dominated by specific
  camera models, capture protocols, and patient populations. Different hardware, field of view,
  illumination, or demographics will degrade accuracy, and the model has not been audited for
  performance differences across subgroups.
- **Image quality.** There is no reject/ungradable option — a blurred, over-exposed, or non-fundus
  image still yields a confident-looking 5-way distribution.
- **Uncalibrated.** Softmax outputs are not calibrated probabilities; do not read them as risk.

## Reproducing and converting

`FastAI_Training.ipynb` reproduces the training run end to end. `Model_Converter.ipynb` reloads the
checkpoint and exports it.

The one non-obvious step in the export: PyTorch's ONNX exporter cannot handle
`AdaptiveMaxPool2d`, which fastai's `AdaptiveConcatPool2d` head layer contains. It is swapped for an
ONNX-friendly equivalent before `torch.onnx.export`:

```python
class ONNXFriendlyConcatPool(nn.Module):
    def forward(self, x):
        max_pool = x.amax(dim=[-1, -2], keepdim=True)
        avg_pool = x.mean(dim=[-1, -2], keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)

learn.model[1][0] = ONNXFriendlyConcatPool()

torch.onnx.export(
    learn.model, torch.randn(1, 3, 224, 224), "model.onnx",
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

For a TensorFlow / TFLite build (for on-device inference), continue with:

```bash
onnx2tf --input_onnx_file_path model.onnx --output_folder_path model_tf
```

## License

Released under [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/) — attribution
required, **non-commercial use only**. The underlying training data carries its own Kaggle /
EyePACS / APTOS terms; check those before redistributing anything derived from it.

## Citation

```bibtex
@software{fundusnap_severitycls_rn34_22m,
  title  = {fundusnap-v1-severitycls-rn34-22m: ResNet34 diabetic retinopathy severity classifier},
  author = {Fundusnap},
  url    = {https://huggingface.co/fundusnap/fundusnap-v1-severitycls-rn34-22m},
  license = {CC-BY-NC-4.0}
}
```
