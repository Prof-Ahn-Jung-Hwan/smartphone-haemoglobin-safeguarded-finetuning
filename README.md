# Cross-population smartphone haemoglobin estimation with safeguarded local fine-tuning and automated sclera–conjunctiva co-analysis

Code for the study *Cross-population smartphone haemoglobin estimation with safeguarded local fine-tuning and automated sclera–conjunctiva co-analysis: a development and external validation study* (submitted to *eClinicalMedicine*, 2026).

A mobile model (MobileNetV3-Large, 2·97 million deployed parameters) estimates haemoglobin from the sclera and palpebral conjunctiva (SCCA). The SCCA region is segmented automatically from the whole photograph with SAM 3 and everyday-language prompts. When the model is moved to a new location it is fine-tuned with 30 local participants (learning rate 1 × 10⁻⁴). Three safeguards keep that local fine-tuning from failing:

| | Safeguard | Where |
|---|---|---|
| S1 | LayerNorm on the final features | `configs/safeguard_S1*.yaml` (`use_norm_only`, or the normalisation before the adaptation layer) |
| S2 | adaptation (linear projection) layer | `use_encoder: true` |
| S3 | backbone batch-normalisation statistics kept at their development values during local fine-tuning | `finetune(..., freeze_extra="bn")` in `main/external/260918_external_grid.py` |
| S4 · S5 | auxiliary losses (anchor · batch mean) — explored during development, **not reported in the paper** | `configs/safeguard_S1_S2_S4_S5.yaml` (`force_lambda`) |

The recommended model is MobileNetV3-Large with S1+S2+S3. The safeguards were also tested in EfficientNet-B0, MobileNetV2, MobileViT-XXS and EdgeNeXt-XS (36 backbone–configuration combinations; 138 000 local fine-tuning runs in all experiments).

> **This repository runs a smoke test only.** The development cohort (762 participants, 1,524 photographs, Republic of Korea) cannot be shared. The demo trains a small model on 40 public images and fine-tunes it with 10 images from a second location. **Its numbers are not study results** and can go in any direction.

---

## Quick start (about one minute on one NVIDIA GPU)

```bash
conda create -n scca python=3.10 -y && conda activate scca
pip install -r requirements.txt          # install PyTorch for your CUDA version first
bash demo/run_demo.sh
```

The three steps:

1. `demo/make_demo_arm.py` builds the roster `demo/work/arm_demo_train.xlsx` (columns `file_path · number · separator · Hb`) from `demo/data/roster.csv`.
2. `code/src/train_bins.py -c configs/demo_S1_S2.yaml` trains the S1 + S2 model. It uses the study's training code with a 40-epoch schedule, on fold 0 of 5, grouped by participant, with an inner validation split.
   - Output: `demo/work/train/demo_S1_S2-fold0/` (`best.ckpt`, `test_pred.csv`, `config.yaml`).
3. `demo/calibrate_and_evaluate.py` calibrates locally with the study's own `finetune()` and evaluates 20 participants from the second location in three settings: before calibration, calibrated with BN statistics updated, and calibrated with SS.
   - Output: `demo/work/demo_results.csv`.

A GPU is required: the training code uses `torch.autocast` on CUDA.

## Repository layout

| Path | What | Writes to |
|---|---|---|
| `code/` | model (`model/`), data loader (`dataset/`), training (`src/train_bins.py`), ROI composition (`common/roi_lib.py`), SAM 3 helpers (`sam3/`) | — |
| `configs/` | exact training configurations of the five safeguard settings (paths removed) + the demo configuration | — |
| `main/external/260918_external_grid.py` | external calibration engine: draws (`split`), `finetune()`, `predict()` | `main/results/external/…` |
| `main/analysis/260922_stage5_ext.py` | external evaluation driver used in the study (`zero` = before calibration, `few` = calibrated; `--freeze-extra bn --tag bn` = S3) | `<experiment>/external/{zero,few,few_bn}/…` |
| `demo/` | smoke test on public images | `demo/work/` |
| `segmentation/segment_photo.py` | automatic segmentation, photograph → ROI, in one script (see below); the staged study scripts are kept beside it for reference | `--out` folder |
| `results_script/` | scripts that produced the manuscript tables from the full set of predictions (not runnable without the development cohort) | `main/results/main_results/*.csv`, tables |
| `tools/check_no_private.py` | pre-publication privacy check | — |

Code comments are mostly in Korean.

## Data

- **Development cohort** (Ajou University Hospital, Republic of Korea; IRB AJOUIRB-MDB-2021-209; 762 participants, 1524 photographs). The datasets generated and analysed during the study are not publicly available because of patient privacy regulations, but are available from the corresponding author on reasonable request for collaborative research, subject to institutional review board approval. No development-cohort photograph, roster or identifier is in this repository.
- **External locations** (India 95, Italy 122): the public Eyes-Defy-Anemia dataset — Dimauro G, et al., IEEE DataPort, doi:10.21227/T5S2-4J73, CC BY 4.0.
  - `demo/data/` holds 70 of these photographs: India 40, Italy 30. They are stored as automatically segmented SCCA regions, downscaled to at most 640 px, with metadata removed.
  - `demo/data/roster.csv` lists the original keys, the laboratory haemoglobin and the role of each image in the demo.
  - The dataset is licensed under CC BY 4.0, which permits redistribution of adapted material with attribution; the images here were cropped to the eye, masked and downscaled.

## Automatic segmentation

`segmentation/segment_photo.py` runs the whole segmentation on one photograph or a folder and writes the model input:

```bash
python segmentation/segment_photo.py --images <photo or folder> --out seg_out --device 0
# seg_out/roi_scca/*.jpg (model input) · roi_bc/*.jpg · masks/*.npz · segmentation_log.csv
```

It uses SAM 3 (`facebook/sam3` through `transformers`; access must be requested on Hugging Face) and makes nine calls per photograph. It needs no human input.

| Stage | Prompts | Role |
|---|---|---|
| 1 First pass | "lower eyelid inner surface" | floor below which sclera is discarded |
| | "white part of the eye" | sclera; if its score is below 0·75, "the bright area in the eye" is used instead |
| | "pupil" | seed of the iris |
| | "eyelashes" | removed from the sclera |
| 2 Conjunctiva | "the reddish area inside the lower eyelid" ∪ "the pink area inside the lower eyelid" | conjunctiva |
| 3 Second-pass region | "the eye surface next to the iris" | sclera when the first pass failed |
| | "skin" | excluded |
| 4 Final | — | see below |
| 5 ROI | — | outside the mask black, cropped to the mask's box |

In stage 4 the first pass is judged failed without human review when (first-pass sclera − conjunctiva) / conjunctiva < 0·30; the second-pass region is then used. The iris and the upper eyelashes are removed at the end.

The stage functions are copied verbatim from the staged scripts that produced the study masks. Those scripts are kept in `segmentation/` for reference; machine paths are replaced by `PATH/TO/…`. On all 202 external photographs, `segment_photo.py` reproduced the study masks and ROI images exactly (identical pixels; 35 photographs used the second pass).

## Reproducing the study

With access to the development cohort:
- folds are assigned per participant, stratified by integer haemoglobin, with one fixed outer seed (`사람분할`, `OUTER_SEED = 2026` in `code/dataset/ajoumc_anemia.py`; the fold roster is written by `experiments/00_분할명부_만들기.py`), so both eyes of a participant are always in the same fold;
- train 10 folds × 10 seeds per configuration with `code/src/train_bins.py -c configs/<setting>.yaml --fold <f> --cv-fold 10 --val-inner 9`;
- run `main/analysis/260922_stage5_ext.py zero|few --few-lr 1e-4` for the external locations (`--freeze-extra bn --tag bn` adds S3; `EXT_TRAIN_ROOT`, `EXT_CELL_PREFIX` select the experiment);
- build the tables and figures with `results_script/261020_전체실행.sh`, then `results_script/supp_tableST13_experiments_scale.py` (appendix table S2, counted from the prediction files) and `results_script/main_figure4_backbone_selection_10seed.py` (figure 4). Script names keep the working numbers (ST1–ST13, SF2); `_공통.py` maps them to the appendix numbers of the paper (tables S1–S10, figure S2).

Participant-level bootstrap with 100,000 resamples is implemented in `results_script/_공통.py` (`boot`, `대응`).

## Collaboration on internal data

The full study used a larger internal clinical dataset from Ajou University Medical Center. Because of patient privacy and institutional regulations (IRB), these data cannot be made public.

We welcome academic and research collaboration. If you are interested in working with the internal dataset, please follow this protocol:

- Contact: erdrajh@naver.com (corresponding author) or erdrajh@ajou.ac.kr (institutional)
- Requirements:
  - IRB approval from your institution and from Ajou University Medical Center
  - a formal Data Use Agreement (DUA)
  - a mutually agreed research collaboration protocol

## Licence

**Source code.** This repository's source code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- Permitted: academic research, internal benchmarking, and other non-commercial evaluation or comparison studies.
- Prohibited: integrating the code into commercial products or services, or any use primarily intended for commercial advantage.

**Documentation.** Unless otherwise indicated, the documentation in this repository follows the same PolyForm Noncommercial License 1.0.0.

**Data.** The sample images and roster under `demo/data/` remain subject to the original Eyes-Defy-Anemia dataset terms (CC BY 4.0; Dimauro G, et al., IEEE DataPort, doi:10.21227/T5S2-4J73).

## Citation

*[Citation to be added on publication]*
