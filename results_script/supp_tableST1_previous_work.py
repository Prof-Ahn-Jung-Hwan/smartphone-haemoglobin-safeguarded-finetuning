# -*- coding: utf-8 -*-
"""**Table ST1. 선행 결막 · 공막 사진 연구 요약 (23편)**  2026-10-23

무엇  결막 · 공막 사진으로 Hb 를 추정하거나 빈혈을 분류한 원저 23편 — 서론의 "선행 연구는 부록에 정리하였다" 를 받친다.
      npj 원고 부록 Supplementary Table 1 의 18편 (`previous_paper_npj/TechCheck_supple_file_npj.pdf`) 을 옮기되 원문 PDF (`근거문헌/npj_ST1_related_work_reference/`) 와 대조해 고쳤고 (10-23 · 원문에 없는 값은 `··`),
      Das 2024 (NiADA) · Kato 2024 · Nakahara 2026 (연속) · Ramos-Soto 2025 · Esfahani 2026 (분류) 5편을 더했다
      (`paper_intro_method_result_dis/투고규정_및_문헌고찰내용/261010_신규문헌_경쟁력검토.md` §3 · 근거문헌 PDF).
기준  결막 · 공막 사진으로 Hb 나 빈혈을 예측한 원저 · 모든 참여자에게 하나의 모델 · 성능 지표 보고.
      원문 값만 옮긴다 (r 을 R² 로 바꾸지 않는다) · 빈 칸은 `··` · 각주는 설계만 적고 평가하는 말은 쓰지 않는다.
      Lancet 형식 : 표는 한 부분 (분류 · 연속은 열로) · 가운데 점 · 참고문헌 번호는 부록 따로 (`mix_final_claude/261022_ST1_reference_mix.md`).
1차 자료  이 파일 안의 표 (문헌에서 옮긴 값 · 계산 없음)
내는 것   results_table_figure/supplementary/TableST1_previous_work.xlsx
쓰는 법   python supp_tableST1_previous_work.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

열 = ["Study", "Year", "Outcome", "Haemoglobin cut-off, g/dL", "Model", "MAE, g/dL", "R²", "Accuracy, %",
     "Specificity, %", "Sensitivity, %", "AUROC", "Participants, n", "Images, n", "Attachment or device"]
N = "··"
행 = [
    # ── npj 부록 Supplementary Table 1 (18편 · 같은 순서) ──
    ["Collings et al¹ᵃ", 2016, "Classification", "<11", "Erythema index threshold", N, N, N, "LX5 83; iPhone 5S 71", "LX5 57; iPhone 5S 74", N, 94, 94, "Colour checker"],
    ["Saldivar-Espinoza et al²ᵇ", 2019, "Classification", "9–11", "Neural network (regressor)", N, N, "42·96–93·53", "36·03–94·20", "22·49–77·58", N, N, 300, "Colour checker"],
    ["Suner et al³ᶜ", 2021, "Classification", "Women <12·5, men <13·5", "Stepwise regression", N, N, "72·6", "72·5", "72·8", "0·80", 344, 1722, "None"],
    ["Bevilacqua et al⁴", 2016, "Classification", "Transfusion needed", "SVM + RGB/CIELab", N, N, "84·4", "82·4", "100", N, 77, 77, "Head-mounted device with LED"],
    ["Dimauro et al⁵ᵈ", 2018, "Classification", "10·5 and 11·5 (three risk classes)", "k-NN (a*, b*, G features)", N, N, "100", "100", "100", N, 113, 113, "Macro lens + LED"],
    ["Dimauro et al⁶ᵉ", 2019, "Classification", "11·5", "k-NN", N, N, "76·5–98·2", "83·1–96·4", "56·0–100", N, 102, "102–708", "Macro lens + LED"],
    ["Dimauro et al⁷ᶠ", 2023, "Classification", "10·5", "RUSBoost", N, N, "83–91", "87–98", "19–69", N, 218, 218, "Macro lens + LED"],
    ["Chen et al⁸ᵍ", 2017, "Classification", "11", "Kalman filter + nonlinear penalty regression", N, N, N, "80·85", "76·19", N, N, 100, "None"],
    ["Ghosal et al⁹ʰ", 2021, "Classification", "11·5", "Colour spectroscopy regression", N, N, N, N, "89", N, 65, N, "None"],
    ["Appiahene et al¹⁰", 2023, "Classification", "11", "CNN + logistic regression + Gaussian blur", N, N, "92·5", "95", "90", N, N, 710, "None"],
    ["Zhao et al¹¹ⁱ", 2024, "Classification", "Sex-specific and transfusion cut-offs", "Polynomial regression (high hue ratio)", N, N, "75·4–92·7", "89·7–96·5", "54·3–92·7", "0·76–0·92", 426, 1704, "None"],
    ["Sehar et al¹²ʲ", 2025, "Classification", N, "VGG16 + ResNet50 + InceptionV3 stacking ensemble", N, N, "89·48", N, "95", "0·97", N, 764, "None"],
    ["Çuvadar and Yılmaz¹³ᵏ", 2023, "Continuous", N, "CNN + MLP", "0·51", "0·86", N, N, N, N, 388, 388, "None"],
    ["Xu et al¹⁴", 2024, "Continuous", N, "Actor-critic reinforcement learning (MobileNetV2)", "1·19", N, N, N, N, N, 284, 1244, "None"],
    ["Chen et al¹⁵ˡ", 2022, "Continuous", N, "Semantic segmentation + MobileNetV3 + SE", "1·521", "0·512", N, N, N, N, 1065, 1065, "None"],
    ["Chen et al¹⁶", 2024, "Continuous", N, "EGE-UNet + DHA (C3AE)", "1·34", "0·34", N, N, N, N, 284, 1265, "None"],
    ["Kasiviswanathan et al¹⁷ᵐ", 2020, "Continuous", "11", "Ridge regression", "1·34", N, N, N, N, N, 135, 1212, "None"],
    ["Hu et al¹⁸ⁿ", 2023, "Continuous", "10", "Mask R-CNN + MobileNetV3", "1·6", "0·503", "85·4", "97·2", N, "0·752", 284, 1065, "None"],
    # ── 10-23 추가 (5편) ──
    ["Das et al (NiADA)¹⁹ᵒ", 2024, "Continuous", "Cohort-specific", "CNN + vision transformer regression", N, N, N, "53·8", "75·8", N, 556, N, "None"],
    ["Kato et al²⁰ᵖ", 2024, "Continuous", "11", "U-Net segmentation + CNN regression", N, N, N, "99", "20", N, 150, 300, "Smartphone flash"],
    ["Nakahara et al²¹ᑫ", 2026, "Continuous", "Sex-specific (men <13, women <12)", "ConvNeXt regression", "1·375", N, "76", "79", "71", "0·75", 225, 9903, "Smartphone-mounted slit-lamp attachment"],
    ["Ramos-Soto et al²²ʳ", 2025, "Classification", "Site-specific", "Vision transformer", N, N, "98·47", "100", "96", N, 217, 217, "Macro lens + LED"],
    ["Esfahani et al²³ˢ", 2026, "Classification", "Sex-specific (men <13, women <12)", "AutoML (whole-eye training, conjunctiva test)", N, N, "83·5", N, "100", "0·899", 218, 218, "Macro lens + LED"],
]
T = pd.DataFrame(행, columns=열)
# 각주 표시 — 연구 이름 끝의 위첨자 글자를 "Note" 열의 보통 글자로 옮긴다 (q 는 위첨자 글자가 없어서)
위 = dict(zip("ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᑫʳˢ", "abcdefghijklmnopqrs"))
T.insert(1, "Note", [위.get(x[-1], "··") for x in T.Study])
T["Study"] = [x[:-1] if x[-1] in 위 else x for x in T.Study]
시작("Table ST1 — 선행 연구 요약 (23편)")
print(f"  {len(T)}편 · 분류 {sum(T.Outcome=='Classification')} · 연속 {sum(T.Outcome=='Continuous')}")
엑셀("TableST1_previous_work", 부록, {"ST1": T},
    제목="Table ST1. Previous studies estimating haemoglobin or classifying anaemia from photographs of the conjunctiva or sclera",
    주석=[
      "Studies that predicted haemoglobin or anaemia from conjunctival or scleral photographs with a single model applied to all participants",
      "  and reported a performance measure. Values are as reported in the original articles; where several models or settings were",
      "  reported, the range is shown. ·· = not reported. References for this table are numbered separately (appendix references).",
      "CNN, convolutional neural network; MAE, mean absolute error; MLP, multilayer perceptron; SE, squeeze-and-excitation; SVM, support",
      "  vector machine; k-NN, k-nearest neighbours; AUROC, area under the receiver operating characteristic curve.",
      "a Validation set (47 participants); accuracy not reported.",
      "b Accuracy, specificity and sensitivity at haemoglobin cut-offs of 9, 10 and 11 g/dL; preprint.",
      "c Study-defined anaemia cut-offs; accuracy, specificity and sensitivity in the validation phase (202 participants); AUROC 0·80.",
      "d Three risk classes (<10·5, 10·5–11·5 and >11·5 g/dL) with 10-fold cross-validation; no misclassification reported.",
      "e Range from the unbalanced dataset to datasets balanced by SMOTE or ROSE before cross-validation (up to 708 instances).",
      "f RUSBoost and random forest; 218 images (Eyes-Defy-Anemia dataset, Indian and Italian participants).",
      "g Three-level scheme; sensitivity and specificity among samples not classed as doubtful; accuracy not reported.",
      "h Haemoglobin estimated within ±0·32 g/dL of laboratory values; sensitivity at <11·5 g/dL; accuracy and specificity not reported.",
      "i Anaemia (sex-specific) and transfusion cut-offs (<9 and <7 g/dL); four images per participant.",
      "j Stacking ensemble; recall reported as sensitivity; 764 images augmented to 4315 with a generative adversarial network.",
      "k Blood-donor cohort; age, height, weight and sex used as model inputs.",
      "l With oversampling (SMOTE); on original whole-eye images, MAE 1·952 and R² 0·306.",
      "m Training 1115 and test 97 images; correlation r 0·705 in the test set reported instead of R².",
      "n Accuracy, specificity and AUROC at <10·0 g/dL; sensitivity not reported in the main text; training and test images up-sampled",
      "  (1065 original images to 1226 training and 288 test images).",
      "o Sex, age and pregnancy status used as inputs; sensitivity and specificity for adult women; MAE not reported.",
      "p Children; average of the two eyes' predictions; correlation r 0·44 reported instead of R².",
      "q Video recorded with a smartphone-mounted slit-lamp attachment; frames after automated extraction; fixed test set of 44",
      "  participants; MAE at video level; sensitivity, specificity and AUROC at frame level; correlation r 0·42.",
      "r Images split at the image level after augmentation; site-specific anaemia cut-offs.",
      "s Test set of 22 images; no cross-validation.",
    ])
