#!/usr/bin/env bash
# 원고의 표 · 그림을 차례로 만들고 검사한다 (11-04 : 1 seed 10겹 앙상블 · 중간값 (범위) 기준)
# ★ 11-07 : 기본은 미세조정 학습률 1e-4 → 표는 paperwork/results_table_figure_lr1e-4/ (옛 2e-5 표는 그대로 둔다) · 옛 판을 다시 만들 때만 env 표_보정lr=2e-5 bash … (한글 변수는 env 로 넘긴다)
# 쓰는 법  bash paperwork/results_script/261020_전체실행.sh                    표 · 그림 + 검사
#          bash paperwork/results_script/261020_전체실행.sh --with-analysis    P3 · lr_선택_환자단위 부터
#          bash paperwork/results_script/261020_전체실행.sh --list             목록
# 🔵 run_all.sh 가 끝에서 이 경로를 부른다 → 이름을 바꾸지 않는다. 실패해도 0 으로 끝난다 (run_all 이 실패로 끝나면 뒤의 E04b · E15 가 시작되지 않는다 · 표는 다시 만들 수 있다). 실패는 🔴 로 찍는다.
# 🔵 ST7 은 E04b · E15 가 끝나야 만들어진다 (그 전에는 "아직" 이라 찍고 넘어간다) → E15 뒤에 이 스크립트를 한 번 더 돌린다.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$(cd "$HERE/../.." && pwd)"; MR="$P/main/results/main_results"; A="$HERE/analysis"
# source <conda>/etc/profile.d/conda.sh && conda activate <env>   # GitHub 판 : 연구 서버의 conda 경로를 지움
export OMP_NUM_THREADS=8

scripts=(
  main_table1_cohorts.py
  main_table2_segmentation.py
  main_table3_ablation_mobilenetv3_10seed.py     # 11-09 번호 바꿈 (옛 Table 4)
  main_table4_input_bilateral_roi.py             # 11-09 번호 바꿈 (옛 Table 3)
  main_table5_thresholds_10seed.py
  supp_tableST1_previous_work.py
  supp_tableST2_backbone_screen.py
  supp_tableST3_ablation_other_backbones_10seed.py
  supp_tableST4_factorial_backbones_10seed.py
  supp_tableST5_recommended_model_10seed.py
  supp_tableST6_alternatives_to_SS_10seed.py
  supp_tableST7_inputs_without_safeguards_10seed.py
  supp_tableST8_seed_values_10seed.py
  supp_tableST10_ranking_10seed.py
  supp_figSF2_calibration_size.py
  main_figure3_calibration_safeguards_10seed.py     # Table 3 · ST3 뒤
)
need=(
  "261020_추론비용.py|P3_추론비용.csv|Table 3 · ST3 · ST10"
  "261005_lr_환자단위검증.py|../grid/lr_선택_환자단위.csv|ST2"
)
if [ "${1:-}" = "--list" ]; then printf '결과 산출  %s\n' "${need[@]%%|*}"; printf '표 · 그림  %s\n' "${scripts[@]}"; exit 0; fi
mkdir -p "$MR"
if [ "${1:-}" = "--with-analysis" ]; then
  for x in "${need[@]}"; do
    s="${x%%|*}"; printf '\n════════ 결과 산출 %s  %s\n' "$s" "$(date +%H:%M:%S)"
    ( cd "$P" && python -u "$A/$s" ) || { echo "🔴 실패 : $s — 표 단계를 멈춘다 (실험은 계속 · 고친 뒤 다시 돌린다)"; exit 0; }
  done
fi
miss=0
[ -s "$MR/L1_임계_내부.csv" ] || { echo "🔴 없음 : L1_임계_내부.csv ← E13 (main/analysis/260929_임계선정.py)"; miss=1; }
for x in "${need[@]}"; do
  s="${x%%|*}"; r="${x#*|}"; f="${r%%|*}"
  [ -s "$MR/$f" ] || { echo "🔴 없음 : $f  ← python paperwork/results_script/analysis/$s"; miss=1; }
done
[ "$miss" = 0 ] || { echo "🔴 재료가 없어 표 단계를 멈춘다 (실험은 계속)"; exit 0; }
failed=()
for s in "${scripts[@]}"; do
  printf '\n──────── %s  %s\n' "$s" "$(date +%H:%M:%S)"
  ( cd "$P" && python -u "$HERE/$s" ) || failed+=("$s")
done
printf '\n════════ 표기 검사\n'
( cd "$P" && python "$A/261018_표검사.py" )
printf '\n════════ 마무리 : %d개 중 실패 %d\n' "${#scripts[@]}" "${#failed[@]}"
if [ ${#failed[@]} -gt 0 ]; then printf '  🔴 실패 : %s (실험은 계속 · 고친 뒤 다시 돌린다)\n' "${failed[*]}"; exit 0; fi
printf '  ✅ 전부 성공 — 검사에 🔴 가 없는지도 위에서 확인\n'
