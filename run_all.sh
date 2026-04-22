#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Execute every case study training script
# =============================================================================
#
# Usage:
#   bash run_all.sh              # run all scripts (requires internet access)
#   bash run_all.sh --offline    # skip scripts that require network downloads
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Notes:
#   - All scripts are run from the repo root unless otherwise noted.
#   - The LIVE Naive Bayes script is an interactive REPL; run it manually.
#   - Jupyter notebooks (NBA, NFL passing yards) are not included here;
#     open them in Jupyter or VS Code.
#   - The Loan Default FastAPI server is not included; start it with:
#       uvicorn api:app --reload  (from the api/ directory, after training)
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- Flags -------------------------------------------------------------------
OFFLINE=false
for arg in "$@"; do
  case $arg in --offline) OFFLINE=true ;; esac
done

# --- Colors ------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- Counters ----------------------------------------------------------------
PASS=0
FAIL=0
SKIP=0
FAILED_LABELS=()

# --- Helpers -----------------------------------------------------------------
run() {
  local label="$1"; shift
  echo -e "${YELLOW}▶  ${label}${NC}"
  python "$@" 2>&1 | sed 's/^/   /'
  local code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then
    echo -e "${GREEN}✓  PASS${NC}\n"
    (( PASS++ )) || true
  else
    echo -e "${RED}✗  FAIL (exit ${code})${NC}\n"
    (( FAIL++ )) || true
    FAILED_LABELS+=("$label")
  fi
}

# run a script from a specific directory (for scripts with cwd-relative paths)
run_in() {
  local label="$1"
  local dir="$2"; shift 2
  echo -e "${YELLOW}▶  ${label}${NC}"
  ( cd "$dir" && python "$@" ) 2>&1 | sed 's/^/   /'
  local code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then
    echo -e "${GREEN}✓  PASS${NC}\n"
    (( PASS++ )) || true
  else
    echo -e "${RED}✗  FAIL (exit ${code})${NC}\n"
    (( FAIL++ )) || true
    FAILED_LABELS+=("$label")
  fi
}

skip() {
  local label="$1"
  local reason="$2"
  echo -e "${CYAN}⊘  SKIP  ${label}${NC}  — ${reason}\n"
  (( SKIP++ )) || true
}

header() {
  echo -e "${BOLD}━━━  $1  ━━━${NC}"
}

# =============================================================================
echo -e "${BOLD}machine-learning-lab — run_all.sh${NC}"
echo "Repo:  $REPO_ROOT"
echo "Mode:  $( $OFFLINE && echo 'offline  (network scripts will be skipped)' || echo 'online' )"
echo ""

# =============================================================================
header "Linear Regression"

run "Basketball points prediction" \
  case-studies/linear-regression-models/basketball-points-prediction/scripts/points_prediction_linear_reg.py

run "Football points prediction — NFL" \
  case-studies/linear-regression-models/football-points-prediction/scripts/points_prediction_linear_reg.py \
  --dataset nfl

run "Football points prediction — CFB" \
  case-studies/linear-regression-models/football-points-prediction/scripts/points_prediction_linear_reg.py \
  --dataset cfb

# =============================================================================
header "Naive Bayes"

run "Text sender identification (batch)" \
  case-studies/naives-bayes-models/text-sender-identification/scripts/text_sender_identification_nb.py

skip "Text sender identification (LIVE)" "interactive REPL — run manually"

# =============================================================================
header "Random Forest"

run "Loan default prediction + model serialization" \
  case-studies/random-forest-models/loan-default-prediction/scripts/loan_default_random_forest.py

run "NFL game prediction" \
  case-studies/random-forest-models/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py

# =============================================================================
header "XGBoost"

run "Breast cancer detection" \
  case-studies/xgb-models/breast-cancer/scripts/train_eval.py

run "Loan approval" \
  case-studies/xgb-models/loan-approval/scripts/loan_approval_model.py

# =============================================================================
header "Monte Carlo"

if $OFFLINE; then
  skip "Crypto market prediction"    "requires yfinance (internet)"
  skip "Stock market prediction"     "requires yfinance (internet)"
  skip "Super Bowl 2026 prediction"  "requires nflreadpy (internet)"
else
  run "Crypto market prediction" \
    case-studies/monte-carlo-models/crypto-market-prediction/scripts/crypto_market_prediction_mc.py

  run "Stock market prediction" \
    case-studies/monte-carlo-models/stock-market-prediction/scripts/stock_market_monte_carlo.py

  run "Super Bowl 2026 prediction" \
    case-studies/monte-carlo-models/superbowl-2026-prediction/scripts/superbowl_2026_mc.py
fi

# =============================================================================
header "Convolutional Neural Network"

run "Handwriting recognition (train + eval, no Gradio demo)" \
  case-studies/convolutional-models/handwriting-recognition/scripts/handwriting_recognition_cnn.py \
  --skip-demo

# =============================================================================
header "Markov Chain"

# Weather script uses cwd-relative Path("results") so must run from its own dir
run_in "Weather prediction (Markov Chain)" \
  notebooks/jupyter/weather-prediction \
  weather_markov.py

skip "NBA points prediction (Jupyter)"          "open nba-points-prediction_ENTERPRISE.ipynb in Jupyter"
skip "NFL passing yards prediction (Jupyter)"   "open nfl-passing-yards-prediction_linear_reg_ENTERPRISE.ipynb in Jupyter"

# =============================================================================
echo ""
header "Results"
echo -e "  ${GREEN}PASS  ${PASS}${NC}"
echo -e "  ${RED}FAIL  ${FAIL}${NC}"
echo -e "  ${CYAN}SKIP  ${SKIP}${NC}"

if [ "${#FAILED_LABELS[@]}" -gt 0 ]; then
  echo -e "\n  Failed scripts:"
  for label in "${FAILED_LABELS[@]}"; do
    echo -e "    ${RED}✗  ${label}${NC}"
  done
  exit 1
fi

echo -e "\n${GREEN}All scripts completed successfully.${NC}"
