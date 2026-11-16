#!/bin/zsh
# Run queue in priority order, so the study is usable wherever it stops. Resumable: finished rows are skipped.
# usage: ./queue.sh [N]   env: NICE (default 10), PAR (default 4)
cd "$(dirname "$0")"
PY=.venv/bin/python; L=../lexsi/models; M=models; N=${1:-300}
run() {  # gguf name passes
  if [[ -f $1 ]]; then
    $PY run_eval.py $1 $2 --n $N --passes $3 --nice ${NICE:-10} --parallel ${PAR:-4} 2>&1 | tee -a logs/queue.log
    echo "[queue $(date +%H:%M)] done $2 $3" | tee -a logs/queue.log
  else echo "[queue $(date +%H:%M)] skip $2 (no file $1)" | tee -a logs/queue.log; fi
}
echo "[queue $(date +%H:%M)] start N=$N" | tee -a logs/queue.log
C=orig,ptrue,verbal,rev
Q8=$M/tiny-aya-global-q8_0.gguf   # near-lossless reference, made from the official f16 file
[[ -f $Q8 ]] || ../lexsi/bin/llama-b12403/llama-quantize $L/tiny-aya-global-f16.gguf $Q8 q8_0 > logs/quantize-q8.log 2>&1 &
run $L/tiny-aya-global-q4_k_m.gguf aya-q4 $C
run $M/gemma-3-4b-it-Q4_K_M.gguf gemma3-4b-q4 $C
run $M/Llama-3.2-3B-Instruct-Q4_K_M.gguf llama32-3b-q4 $C
run $L/tiny-aya-global-q2_k.gguf aya-q2 orig,rev
run $L/tiny-aya-global-q3_k_m.gguf aya-q3 orig,rev
run $M/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf phi4-mini-q4 $C
run $Q8 aya-q8 orig,rev
run $L/tiny-aya-earth-q4_k_m.gguf aya-earth-q4 orig,rev
echo "[queue $(date +%H:%M)] ALL DONE" | tee -a logs/queue.log
