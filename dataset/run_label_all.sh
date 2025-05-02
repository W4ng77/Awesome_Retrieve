#!/bin/bash

# === ✅ Step 2: 设置参数 ===
INPUT_DIR="longbench_support_facts_combined"
SCRIPT="label_support_facts_with_openai.py"
MODEL="gpt-3.5-turbo"
MAX_WORKERS=10
TASKS=("hotpotqa" "2wikimqa" "musique" "multifieldqa_en")

# === ✅ Step 3: 逐个运行 ===
for TASK in "${TASKS[@]}"; do
  INPUT_FILE="$INPUT_DIR/${TASK}_chunk64.json"
  OUTPUT_FILE="$INPUT_DIR/${TASK}_chunk64_labeled.json"

  if [ -f "$INPUT_FILE" ]; then
    echo "🚀 Running labeling on: $INPUT_FILE"
    python "$SCRIPT" --input "$INPUT_FILE" --model "$MODEL" --max_workers $MAX_WORKERS
    echo "✅ Finished: $OUTPUT_FILE"
  else
    echo "⚠️  Skipped (not found): $INPUT_FILE"
  fi
done

echo "🎉 All tasks completed!"
