#!/bin/bash
# Pelny pipeline: konwersja HF -> BF16 GGUF -> poprawki (pad_gguf.py) -> kwanty.
# Uzycie: ./make_gguf.sh <katalog_modelu_hf> <katalog_wyjsciowy> [llama-quantize]
set -euo pipefail

MODEL_DIR=$(realpath "$1")
OUT_DIR=$(realpath -m "$2")
QUANTIZE=${3:-llama-quantize}
QUANTS=${QUANTS:-"Q8_0 Q6_K Q5_K_M Q4_K_M"}

HERE=$(dirname "$(realpath "$0")")
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HERE/llama.cpp}
LLAMA_CPP_COMMIT=0ec4027a2a0183de87c007069adddc06e665f7de  # = release b12020
NAME=$(basename "$MODEL_DIR")

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
    git -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_COMMIT"
    git -C "$LLAMA_CPP_DIR" apply "$HERE/llama.cpp-bailingmoe2-spm-tokenizer.patch"
fi

mkdir -p "$OUT_DIR"
RAW="$OUT_DIR/$NAME-raw-BF16.gguf"

python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype bf16 --outfile "$RAW"
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" python3 "$HERE/pad_gguf.py" "$RAW" "$OUT_DIR/$NAME-BF16.gguf" "$MODEL_DIR/chat_template.jinja"
rm "$RAW"

for q in $QUANTS; do
    "$QUANTIZE" "$OUT_DIR/$NAME-BF16.gguf" "$OUT_DIR/$NAME-$q.gguf" "$q"
done
