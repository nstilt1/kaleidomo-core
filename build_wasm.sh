#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname -- "$SCRIPT_DIR")"
WASM_DIR="$WORKSPACE_DIR/pkg/src/wasm"
PUBLIC_WASM_DIR="$WORKSPACE_DIR/public/wasm"
TYPES_DIR="$WORKSPACE_DIR/src/wasm"
TARGET_WASM="$WORKSPACE_DIR/target/wasm32-unknown-unknown/release/kaleidomo_core.wasm"

mkdir -p "$WASM_DIR"

RUSTFLAGS="--cfg=web_sys_unstable_apis" \
cargo build \
    --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    --package kaleidomo-core \
    --release \
    --target wasm32-unknown-unknown
#cargo build --release --target wasm32-unknown-unknown --features dev
 
wasm-bindgen --target web \
    --out-dir "$WASM_DIR" \
    "$TARGET_WASM"
 
# wasm-bindgen outputs kaleidomo_core_bg.wasm — optimise that file in-place
wasm-opt "$WASM_DIR/kaleidomo_core_bg.wasm" \
    -o "$WASM_DIR/kaleidomo_core_bg.wasm" \
    -Oz \
    --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int
 
rm -f "$WASM_DIR/kaleidomo_core_bg.wasm.br"
rm -f "$WASM_DIR/kaleidomo_core_bg.wasm.gz"

brotli --best "$WASM_DIR/kaleidomo_core_bg.wasm"
gzip -k -9 "$WASM_DIR/kaleidomo_core_bg.wasm"

echo "Compressed wasm files"

EXTERNAL_PUBLIC_WASM_DIR="/home/coding/.openclaw/workspace/abc/frontend/public/wasm"
EXTERNAL_TYPES_DIR="/home/coding/.openclaw/workspace/abc/frontend/src/wasm"

copy_if_destination_exists() {
    local source_path="$1"
    local destination_dir="$2"

    if [[ -d "$destination_dir" ]]; then
        cp "$source_path" "$destination_dir/$(basename -- "$source_path")"
    else
        echo "x $destination_dir not found, not copying"
    fi
}

copy_if_destination_exists "$WASM_DIR/kaleidomo_core_bg.wasm" "$EXTERNAL_PUBLIC_WASM_DIR"
copy_if_destination_exists "$WASM_DIR/kaleidomo_core.js" "$EXTERNAL_PUBLIC_WASM_DIR"
copy_if_destination_exists "$WASM_DIR/kaleidomo_core_bg.wasm.br" "$EXTERNAL_PUBLIC_WASM_DIR"
copy_if_destination_exists "$WASM_DIR/kaleidomo_core_bg.wasm.gz" "$EXTERNAL_PUBLIC_WASM_DIR"
copy_if_destination_exists "$WASM_DIR/kaleidomo_core.d.ts" "$EXTERNAL_TYPES_DIR"
copy_if_destination_exists "$WASM_DIR/kaleidomo_core_bg.wasm.d.ts" "$EXTERNAL_TYPES_DIR"

mkdir -p "$PUBLIC_WASM_DIR"
cp "$WASM_DIR/kaleidomo_core_bg.wasm" "$PUBLIC_WASM_DIR/kaleidomo_core_bg.wasm"
cp "$WASM_DIR/kaleidomo_core.js" "$PUBLIC_WASM_DIR/kaleidomo_core.js"
echo "Copied wasm to public/wasm"

mkdir -p "$TYPES_DIR"
cp "$WASM_DIR/kaleidomo_core.d.ts" "$TYPES_DIR/kaleidomo_core.d.ts"
cp "$WASM_DIR/kaleidomo_core_bg.wasm.d.ts" "$TYPES_DIR/kaleidomo_core_bg.wasm.d.ts"

echo "Build successful"
