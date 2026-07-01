#!/usr/bin/env bash

RUSTFLAGS="--cfg=web_sys_unstable_apis" \
cargo build --release --target wasm32-unknown-unknown
#cargo build --release --target wasm32-unknown-unknown --features dev
 
wasm-bindgen --target web \
    --out-dir ../pkg/src/wasm \
    ../target/wasm32-unknown-unknown/release/kaleidomo_core.wasm
 
# wasm-bindgen outputs kaleidomo_core_bg.wasm — optimise that file in-place
wasm-opt ../pkg/src/wasm/kaleidomo_core_bg.wasm \
    -o ../pkg/src/wasm/kaleidomo_core_bg.wasm \
    -Oz \
    --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int
 
echo "Build successful"

rm -rf ../pkg/src/wasm/*.br
rm -rf ../pkg/src/wasm/*.gz

brotli --best ../pkg/src/wasm/kaleidomo_core_bg.wasm
gzip -k -9 ../pkg/src/wasm/kaleidomo_core_bg.wasm

echo "Compressed wasm files"

cp ../pkg/src/wasm/kaleidomo_core_bg.wasm /home/coding/.openclaw/workspace/abc/frontend/public/wasm/kaleidomo_core_bg.wasm
cp ../pkg/src/wasm/kaleidomo_core.js /home/coding/.openclaw/workspace/abc/frontend/public/wasm/kaleidomo_core.js
cp ../pkg/src/wasm/kaleidomo_core_bg.wasm.br /home/coding/.openclaw/workspace/abc/frontend/public/wasm/kaleidomo_core_bg.wasm.br
cp ../pkg/src/wasm/kaleidomo_core_bg.wasm.gz /home/coding/.openclaw/workspace/abc/frontend/public/wasm/kaleidomo_core_bg.wasm.gz

cp ../pkg/src/wasm/kaleidomo_core.d.ts /home/coding/.openclaw/workspace/abc/frontend/src/wasm/kaleidomo_core.d.ts
cp ../pkg/src/wasm/kaleidomo_core_bg.wasm.d.ts /home/coding/.openclaw/workspace/abc/frontend/src/wasm/kaleidomo_core_bg.wasm.d.ts

mkdir -p ../public/wasm
cp ../pkg/src/wasm/kaleidomo_core_bg.wasm ../public/wasm/kaleidomo_core_bg.wasm
cp ../pkg/src/wasm/kaleidomo_core.js      ../public/wasm/kaleidomo_core.js
echo "Copied wasm to public/wasm"

mkdir -p ../src/wasm
cp ../pkg/src/wasm/kaleidomo_core.d.ts    ../src/wasm/kaleidomo_core.d.ts