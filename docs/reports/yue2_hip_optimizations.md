# YuE2 HIP optimization experiments

Tested on 2026-09-12 with Radeon 8060S (`gfx1151`), the installed ROCm
`10.1.0a20260910` package, and upstream `dev` commit
`13c4192a28d6a212f075c4cbefc5e4983e6ed52a`. This branch retains an **opt-in**
NAR SwiGLU fusion and reproducible CLI/latent comparison tools. It does not
change model precision, sampling, the 32 midpoint steps, or defaults.

## Retained candidate

`--session-option yue2.nar_fused_swiglu=true` combines the acoustic decoder's
SiLU and multiplication using the existing GGML `swiglu_split` operation.
The separate gate/up projections remain. The option is per session and defaults
to false, so it can be compared without rebuilding or global environment flags.

On this AMD backend the first complete WAV matched the reference byte for byte.
Additional fixed-input checks compared raw float32 acoustic latents: 256 frames
with seed 832002 and 1019 frames with seed 832003, both with a 749-token prefix,
32 midpoint steps and exactly equal output bytes. This is measured equivalence
for these cases, not a promise about every backend, weight format or input.

## Initial screening

Q8_0 main and F16 VAE from `audio-cpp/Yue2-3B-GGUF`, pinned to
`4c1742021dfaaa3f32d89d9eba89d3fde16a9d07`. The supplied instrumental ABC score
and seed 832002 generated 1237 codec tokens and 49.479 seconds of 48 kHz stereo.
Each row starts a fresh CLI process. No concurrent generation jobs were run.

| Candidate | Process seconds | NAR graph seconds | Identical WAV |
|---|---:|---:|---|
| Fork baseline | 44.31 | 23.09 | yes |
| Experimental switches off | 43.66 | 23.00 | yes |
| Fused SwiGLU | 43.27 | 22.72 | yes |
| Packed QKV projection | 43.36 | 23.34 | yes |
| Packed gate/up projection | 44.86 | 24.41 | yes |
| NAR weights converted to F16 | 44.95 | 24.46 | no |
| NAR weights converted to BF16 | 44.88 | 24.56 | no |

Packing did not establish a speed benefit. Changing NAR storage precision from
the loaded Q8 weights changed audio and did not improve process time. These
candidates are not included in the runtime patch. The initial SwiGLU difference
is small; the final alternating runs below are more useful than a single timing.
Do not compare different-duration songs by raw wall time.

## Reproduction

Build from the repository root:

```sh
scripts/build_linux.sh --backend hip --gpu-targets gfx1151 \
  --build-dir build/yue2-hip-opt --model-set custom --models yue2 \
  --target audiocpp_cli --jobs 12
cmake -S . -B build/yue2-hip-opt -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build/yue2-hip-opt --target yue2_nar_parity_probe -j 8
```

For full music, provide a JSON recipe with `style`, `lyrics`, `abc`, `cot`, and
`seed`. The helper defaults to a 1500 semantic-token cap and 32 synthesis steps;
optional `semantic_sampling.max_tokens` and `num_inference_steps` override them.
Use unique output directories. This tests generation without local playback.

```sh
python tests/yue2/benchmark_cli.py --binary build/yue2-hip-opt/bin/audiocpp_cli \
  --model /path/to/Yue2-3B-GGUF --recipe /path/to/recipe.json --out /tmp/yue2-control
python tests/yue2/benchmark_cli.py --binary build/yue2-hip-opt/bin/audiocpp_cli \
  --model /path/to/Yue2-3B-GGUF --recipe /path/to/recipe.json --out /tmp/yue2-fused \
  --session-option yue2.nar_fused_swiglu=true --reference /tmp/yue2-control/audio.wav
```

The helper retains the recipe, full command, source revision/diff, binary hash,
process time, logs and WAV hash. A differing reference WAV returns exit code 2.
Pin and retain the model revision alongside the run; the helper does not download
models or choose a revision.

For the stricter acoustic-only comparison, use raw native-endian int32 prefix
IDs and codec IDs. Codec IDs exclude the 151853 vocabulary offset. The probe
loads Q8 main weights, generates its seeded initial noise, and saves raw float32
latents before VAE decoding. The same seed must accompany each paired run.

```sh
build/yue2-hip-opt/bin/yue2_nar_parity_probe --model /path/to/Yue2-3B-GGUF \
  --prefix /path/to/prefix.i32 --codec /path/to/codec.i32 --seed 832002 \
  --output /tmp/nar-control.f32
build/yue2-hip-opt/bin/yue2_nar_parity_probe --model /path/to/Yue2-3B-GGUF \
  --prefix /path/to/prefix.i32 --codec /path/to/codec.i32 --seed 832002 \
  --fused-swiglu true --output /tmp/nar-fused.f32 --reference /tmp/nar-control.f32
```

The probe exits 2 on a bitwise mismatch and checks finite latents. Neither tool
establishes musical quality. Cross-runtime comparisons against Python require
separate care: the RNG and numeric paths differ even for the same seed.

## Local evidence

The private workstation's exact recipes, outputs, experimental source patch,
logs and model revision are retained under
`a-type/artifacts/analysis/20260912-audiocpp-optimizations/`. Those input scores
and generated media are not part of this public source patch. The existing
voice service continues to run its upstream build; this branch is an isolated
experiment and does not replace its binary.

## Final alternating comparisons

All three fused runs match their paired control WAV byte for byte. Pair B uses
seed 832003 (47.199 s audio); A and C use seed 832002 (49.479 s audio).

| Run | Seed | Process seconds | NAR graph seconds |
|---|---:|---:|---:|
| A control | 832002 | 43.518 | 23.059 |
| A fused | 832002 | 43.315 | 22.821 |
| B control | 832003 | 41.026 | 21.388 |
| B fused | 832003 | 40.922 | 21.261 |
| C control | 832002 | 43.707 | 23.201 |
| C fused | 832002 | 43.585 | 23.067 |

Mean paired process saving: **0.143 seconds**, approximately
**0.33%**.
This is a small, consistent observed difference in three pairs, not a statistically
established broad speedup. The option remains off by default. The larger
optimization candidates did not help this workload.

Rejected implementations remain available in experimental commit `1d0797f`;
the branch tip removes their environment switches and keeps only the explicit
SwiGLU option and verification tools.
