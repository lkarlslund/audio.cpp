#!/usr/bin/env python3
"""Run one reproducible YuE2 CLI audition; optionally require identical WAV bytes.

The recipe is JSON with style, lyrics, abc, cot, seed and optional
semantic_sampling.max_tokens / num_inference_steps. Outputs are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', default='hip')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--session-option', action='append', default=[])
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    (out / 'score.abc').write_text(recipe['abc'])
    command = [str(args.binary.resolve()), '--model', str(args.model.resolve()),
               '--family', 'yue2', '--task', 'gen', '--backend', args.backend,
               '--threads', str(args.threads), '--lyrics', recipe['lyrics'],
               '--seed', str(recipe['seed']), '--out', str(out / 'audio.wav'), '--log']
    options = dict(style=recipe['style'], cot=recipe.get('cot', 'full'),
                   abc_file=str(out / 'score.abc'),
                   semantic_max_tokens=recipe.get('semantic_sampling', {}).get('max_tokens', 1500),
                   num_inference_steps=recipe.get('num_inference_steps', 32))
    for group in ('semantic', 'abc'):
        for key, value in recipe.get(group + '_sampling', {}).items():
            options[group + '_' + key] = value
    if recipe.get('cfg_scale') is not None:
        options['cfg_scale'] = recipe['cfg_scale']
    for key, value in options.items():
        command += ['--request-option', f'{key}={value}']
    for option in args.session_option:
        command += ['--session-option', option]
    root = Path(__file__).resolve().parents[2]
    metadata = dict(command=command, binary_sha256=sha256(args.binary),
                    source_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                    environment={k: v for k, v in os.environ.items()
                                 if k.startswith(('GGML_', 'HIP_', 'ROCR_', 'HSA_'))})
    (out / 'source.patch').write_bytes(subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD']))
    (out / 'command.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    with (out / 'run.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(seconds=time.monotonic() - started, returncode=result.returncode)
    if result.returncode == 0:
        audio = out / 'audio.wav'
        metadata['audio_sha256'] = sha256(audio)
        with wave.open(str(audio)) as wav:
            metadata.update(duration=wav.getnframes() / wav.getframerate(),
                            sample_rate=wav.getframerate(), channels=wav.getnchannels())
        if args.reference:
            metadata['identical_wav'] = metadata['audio_sha256'] == sha256(args.reference)
    (out / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    return result.returncode or (2 if metadata.get('identical_wav') is False else 0)


if __name__ == '__main__':
    raise SystemExit(main())
