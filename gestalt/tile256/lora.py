# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Decoder adaptation: one entry point for both conditioning signals.

    --signal mapper   condition on a predicted foundation-model embedding

Both train a rank-32 (alpha 64) LoRA for 8000 steps with PixCell's own denoising objective, on
real (embedding, image) pairs only.

--train-seed controls adapter initialization, batch order and CUDA. --split-seed selects the
holdout split and is kept separate, so the training set is identical across training seeds.
"""
import sys


def main():
    # --signal is accepted for backward compatibility with older invocations; 'mapper' is the
    # only conditioning signal this release trains an adapter for.
    if '--signal' in sys.argv:
        i = sys.argv.index('--signal')
        signal = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if signal != 'mapper':
            raise SystemExit(f'lora.py: unknown --signal {signal!r}; only \'mapper\' is supported')
    from ._lora_mapper import main as run
    run()


if __name__ == '__main__':
    main()
