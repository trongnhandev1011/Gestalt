# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Figure generation.

    python -m gestalt.tile256.viz qualitative ...   GT / no-LoRA / LoRA comparison grids
    python -m gestalt.tile256.viz contact ...       contact sheets of generated tiles

Region-synthesis figures are produced by `gestalt.region1024.viz`.
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('qualitative', 'contact'):
        raise SystemExit('usage: viz.py {qualitative,contact} [args...]')
    what = sys.argv.pop(1)
    if what == 'qualitative':
        from ._viz_qualitative import main as run
    else:
        from ._viz_contact import main as run
    run()


if __name__ == '__main__':
    main()
