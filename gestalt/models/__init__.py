# Copyright (c) 2026, the Gestalt authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The three conditioning-signal objectives.

All three share `gestalt.core.backbone`; only the head and the training loss differ, and each
gets its own module.

    mapper       supervised regression to a foundation-model embedding
"""

__all__ = []
