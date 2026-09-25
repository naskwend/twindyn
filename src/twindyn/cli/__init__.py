"""Command-line entry points.

Ref: Sec. 4.4 (train), Sec. 4.1 (prepare and evaluate), Table 2 (ablation),
Table 4 and Sec. 3.10 (sweep).
"""

from twindyn.cli.ablation import main as ablation_main
from twindyn.cli.evaluate import main as evaluate_main
from twindyn.cli.prepare_data import main as prepare_data_main
from twindyn.cli.sweep import main as sweep_main
from twindyn.cli.train import main as train_main
from twindyn.cli.verify import main as verify_main

__all__ = [
    "ablation_main",
    "evaluate_main",
    "prepare_data_main",
    "sweep_main",
    "train_main",
    "verify_main",
]
