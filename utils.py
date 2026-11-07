"""Helpers shared by the training scripts."""
import argparse


def str2bool(value):
    """Parse 0/1, true/false, yes/no."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("1", "true", "t", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{value}'")
