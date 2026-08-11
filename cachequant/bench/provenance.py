import os
import platform

import torch
import transformers


def cpu_model_name() -> str:
    """Best-effort human-readable CPU model name.

    platform.processor() is frequently an empty string on Linux (it relies
    on info the OS doesn't always populate), so this falls back to parsing
    /proc/cpuinfo, and finally to platform.machine(), so the provenance
    disclosure always names *something* concrete.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def provenance() -> dict:
    return {
        "platform_processor": platform.processor(),
        "cpu_model": cpu_model_name(),
        "platform_machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
