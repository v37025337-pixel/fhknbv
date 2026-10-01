from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_mind_core.checkpoint import (
    read_checkpoint_document,
    restore_checkpoint_document,
    save_checkpoint,
)
from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.supabase_checkpoint import SupabaseCheckpointStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    mind = UnifiedMind(KernelConfig(seed=args.seed))
    mind.run(args.steps)
    checkpoint_id = save_checkpoint(mind, args.checkpoint_out)

    result = {
        "checkpoint_id": checkpoint_id,
        "steps": mind.fast_steps,
        "path": str(args.checkpoint_out),
        "uploaded": False,
    }

    if not args.no_upload:
        store = SupabaseCheckpointStore()
        row = store.push_file(args.checkpoint_out)
        pulled = store.pull_document()
        restored = restore_checkpoint_document(pulled)
        if restored.fast_steps != mind.fast_steps:
            raise RuntimeError("Supabase readback restored a different step count")
        if pulled["integrity"]["checkpoint_id"] != checkpoint_id:
            raise RuntimeError("Supabase readback checkpoint id mismatch")
        result.update({
            "uploaded": True,
            "generation": row["generation"],
            "byte_size": row["byte_size"],
            "materialized_at": row["materialized_at"],
            "verified_at": row["verified_at"],
        })

    # Re-read locally as a final artifact integrity check.
    read_checkpoint_document(args.checkpoint_out)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
