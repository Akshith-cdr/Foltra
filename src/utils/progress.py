"""Dependency-free batch progress including loading and computation time."""
from time import perf_counter


def progress_batches(loader, label, interval=20):
    total = len(loader)
    started = perf_counter()
    print(f"{label}: starting {total} batches", flush=True)
    for step, batch in enumerate(loader, 1):
        yield batch
        if step == 1 or step % max(1, interval) == 0 or step == total:
            elapsed = perf_counter() - started
            remaining = elapsed / step * (total - step)
            print(f"{label}: {step}/{total} batches | elapsed {elapsed / 60:.1f} min | "
                  f"ETA {remaining / 60:.1f} min ({elapsed / step:.2f} s/batch)", flush=True)
