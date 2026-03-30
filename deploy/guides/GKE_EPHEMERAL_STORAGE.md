# GKE Ephemeral Storage & Volume Strategy

## The Problem

Unity jobs run on **GKE Autopilot**, which enforces a hard **10Gi cap on ephemeral storage** per pod. This limit is imposed by Google's `autogke-pod-limit-constraints` admission webhook and cannot be raised — it's a platform constraint, not a cluster configuration.

Ephemeral storage covers the container's writable filesystem layer: anything written to paths not backed by an explicit volume mount (e.g. `/root/.cache/`, `/usr/local/lib/`, `/var/log/`). Some Unity components lazily download large artifacts at runtime, notably:

- **Docling's SmolVLM-500M-Instruct model** (~5GB via HuggingFace Hub) — downloaded on the first PDF parse with picture description enabled
- **PackageOverlay pip installs** — dynamic package installs during actor execution
- **General temp files** — `tempfile.mkdtemp()` calls throughout the codebase

Without intervention, the HuggingFace model download alone nearly exhausts the 10Gi limit.

## The Solution: `emptyDir` Volume on `/tmp`

The job spec in `communication/infra/helpers.py` mounts an `emptyDir` volume at `/tmp`:

```python
"volumeMounts": [
    {"name": "tmp-vol", "mountPath": "/tmp"},
]
"volumes": [
    {"name": "tmp-vol", "emptyDir": {}},
]
```

An `emptyDir` volume:
- Uses the **node's disk**, not counted against the pod's `ephemeral-storage` limit
- Has **no size cap** by default (bounded only by node disk availability)
- Is **automatically cleaned up** when the pod terminates
- Has **identical performance** to ephemeral storage (same underlying disk)
- Is **fully isolated** per pod — two jobs on the same node cannot see each other's volumes
- Is **billed through the pod's `ephemeral-storage` request** on Autopilot, not based on actual usage

## Environment Variables

Two env vars are set in the job spec (`communication/infra/helpers.py`):

| Env Var | Value | What It Redirects |
|---------|-------|-------------------|
| `HF_HOME` | `/tmp/huggingface` | HuggingFace Hub model cache (turn detector, Docling SmolVLM, any future HF models) |
| `XDG_CACHE_HOME` | `/tmp/.cache` | Catch-all for XDG-compliant tools that write to `~/.cache/` (pip, various CLI tools, etc.) |

Both point to paths under `/tmp`, which is backed by the `emptyDir` volume (not counted against ephemeral storage).

## What Goes Where

| Component | Path | Storage Type | Mechanism |
|-----------|------|-------------|-----------|
| Turn detector models | `/tmp/huggingface/` | `emptyDir` volume (copied from image at startup) | `HF_HOME` env var + entrypoint copy |
| Docling / SmolVLM (~5GB) | `/tmp/huggingface/` | `emptyDir` volume (runtime download) | `HF_HOME` env var |
| PackageOverlay (actor pip installs) | `/tmp/unity_act_pkgs/` | `emptyDir` volume | Already uses `/tmp` |
| `tempfile.mkdtemp()` calls | `/tmp/` | `emptyDir` volume | Already uses `/tmp` |
| XDG-compliant caches | `/tmp/.cache/` | `emptyDir` volume | `XDG_CACHE_HOME` env var |
| Playwright browsers | Image layer | Image layer (baked in) | Installed at Docker build time |
| Python packages | Image layer | Image layer (baked in) | Installed at Docker build time |

## What's Baked Into the Docker Image (Not a Concern)

The Dockerfile installs these at build time, so they live in the image layers and don't consume ephemeral storage:

- All Python packages (`uv pip install`)
- Playwright browsers (`playwright install`)
- Turn detector models (`call.py download-files`)
- Node.js packages (`npm ci`)
- PyTorch CPU

### Build-Time Models and the `emptyDir` at `/tmp`

The Dockerfile downloads turn detector models to the default HuggingFace cache (`/root/.cache/huggingface/`). At runtime, `HF_HOME=/tmp/huggingface` redirects all HuggingFace lookups to the `emptyDir` volume — but that volume starts empty. An `emptyDir` mount **replaces** everything the image had at `/tmp`, so we can't download directly to `/tmp` during the build.

To bridge this, `entrypoint.sh` copies the baked-in models into the `emptyDir` on container startup:

```bash
if [ -d /root/.cache/huggingface ] && [ ! -d /tmp/huggingface ]; then
    cp -r /root/.cache/huggingface /tmp/huggingface
fi
```

This is a fast one-time copy (turn detector models are small). After that, all HuggingFace operations — both lookups of pre-downloaded models and lazy runtime downloads like Docling's SmolVLM — use `/tmp/huggingface/` on the `emptyDir` volume, keeping everything off ephemeral storage.

## Why Not Just Increase `ephemeral-storage`?

GKE Autopilot caps `ephemeral-storage` at **10Gi per pod**. This is a hard platform limit enforced by the `autogke-pod-limit-constraints` admission webhook. Requesting more (e.g. 100Gi) results in a 400 error:

```
GKE Warden rejected the request because it violates one or more constraints.
Violations: Total ephemeral-storage requested by containers for workload is
higher than the Autopilot maximum of '10Gi'.
```

The only way to exceed 10Gi of scratch space on Autopilot is via volumes (`emptyDir` or PVC).

## Adding Future Large Downloads

When a new tool or library needs to download or cache large files at runtime, follow this priority order:

1. **Check if it respects `XDG_CACHE_HOME`.** Many CLI tools and Python libraries do. If so, it's already handled — the `XDG_CACHE_HOME=/tmp/.cache` env var catches it automatically with no changes needed.

2. **Check if it has its own cache env var.** Most well-known libraries do (e.g. `HF_HOME` for HuggingFace, `PLAYWRIGHT_BROWSERS_PATH` for Playwright, `PIP_CACHE_DIR` for pip, `TORCH_HOME` for PyTorch Hub). Add the env var to the job spec in `communication/infra/helpers.py`, pointing to `/tmp/<something>`.

3. **If the tool has no env var and hardcodes a path** (rare), either symlink that path to `/tmp` in the Dockerfile, or patch the tool's config at startup in `entrypoint.sh`.

The one case this strategy doesn't cover is runtime `pip install` to the system Python (`/usr/local/lib/...`), which always hits ephemeral storage since it modifies the root filesystem. `PackageOverlay` already installs to `/tmp/unity_act_pkgs/`, so the main actor flow is fine. Only `EnvironmentManager._ensure_dependencies()` installs to the system path, and those are typically small.

**The general rule: if something downloads large files at runtime, find the env var, point it at `/tmp`, done. No infrastructure changes needed.**

## Ephemeral Storage Sizing

The job spec retains a **10Gi** `ephemeral-storage` request. With the heavy-hitters redirected to `/tmp` (backed by `emptyDir`), the ephemeral storage only needs to cover lightweight writes to the container's root filesystem:

- Container writable layer overhead
- `EnvironmentManager` runtime pip installs to `/usr/local/lib/...` (small, occasional)
- `FunctionManager` venv creation if it writes outside `/tmp`
- Log files written to paths like `/var/log/`

10Gi is generous for these, but the cost is negligible — Autopilot charges ~$0.0003/GiB-hour, so 10Gi works out to roughly ~$2/month for a pod running 24/7. Since Unity pods are short-lived, the actual cost is far less. Keeping 10Gi provides a comfortable buffer against unexpected writes without meaningful cost impact.

The real value of the `emptyDir` + env var setup isn't cost savings — it's breaking through the 10Gi Autopilot ceiling for downloads that genuinely need more (like the ~5GB HuggingFace model).

## Alternatives Considered

| Approach | Verdict |
|----------|---------|
| **Switch to GKE Standard** | Removes the 10Gi cap but adds significant ops burden (node pool management, autoscaler tuning, OS patching, security hardening). Not worth it for this use case. |
| **PersistentVolumeClaim** | Overkill for ephemeral scratch data. PVCs persist beyond pod lifetime and require cleanup. `emptyDir` matches the pod lifecycle exactly. |
| **Pre-download models in Dockerfile** | Would work for Docling, but increases image size by ~5GB and slows down every image pull. The model is only needed for PDF parsing with picture description, so lazy download is more efficient. |
| **`emptyDir` with `medium: Memory`** | Faster (backed by tmpfs/RAM) but counts against the pod's memory limit. Would require increasing the 8Gi memory request significantly. |
