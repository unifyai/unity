# Grid Search (`grid_search.sh`)

Run tests across all combinations of settings values. This is useful for:

- **Model comparisons**: Compare behavior across different LLMs
- **Feature flag ablations**: Test with settings enabled/disabled
- **Configuration sweeps**: Find optimal settings combinations

## Quick Start

```bash
# Grid search across models
grid_search.sh --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" tests/contact_manager/

# Grid search across models AND cache settings (2×2 = 4 combinations)
grid_search.sh --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" --env UNILLM_CACHE="true|false" tests/
```

---

## How It Works

1. **Parse grid variables**: Settings with `|` separators define the search space
2. **Generate combinations**: Full Cartesian product of all values
3. **Launch runs**: Each combination spawns a separate `parallel_run` invocation
4. **Auto-tag**: Each run is automatically tagged with its `--env` values for easy filtering
5. **Log results**: Each run logs both tags and full settings dict to `Combined`

---

## Syntax

```bash
grid_search.sh [options] --env KEY=val1|val2|val3 [--env KEY2=a|b] [targets...]
```

- **Pipe (`|`)**: Separates values to grid over
- **No pipe**: Single value passed through to all runs
- **Targets**: Test files/directories (same as `parallel_run`)

---

## Options

| Option | Description |
|--------|-------------|
| `--env KEY=val1\|val2` | Grid variable (multiple values, pipe-separated); each value becomes a separate run |
| `--env KEY=value` | Constant variable (single value for all runs, included in auto-tags) |
| `-n`, `--dry-run` | Show generated commands without executing (including auto-tags) |
| `-h`, `--help` | Show help |

All other options are passed through to `parallel_run`.

Note: Combinations run sequentially since each `parallel_run` call blocks until tests complete.

---

## Auto-Tagging

Each run is **automatically tagged** with all `--env` values passed to `grid_search.sh`. This makes post-hoc analysis trivial—you can filter results by the exact configuration used for each run.

**How it works:**

- Tags are formatted as `KEY1=val1,KEY2=val2,...` (comma-separated)
- **Grid variables** (with `|`): The specific value selected for that run is tagged
- **Constant variables** (no `|`): Included in tags for all runs
- **Background variables** (from `.env` file): NOT included in tags

**Example:**

```bash
grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNILLM_CACHE="true|false" tests/
```

Generates 4 runs with these auto-tags:

| Run | Tags |
|-----|------|
| 1 | `UNIFY_MODEL=gpt-4o,UNILLM_CACHE=true` |
| 2 | `UNIFY_MODEL=gpt-4o,UNILLM_CACHE=false` |
| 3 | `UNIFY_MODEL=claude-3,UNILLM_CACHE=true` |
| 4 | `UNIFY_MODEL=claude-3,UNILLM_CACHE=false` |

With a constant variable:

```bash
grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env EXPERIMENT_ID="exp-42" tests/
```

Generates 2 runs:

| Run | Tags |
|-----|------|
| 1 | `UNIFY_MODEL=gpt-4o,EXPERIMENT_ID=exp-42` |
| 2 | `UNIFY_MODEL=claude-3,EXPERIMENT_ID=exp-42` |

---

## Examples

**Model comparison with eval tests:**

```bash
grid_search.sh \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic|gemini-2.5-pro@google" \
  --env UNILLM_CACHE="false" \
  --eval-only \
  tests/contact_manager/
```

This generates 3 runs (one per model), each with fresh LLM calls.

**Feature flag ablation:**

```bash
grid_search.sh \
  --env FIRST_ASK_TOOL_IS_SEARCH="true|false" \
  --env FIRST_MUTATION_TOOL_IS_ASK="true|false" \
  tests/actor/
```

This generates 4 runs (2×2 grid) testing all combinations of these two feature flags.

**Dry run to preview:**

```bash
grid_search.sh -n \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" \
  --env UNILLM_CACHE="true|false" \
  tests/
```

Output:

```
Grid Search Configuration
=========================
Grid variables:
  UNIFY_MODEL: gpt-4o@openai | claude-sonnet-4-20250514@anthropic
  UNILLM_CACHE: true | false

Total combinations: 4

Generated runs:
  [1/4] UNIFY_MODEL=gpt-4o@openai UNILLM_CACHE=true
  [2/4] UNIFY_MODEL=gpt-4o@openai UNILLM_CACHE=false
  [3/4] UNIFY_MODEL=claude-sonnet-4-20250514@anthropic UNILLM_CACHE=true
  [4/4] UNIFY_MODEL=claude-sonnet-4-20250514@anthropic UNILLM_CACHE=false

Dry run - commands that would be executed:

  parallel_run --env UNIFY_MODEL=gpt-4o@openai --env UNILLM_CACHE=true --tags UNIFY_MODEL=gpt-4o@openai,UNILLM_CACHE=true tests/
  parallel_run --env UNIFY_MODEL=gpt-4o@openai --env UNILLM_CACHE=false --tags UNIFY_MODEL=gpt-4o@openai,UNILLM_CACHE=false tests/
  ...
```

---

## Analyzing Results

After a grid search, query the `Combined` context to compare results:

```python
import unify

unify.activate("UnityTests")
logs = unify.get_logs(context="Combined")

# Filter by tags (contains the exact --env values from the grid search)
for log in logs:
    tags = log.get("tags", [])
    duration = log.get("duration", 0)
    # Tags are like ["UNIFY_MODEL=gpt-4o", "UNILLM_CACHE=true"]
    print(f"{tags}: {duration:.2f}s")

# Or filter by specific tag values
gpt4_runs = [log for log in logs if "UNIFY_MODEL=gpt-4o" in log.get("tags", [])]
```

Or use the Unify dashboard to filter by `tags` (exact match) or `settings.UNIFY_MODEL` (for all values).

---

## Combining with Other Features

Grid search composes with all `parallel_run` features:

```bash
# Grid + eval-only + repeat for statistical sampling
grid_search.sh \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" \
  --env UNILLM_CACHE="false" \
  --eval-only \
  --repeat 5 \
  tests/contact_manager/test_ask.py
```

This generates 2 models × 5 repeats = 10 runs, useful for comparing pass rates across models.
