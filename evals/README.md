# HierarchicalActor Initial Plan Generator

This directory contains scripts for generating initial HierarchicalActor plans for evaluation tasks.

## Overview

The `generate_initial_plans.py` script generates fully-implemented Python plans for tasks without executing them. It supports multiple JSONL input formats and ensures all generated code includes proper imports and initialization.

## Features

- **No Stub Generation**: All functions are fully implemented (no `NotImplementedError`)
- **Automatic Setup**: Adds required imports and `action_provider` initialization
- **Multiple Format Support**: Handles various JSONL task formats
- **Batch Processing**: Process multiple tasks from JSONL files
- **Web Navigation Support**: Enhanced prompts for browser automation tasks

## Supported JSONL Formats

### Format 1: Simple task format
```json
{"task_id": "wv_001", "task": "Find a vegetarian lasagna recipe..."}
```

### Format 2: WebVoyager format with website context
```json
{
  "web_name": "Allrecipes",
  "id": "Allrecipes--0",
  "ques": "Provide a recipe for vegetarian lasagna...",
  "web": "https://www.allrecipes.com/"
}
```

The script automatically detects the format and enhances web-based tasks with navigation context.

## Usage

### Single Task Mode
```bash
python evals/generate_initial_plans.py "Find the latest news about AI" -i my_task_001
```

### Batch Mode with JSONL
```bash
# Process all tasks
python evals/generate_initial_plans.py tasks.jsonl -o output_dir

# Process first 10 tasks
python evals/generate_initial_plans.py tasks.jsonl -o output_dir -l 10
```

### Examples

1. **WebVoyager Tasks** (example_tasks.jsonl):
```bash
python evals/generate_initial_plans.py evals/example_tasks.jsonl -o evals/batch_output -l 3
```

2. **Patched Tasks** (patchedTasks.jsonl):
```bash
./evals/test_patched_tasks.sh
# Or manually:
python evals/generate_initial_plans.py evals/patchedTasks.jsonl -o evals/patched_plans -l 5
```

## Output

Each generated plan is saved as a separate Python file with:
- Full imports (asyncio, re, pydantic, typing, ActionProvider)
- Initialized `action_provider = ActionProvider()`
- Fully implemented async functions
- Proper error handling and structured data extraction

### Output Structure
```
output_dir/
├── task_001.py
├── task_002.py
├── ...
└── generation_summary.json
```

The `generation_summary.json` contains:
- Total tasks processed
- Success/failure counts
- Any plans that contain stubs (should be 0)
- Error messages for failed generations

## Generated Code Features

All generated plans include:

1. **Standard Imports**:
```python
import asyncio
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from unity.actor.action_provider import ActionProvider
```

2. **Action Provider Initialization**:
```python
action_provider = ActionProvider()
```

3. **Main Entry Point**:
```python
async def main_plan():
    # Implementation
```

4. **Browser Navigation** (for web tasks):
```python
await action_provider.browser_act("Navigate to https://...")
await action_provider.browser_observe("What elements are visible?")
```

5. **Structured Data Extraction**:
```python
class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: str = Field(description="Price with currency")
```

## Troubleshooting

If plans are missing imports or `action_provider` initialization:
1. Check that you're using the latest version of the script
2. Verify the `post_process_generated_code` function is being called
3. Check the generation logs in `evals/plan_generation.log`
