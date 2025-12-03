"""
Custom virtual environments for auto-sync.

Place .toml files in this directory containing pyproject.toml content.
The filename (without .toml) becomes the venv name for reference in
@custom_function(venv_name="...").

Example:
    # custom/venvs/ml_env.toml
    [project]
    name = "ml-env"
    version = "0.1.0"
    dependencies = ["torch>=2.0", "transformers>=4.30"]

Then reference in functions:
    @custom_function(venv_name="ml_env")
    async def my_ml_function():
        import torch
        ...
"""
