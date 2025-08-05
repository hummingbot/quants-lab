# Contributor Guide

## Dev Environment Tips
- For package and project management use [`uv`](https://github.com/astral-sh/uv).
- For example, to run python file `ma_cross.py` located in `/backtests` folder run command `uv run -m backtests.ma_cross`
-  For linting and code  and imports formatting use [`ruff`](https://github.com/astral-sh/ruff), to run ruff use command : `uvx ruff check .`
- For Static Typing for Python use [`mypy`](https://github.com/python/mypy), as `uvx mypy <file_name> `
- Alway run tests after any code chages: ```uv run pytest tests/```
