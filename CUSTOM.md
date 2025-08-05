# Custom modifcation
- Toolchain migrated to `uv` and `ruff`
- Local data placed in `/history` folder
- Added `Nexus` libary
- Custom trading stratagey and backtests placed in `/lab` folder
   To run an strategy: ```uv run -m lab.local_pmm_simple_backtest```




Editable install for active development
```
    git clone https://github.com/madpower2000/nexus.git 
    pip install -e ./nexus
```


Run RUFF
```
uv run ruff format .
uv run ruff check --fix .
uv run ruff check .
```

Show prochect folder structure" ```tree -L 2```
