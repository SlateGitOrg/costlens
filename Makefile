# costlens developer commands. `make check` is the single gate CI runs.
UV ?= uv
RUN := $(UV) run --frozen
COV_MIN ?= 90
FIXTURE := examples/demo-cluster

.PHONY: help sync fmt format-check lint typecheck test cov check demo eval bench fixture clean

help:
	@echo "make check   - format-check + lint + type-check + tests with coverage >= $(COV_MIN)%"
	@echo "make demo    - offline end-to-end run against $(FIXTURE)"
	@echo "make eval    - allocation vs ground truth on the bundled fixture"
	@echo "make bench   - timing benchmark"
	@echo "make fixture - regenerate the bundled fixture (seed 7)"

sync:
	$(UV) sync --frozen

fmt:
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

format-check: sync
	$(RUN) ruff format --check src tests scripts

lint: sync
	$(RUN) ruff check src tests scripts

typecheck: sync
	$(RUN) mypy

test: sync
	$(RUN) pytest

cov: sync
	$(RUN) pytest --cov=costlens --cov-report=term-missing:skip-covered --cov-fail-under=$(COV_MIN)

check: format-check lint typecheck cov

demo: sync
	@echo "== validate ==================================================================="
	$(RUN) costlens validate $(FIXTURE)
	@echo "== allocate (examples/allocation.toml: team, shared platform, external costs) ==="
	$(RUN) costlens allocate $(FIXTURE) --config examples/allocation.toml
	@echo "== recommend (VPA defaults: p90 + 15%, 24h half-life, OOM bump) ================"
	$(RUN) costlens recommend $(FIXTURE)
	@echo "== backtest (train 3w | validate 1w | test 2w) ================================="
	$(RUN) costlens backtest $(FIXTURE)
	@echo "== report ===================================================================="
	$(RUN) costlens report $(FIXTURE) --config examples/allocation.toml --out out/demo-report
	@echo "== ground truth =============================================================="
	$(RUN) python scripts/eval_ground_truth.py $(FIXTURE)

eval: sync
	$(RUN) python scripts/eval_ground_truth.py $(FIXTURE)

bench: sync
	$(RUN) python scripts/bench.py $(FIXTURE)

fixture: sync
	rm -rf $(FIXTURE)
	$(RUN) costlens generate --out $(FIXTURE) --weeks 6 --step 15m --seed 7

clean:
	rm -rf out .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
