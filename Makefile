# Thin wrapper over `python -m pipeline`; every target also works without make.
PY ?= python
TECTONIC ?= tectonic

.PHONY: all fetch clean run failures figures report paper test lint

all:
	$(PY) -m pipeline all

fetch clean run failures figures report:
	$(PY) -m pipeline $@

paper:
	cd paper && $(TECTONIC) --outdir build main.tex
	cp paper/build/main.pdf paper/solar-two-sites-v1.pdf

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
