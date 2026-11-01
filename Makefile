PY ?= $(HOME)/miniconda3/bin/python
FLOW = $(PY) -m flow

.PHONY: all build simulate check power fetch compare table clean

all: table

build:
	$(FLOW) build

simulate: build
	$(FLOW) simulate

check: build
	$(FLOW) check-gating

power:
	$(FLOW) power

fetch:
	$(FLOW) fetch

compare:
	$(FLOW) compare

table:
	$(FLOW) table

clean:
	rm -rf build results/sims results/activity results/problems_p32 results/power results/resynth results/lm/*/problems results/lm/*/sims
