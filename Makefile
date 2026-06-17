# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — Deep Learning Project
#  Targets: install | notebook | run | script | report | clean | help
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurable variables ────────────────────────────────────────────────────
PYTHON      := python
PIP         := $(PYTHON) -m pip
JUPYTER     := $(PYTHON) -m jupyter
NOTEBOOK    := notebooks/Proj1.ipynb
PY_SCRIPT   := src/python/som_project.py
REPORT_DIR  := docs/report
REPORT_TEX  := $(REPORT_DIR)/Report.tex
REPORT_PDF  := $(REPORT_DIR)/Report.pdf
FIGURES_DIR := outputs/figures
ANIM_DIR    := outputs/animations
VENV        := .venv
REQUIREMENTS := requirements.txt

# ── Detect OS for open command ────────────────────────────────────────────────
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    OPEN := open
else ifeq ($(UNAME), Linux)
    OPEN := xdg-open
else
    OPEN := start
endif

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all install venv notebook run script report open-report figures-dir animations-dir clean help

# ── Default target ────────────────────────────────────────────────────────────
all: help

# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

## Create a virtual environment
venv:
	@echo ">>> Creating virtual environment in $(VENV)/"
	$(PYTHON) -m venv $(VENV)
	@echo ">>> Activate it with: source $(VENV)/bin/activate"

## Generate requirements.txt from current environment (run after venv is active)
freeze:
	$(PIP) freeze > $(REQUIREMENTS)
	@echo ">>> $(REQUIREMENTS) updated."

## Install Python dependencies (system-wide or inside active venv)
install:
	@echo ">>> Installing Python dependencies..."
	$(PIP) install --upgrade pip
	@if [ -f $(REQUIREMENTS) ]; then \
		$(PIP) install -r $(REQUIREMENTS); \
	else \
		$(PIP) install numpy pandas matplotlib scikit-learn minisom jupyter nbconvert; \
		echo ">>> No requirements.txt found — installed default packages."; \
	fi
	@echo ">>> Done."

# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────

## Launch Jupyter Notebook (opens browser)
notebook:
	@echo ">>> Starting Jupyter Notebook..."
	$(JUPYTER) notebook $(NOTEBOOK)

## Execute the notebook non-interactively and save output in place
run:
	@echo ">>> Running notebook: $(NOTEBOOK)"
	$(JUPYTER) nbconvert \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.timeout=600 \
		$(NOTEBOOK)
	@echo ">>> Notebook executed successfully."

## Convert the executed notebook to HTML for easy sharing
html:
	@echo ">>> Converting notebook to HTML..."
	$(JUPYTER) nbconvert --to html $(NOTEBOOK)
	@echo ">>> Output: $(basename $(NOTEBOOK)).html"

## Run the Python script version of the notebook
script:
	@echo ">>> Running Python script: $(PY_SCRIPT)"
	$(PYTHON) $(PY_SCRIPT)

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

## Create the figures directory if it does not exist
figures-dir:
	@mkdir -p $(FIGURES_DIR)
	@echo ">>> $(FIGURES_DIR)/ is ready."

## Create the animations output directory
animations-dir:
	@mkdir -p $(ANIM_DIR)
	@echo ">>> $(ANIM_DIR)/ is ready."

# ─────────────────────────────────────────────────────────────────────────────
#  LATEX REPORT
# ─────────────────────────────────────────────────────────────────────────────

## Compile the LaTeX report to PDF (runs pdflatex twice for TOC)
report: $(REPORT_PDF)

$(REPORT_PDF): $(REPORT_TEX)
	@echo ">>> Compiling LaTeX report (pass 1)..."
	cd $(REPORT_DIR) && pdflatex -interaction=nonstopmode $(notdir $(REPORT_TEX))
	@echo ">>> Compiling LaTeX report (pass 2 — resolving TOC/references)..."
	cd $(REPORT_DIR) && pdflatex -interaction=nonstopmode $(notdir $(REPORT_TEX))
	@echo ">>> Report ready: $(REPORT_PDF)"

## Open the compiled PDF
open-report: $(REPORT_PDF)
	$(OPEN) $(REPORT_PDF)

## Remove LaTeX auxiliary files (keep the PDF)
clean-report:
	@echo ">>> Cleaning LaTeX auxiliary files..."
	cd $(REPORT_DIR) && rm -f \
		*.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk \
		*.synctex.gz *.blg *.bbl
	@echo ">>> Done."

# ─────────────────────────────────────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

## Remove all generated artefacts (LaTeX aux, notebook checkpoints, caches)
clean: clean-report
	@echo ">>> Removing notebook checkpoints and Python caches..."
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints __pycache__ *.pyc
	@echo ">>> Clean complete."

## Remove everything including the compiled PDF and virtual environment
distclean: clean
	@echo ">>> Removing compiled PDF and virtual environment..."
	rm -f $(REPORT_PDF)
	rm -rf $(VENV)
	@echo ">>> Distclean complete."

# ─────────────────────────────────────────────────────────────────────────────
#  HELP
# ─────────────────────────────────────────────────────────────────────────────

## Print this help message
help:
	@echo ""
	@echo "  Deep Learning Project — Makefile"
	@echo "  ─────────────────────────────────────────────────────"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  Environment"
	@echo "    venv          Create a Python virtual environment in .venv/"
	@echo "    install       Install Python dependencies (pip)"
	@echo "    freeze        Save current pip packages to requirements.txt"
	@echo ""
	@echo "  Notebook"
	@echo "    notebook      Launch Jupyter Notebook in browser"
	@echo "    run           Execute notebook non-interactively (saves output)"
	@echo "    html          Convert notebook to HTML"
	@echo "    script        Run the Python script version"
	@echo ""
	@echo "  Report"
	@echo "    report        Compile LaTeX report → PDF"
	@echo "    open-report   Open the compiled PDF"
	@echo "    clean-report  Remove LaTeX auxiliary files (keeps PDF)"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean         Remove aux files, caches, checkpoints"
	@echo "    distclean     Remove everything including PDF and venv"
	@echo ""
	@echo "  Other"
	@echo "    figures-dir    Create outputs/figures/ directory"
	@echo "    animations-dir  Create outputs/animations/ directory"
	@echo "    help          Show this message"
	@echo ""
