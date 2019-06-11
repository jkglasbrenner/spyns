.PHONY: beautify build build-ext clean conda dev docs help hooks lint sdist test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = spyns

RM = rm
COPY = cp
FIND = find

CONDA = conda
CONDA_ENV_FILE = environment.yml

PY ?= python3
PY_SRC = $(PROJECT_NAME)
PY_SETUP = setup.py
PY_SETUP_BUILD = build
PY_SETUP_BUILD_EXT = build_ext --inplace
PY_SETUP_DOCS = build_sphinx
PY_SETUP_SDIST = sdist
PY_SETUP_TEST = test

PRECOMMIT = pre-commit

BLACK = black
BLACK_OPTS = -t py37

FLAKE8 = flake8

CLEAN_FILES = build/ *_cache/ docs/_build/ dist/ .pytest_cache/ *.egg-info/

#################################################################################
# FUNCTIONS                                                                     #
#################################################################################

define python_black
    $(BLACK) $(BLACK_OPTS) $(PY_SRC)
endef

define cleanup
    $(FIND) -name "__pycache__" -type d -exec $(RM) -rf {} +
    $(FIND) -name "*.py[co]" -type f -exec $(RM) -rf {} +
    $(FIND) -name "*.so" -type f -exec $(RM) {} +
    $(FIND) ./spyns -name "*.html" -type f -exec $(RM) {} +
    -$(RM) -rf $(CLEAN_FILES)
endef

define lint
    $(FLAKE8) $(PY_SRC)
endef

define make_subdirectory
    mkdir -p "$@"
endef

define precommit_cmd
    $(PRECOMMIT) $(1)
endef

define run_setup_py
    $(PY) ./$(PY_SETUP) $(1)
endef

define update_conda_env
    $(CONDA) env update --file $(CONDA_ENV_FILE)
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Reformat Python code with black
beautify:
	$(call python_black)

## Build python project
build:
	$(call run_setup_py, $(PY_SETUP_BUILD))

## Compile Cython code
build-ext:
	USE_CYTHON=y $(call run_setup_py, $(PY_SETUP_BUILD_EXT))

## Remove temporary and build files
clean:
	$(call cleanup)

## Create/update conda-based virtual environment
conda:
	$(call update_conda_env)

## Setup development environment
dev: conda hooks

## Generate documentation
docs:
	$(call run_setup_py, $(PY_SETUP_DOCS))

## Install pre-commit hooks
hooks:
	$(call precommit_cmd, install)

## Lint using flake8
lint:
	$(call lint)

## Create source distribution for Python project
sdist:
	$(call run_setup_py, $(PY_SETUP_SDIST))

## Run unit tests
test:
	$(call run_setup_py, $(PY_SETUP_TEST))

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
