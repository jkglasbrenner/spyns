# Credit for ROOTDIR implementation:
# kenorb (https://stackoverflow.com/users/55075/kenorb),
# How to get current relative directory of your Makefile?,
# URL (version: 2017-05-23): https://stackoverflow.com/a/35698978
ROOTDIR				=	$(abspath $(patsubst %/,%,$(dir $(abspath $(lastword 	\
						$(MAKEFILE_LIST))))))

SHELL				=	/bin/sh

PROJNAME			=	spyns

RM					=	rm
COPY				=	cp
FIND				=	find

CONDA				=	conda
CONDA_ENV_FILE		=	environment.yml

PY					?=	python3
PY_SETUP			=	setup.py
PY_SETUP_DOCS		=	build_sphinx

CLEAN_FILES			=	build/													\
						*_cache/												\
						docs/_build/ 											\
						dist/													\
						.pytest_cache/											\
						*.egg-info/

define makefile_help
	@echo 'Makefile for the spyns project                                            '
	@echo '                                                                          '
	@echo 'Usage:                                                                    '
	@echo '   make help                           display this message (default)     '
	@echo '                                                                          '
	@echo '   make build                          build everything needed to install '
	@echo '   make clean                          remove temporary and build files   '
	@echo '   make develop                        install project in development mode'
	@echo '   make docs                           generate documentation             '
	@echo '   make env                            create conda venv and install deps '
	@echo '   make sdist                          create a source distribution       '
	@echo '   make test                           run unit tests                     '
	@echo '                                                                          '
endef

define build_cleanup
	-$(RM) -rf $(CLEAN_FILES)
endef

define pycache_cleanup
	$(FIND) -name "__pycache__" -type d -exec $(RM) -rf {} +
endef

define update_conda_env
	bash -lc "$(CONDA) env update --file $(CONDA_ENV_FILE)"
endef

define run_setup_py
	$(PY) ./$(PY_SETUP) $(1)
endef

help :
	$(call makefile_help)

build :
	$(call run_setup_py,build)

clean :
	$(call build_cleanup)
	$(call pycache_cleanup)

develop :
	$(call run_setup_py,develop)

docs :
	$(call run_setup_py,$(PY_SETUP_DOCS))

env :
	$(call update_conda_env)

sdist :
	$(call run_setup_py,sdist)

test :
	$(call run_setup_py,test)

.PHONY : help build clean develop docs env
