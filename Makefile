.PHONY: compile, clean_compile, build, conda_install, requirements, clean

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

SHELL=bash
CONDA_PATH=.conda_env
TMP_PATH=.tmp
CONDA_SCRIPT=$(TMP_PATH)/miniconda_script.sh
CONDA_BIN=$(CONDA_PATH)/bin
PIP=$(CONDA_BIN)/pip
PYTHON=$(CONDA_BIN)/python
CONDA=$(CONDA_BIN)/conda

CONDA_REQUIREMENTS=-c conda-forge gsl blas lapack

download_conda:
	mkdir -p $(TMP_PATH)
	wget -O $(CONDA_SCRIPT) $(MINICONDA_URL)
	chmod +x $(CONDA_SCRIPT)

install_conda:
	$(CONDA_SCRIPT) -b -p $(CONDA_PATH) -f

requirements: 
	$(PIP) install -r requirements.txt
	$(CONDA) install $(CONDA_REQUIREMENTS) -y

install: download_conda install_conda requirements

clean_conda:
	rm -rf $(CONDA_PATH)
	rm -rf $(TMP_PATH)

code_path=all_code
bin_path=$(code_path)/bin
file_base_name=general_clustering_wraper
file_path=$(code_path)/c_code
file=$(code_path)/c_code/general_clustering_wraper

compile:
	PATH=${PATH}:$(CONDA_PATH) gcc -c -o $(code_path)/$(file_base_name).o \
	-fPIC -Wall -Wextra -Wpedantic -Wformat -D_GNU_SOURCE \
	-I$(file_path) $(file).c -lgsl -lm -llapacke -Ofast \
	&& gcc -shared -o $(code_path)/$(file_base_name).so \
	-Wall -Wextra -Wpedantic -Wformat \
	-I$(file_path) $(code_path)/$(file_base_name).o \
	-lgsl -lm -llapacke -Ofast

clean_compile:
	rm -f $(code_path)/*.o

build: compile clean_compile

all: install build

clean: clean_compile clean_conda
	rm -f $(code_path)/*.so