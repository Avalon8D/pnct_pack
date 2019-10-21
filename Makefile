.PHONY: compile, clean_compile, build, download_conda, install_conda, requirements, all, clean, dotenv
.ONESHELL:

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

CONDA_PATH=.conda_env
TMP_PATH=.tmp
CONDA_SCRIPT=$(TMP_PATH)/miniconda_script.sh
CONDA_BIN=$(CONDA_PATH)/bin
PIP=$(CONDA_BIN)/pip
PYTHON=$(CONDA_BIN)/python
CONDA=$(CONDA_BIN)/conda
GCC=$(CONDA_BIN)/gcc

CONDA_FORGE_REQUIREMENTS=-c conda-forge gsl blas lapack

download_conda:
	if [ ! -f $(CONDA_SCRIPT) ]; then
		mkdir -p $(TMP_PATH)
		wget -O $(CONDA_SCRIPT) $(MINICONDA_URL)
		chmod +x $(CONDA_SCRIPT)
	fi

install_conda:
	$(CONDA_SCRIPT) -b -p /tmp/$(CONDA_PATH) -f
	/tmp/$(CONDA) create -f -p $(CONDA_PATH) python==3.7 conda -y
	rm -rf /tmp/$(CONDA)

requirements: 
	$(PIP) install -r requirements.txt
	$(CONDA) install $(CONDA_FORGE_REQUIREMENTS) -y

install: download_conda install_conda requirements

clean_conda:
	rm -rf $(CONDA_PATH)
	rm -rf $(TMP_PATH)
	rm -rf /tmp/$(CONDA_PATH)

code_path=all_code
bin_path=$(code_path)/bin
file_base_name=general_clustering_wraper
file_path=$(code_path)/c_code
file=$(code_path)/c_code/general_clustering_wraper

compile:
	gcc -c -o $(code_path)/$(file_base_name).o \
	-fPIC -Wall -Wextra -Wpedantic -Wformat -D_GNU_SOURCE \
	-I$(file_path) -I$(CONDA_PATH)/include $(file).c \
	-lm -Ofast \
	&& gcc -shared -o $(code_path)/$(file_base_name).so \
	-Wall -Wextra -Wpedantic -Wformat \
	-I$(file_path) -I$(CONDA_PATH)/include $(code_path)/$(file_base_name).o \
	-lm -Ofast

clean_compile:
	rm -f $(code_path)/*.o

build: compile clean_compile

dotenv:
	echo "export CONDA_PATH=$(CONDA_PATH)
	export TMP_PATH=$(TMP_PATH)
	export CONDA_SCRIPT=$(CONDA_SCRIPT)
	export CONDA_BIN=$(CONDA_BIN)
	export PIP=$(PIP)
	export PYTHON=$(PYTHON)
	export CONDA=$(CONDA)
	export GCC=$(GCC)" > .env

all: install build dotenv

clean: clean_compile clean_conda
	rm -f $(code_path)/*.so
	rm .env