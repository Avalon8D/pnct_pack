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
GCC=$(CONDA_BIN)/x86_64-conda_cos6-linux-gnu-gcc

CONDA_FORGE_REQUIREMENTS=-c conda-forge gsl blas lapack gcc_impl_linux-64

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

CODE_PATH=./all_code
bin_path=$(CODE_PATH)/bin
file_base_name=general_clustering_wraper
file_path=$(CODE_PATH)/c_code
file=$(CODE_PATH)/c_code/general_clustering_wraper
CLUSTERING_LIB_SO=$(CODE_PATH)/$(file_base_name).so

compile:
	$(GCC) -c -o $(CODE_PATH)/$(file_base_name).o \
	-fPIC -Wall -Wextra -Wpedantic -Wformat -D_GNU_SOURCE \
	-I$(file_path) -I$(CONDA_PATH)/include $(file).c \
	-B $(CONDA_PATH)/lib -lm -lgsl -lgslcblas -llapacke -Ofast \
	&& \
	$(GCC) -shared -o $(CLUSTERING_LIB_SO) \
	-Wall -Wextra -Wpedantic -Wformat \
	-I$(file_path) -I$(CONDA_PATH)/include $(CODE_PATH)/$(file_base_name).o \
	-B $(CONDA_PATH)/lib -lm -lgsl -lgslcblas -llapacke -Ofast

clean_compile:
	rm -f $(CODE_PATH)/*.o

build: compile clean_compile

dotenv:
	echo "export CONDA_PATH=./$(CONDA_PATH)
	export TMP_PATH=./$(TMP_PATH)
	export CONDA_SCRIPT=./$(CONDA_SCRIPT)
	export CONDA_BIN=./$(CONDA_BIN)
	export PIP=./$(PIP)
	export PYTHON=./$(PYTHON)
	export CONDA=./$(CONDA)
	export CODE_PATH=./$(CODE_PATH)
	export CLUSTERING_LIB_SO=./$(CLUSTERING_LIB_SO)
	export GCC=./$(GCC)" > .env

all: install build dotenv

clean: clean_compile clean_conda
	rm -f $(code_path)/*.so
	rm -f .env