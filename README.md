# Requerimentos

- **make**
- **wget**
- **gcc**
- **libc**

# Para instalar o conda localmente, com todas as dependências

`make install`

# Caminho para alguns comandos

- python: `.conda_env/bin/python`
- pip: `.conda_env/bin/pip`
- conda: `.conda_env/bin/conda`

## Alternaivamente rode

`make dotenv && source .env`

Daí as seguintes variáveis de ambiente ficam disponíveis:

- python: `PYTHON`
- pip: `PIP`
- conda: `CONDA`

Todas as variáveis de ambientes geradas podem ser encontradas em:

`.env`

# Para compilar o código C

`make build`

# Para fazer tudo de uma vez

`make all`

# Para limpar tudo

`make clean`
