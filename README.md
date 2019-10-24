# Requerimentos

- **make**
- **wget**
- **libc**

# Para instalar o conda localmente, com todas as dependências

`make install`

# Para compilar o código C

`make build`

# Para fazer tudo de uma vez

`make all`

# Caminho para comandos

Rode:

`make dotenv && source .env`

**Isso é necessário para rodar os scrips batch!**

Daí as seguintes variáveis de ambiente ficam disponíveis:

- python: `PYTHON`
- pip: `PIP`
- conda: `CONDA`

Todas as variáveis de ambientes geradas podem ser encontradas em:

`.env`

# Para limpar tudo

`make clean`
