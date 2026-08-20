from __future__ import annotations

from pathlib import Path


def selecionar_arquivo_cliente(diretorio_raw: Path) -> Path:
    """Lista os arquivos em 01_raw e retorna o caminho escolhido pelo usuário via input."""
    arquivos = sorted(p for p in diretorio_raw.iterdir() if p.is_file())
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {diretorio_raw}")

    print("\nArquivos disponíveis para processamento:")
    for i, arquivo in enumerate(arquivos, start=1):
        print(f"  arquivo {i}: {arquivo.name} - digite {i} para ler esse arquivo")

    while True:
        escolha = input("\nDigite o número do arquivo desejado: ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(arquivos):
            return arquivos[int(escolha) - 1]
        print("[ERRO] Opção inválida. Tente novamente.")


def gerar_caminho_saida_versionado(dir_processed: Path, nome_arquivo_cliente: str, nome_base: str) -> Path:
    """Cria a pasta do cliente em 03_processed e retorna o caminho de saída, incrementando _v2, _v3... se já existir."""
    pasta_cliente = dir_processed / nome_arquivo_cliente
    pasta_cliente.mkdir(parents=True, exist_ok=True)

    caminho = pasta_cliente / f"{nome_base}.xlsx"
    if not caminho.exists():
        return caminho

    versao = 2
    while True:
        candidato = pasta_cliente / f"{nome_base}_v{versao}.xlsx"
        if not candidato.exists():
            return candidato
        versao += 1
