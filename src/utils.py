from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# Funções utilitárias de texto e colunas
# -----------------------------------------------------------------------------
def normalize_text(value):
    """Converte texto para lowercase, remove acentos, pontuação e espaços extras."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_column(columns, aliases):
    """Busca uma coluna por lista de nomes esperados, ignorando acentos e maiúsculas."""
    normalized_columns = {normalize_text(col): col for col in columns}
    for alias in aliases:
        key = normalize_text(alias)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def ensure_processed_dir(dir_processed: Path):
    """Cria a pasta de arquivos processados caso não exista."""
    dir_processed.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Diretório de saída confirmado: {dir_processed}")


# -----------------------------------------------------------------------------
# Carregamento e validação dos arquivos
# -----------------------------------------------------------------------------
def load_excel_file(file_path: Path, label: str) -> pd.DataFrame:
    """Carrega uma planilha Excel e valida a existência do arquivo."""
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de {label} não encontrado: {file_path}")

    print(f"[INFO] Carregando {label}: {file_path}")
    return pd.read_excel(file_path)


# -----------------------------------------------------------------------------
# Preparação dos dados do cliente e do catálogo mestre
# -----------------------------------------------------------------------------
def prepare_client_dataframe(df_cliente: pd.DataFrame) -> dict:
    """Padroniza o DataFrame do cliente para o processamento de automação."""
    print("[INFO] Preparando dados do cliente...")

    codigo_col_cliente = find_column(df_cliente.columns, ["codigo", "código", "Código", "Codigo", "cod", "cod_servico"])
    if codigo_col_cliente is None:
        codigo_col_cliente = "Código"
        df_cliente.insert(0, codigo_col_cliente, "")
    else:
        df_cliente[codigo_col_cliente] = ""

    descricao_col_cliente = find_column(
        df_cliente.columns,
        [
            "descricao dos servicos",
            "descrição dos serviços",
            "descricao",
            "descrição",
            "servico",
            "item",
            "descricao do servico",
        ],
    )
    if descricao_col_cliente is None:
        raise ValueError("Não foi possível localizar a coluna de descrição dos serviços no arquivo do cliente.")

    qtd_col_cliente = find_column(df_cliente.columns, ["quantidade", "qtde", "qtd", "quant"])
    unidade_col_cliente = find_column(df_cliente.columns, ["unidade", "unid", "umed", "um"])

    df_cliente = df_cliente.copy()
    df_cliente["_descricao_normalizada"] = df_cliente[descricao_col_cliente].map(normalize_text)

    # Agrupador/cabeçalho: linha sem quantidade e sem unidade preenchidas (colunas brutas, sem normalização).
    if qtd_col_cliente is not None:
        quantidade_vazia = df_cliente[qtd_col_cliente].fillna("").astype(str).str.strip().eq("")
    else:
        quantidade_vazia = pd.Series(True, index=df_cliente.index)

    if unidade_col_cliente is not None:
        unidade_vazia = df_cliente[unidade_col_cliente].fillna("").astype(str).str.strip().eq("")
    else:
        unidade_vazia = pd.Series(True, index=df_cliente.index)

    df_cliente["_is_agrupador"] = quantidade_vazia & unidade_vazia

    return {
        "df": df_cliente,
        "codigo_col": codigo_col_cliente,
        "descricao_col": descricao_col_cliente,
        "qtd_col": qtd_col_cliente,
        "unidade_col": unidade_col_cliente,
    }


def prepare_master_dataframe(df_master: pd.DataFrame) -> dict:
    """Padroniza o DataFrame do catálogo mestre CPU para comparação por similaridade."""
    print("[INFO] Preparando base mestre CPU...")

    codigo_col_master = find_column(df_master.columns, ["codigo", "Código", "código", "Codigo", "cod", "cod_servico", "codigo_do_servico"])
    if codigo_col_master is None:
        raise ValueError("Não foi possível localizar a coluna de código no catálogo mestre CPU.")

    descricao_col_master = find_column(
        df_master.columns,
        [
            "descricao completa",
            "descrição completa",
            "descricao",
            "descrição",
            "descricao do servico",
            "descrição do serviço",
            "servico",
        ],
    )
    if descricao_col_master is None:
        raise ValueError("Não foi possível localizar a coluna de descrição completa no catálogo mestre CPU.")

    df_master = df_master.copy()
    df_master["_descricao_normalizada"] = df_master[descricao_col_master].map(normalize_text)
    df_master = df_master[df_master["_descricao_normalizada"].str.len() > 0].copy()

    return {
        "df": df_master,
        "codigo_col": codigo_col_master,
        "descricao_col": descricao_col_master,
    }


# -----------------------------------------------------------------------------
# Classificação de similaridade
# -----------------------------------------------------------------------------
def classify_similarity(score: float) -> str:
    """Define o status de automação pela taxa de similaridade."""
    if score >= 0.75    :
        return "Alto Grau de Confiança"
    if score >= 0.50:
        return "Revisão Recomendada"
    return "Baixa Similaridade"


# -----------------------------------------------------------------------------
# Dicionário de sinônimos e similaridade ponderada
# -----------------------------------------------------------------------------
def carregar_dicionario_sinonimos(caminho_dicionario: Path) -> dict:
    """Carrega o dicionário de sinônimos e mapeia cada palavra normalizada ao termo canônico do seu grupo."""
    if not caminho_dicionario.exists():
        print(f"[AVISO] Dicionário de sinônimos não encontrado: {caminho_dicionario}. Pesos de sinônimos desativados.")
        return {}

    df_sinonimos = pd.read_excel(caminho_dicionario)
    coluna_grupos = df_sinonimos.columns[0]

    mapa_sinonimos: dict = {}
    for grupo in df_sinonimos[coluna_grupos].dropna():
        palavras = [normalize_text(palavra) for palavra in str(grupo).split(",")]
        palavras = [palavra for palavra in palavras if palavra]
        if not palavras:
            continue

        # A primeira palavra do grupo vira o termo canônico usado para unificar os sinônimos.
        termo_canonico = palavras[0]
        for palavra in palavras:
            mapa_sinonimos[palavra] = termo_canonico

    return mapa_sinonimos


def aplicar_dicionario_sinonimos(texto_normalizado: str, mapa_sinonimos: dict) -> str:
    """Substitui cada palavra do texto pelo termo canônico do seu grupo de sinônimos, quando existir."""
    if not texto_normalizado or not mapa_sinonimos:
        return texto_normalizado

    palavras_canonicas = [mapa_sinonimos.get(palavra, palavra) for palavra in texto_normalizado.split(" ")]
    return " ".join(palavras_canonicas)


def calcular_matriz_similaridade_ponderada(
    descricoes_origem: pd.Series,
    descricoes_master: pd.Series,
    mapa_sinonimos: dict,
    peso_sinonimos: float = 0.6,
    peso_caracteres: float = 0.4,
) -> np.ndarray:
 
    descricoes_origem_canonicas = descricoes_origem.map(lambda texto: aplicar_dicionario_sinonimos(texto, mapa_sinonimos))
    descricoes_master_canonicas = descricoes_master.map(lambda texto: aplicar_dicionario_sinonimos(texto, mapa_sinonimos))

    vectorizer_caracteres = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    matriz_master_caracteres = vectorizer_caracteres.fit_transform(descricoes_master)
    matriz_origem_caracteres = vectorizer_caracteres.transform(descricoes_origem)
    similaridade_caracteres = cosine_similarity(matriz_origem_caracteres, matriz_master_caracteres)

    vectorizer_sinonimos = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False)
    matriz_master_sinonimos = vectorizer_sinonimos.fit_transform(descricoes_master_canonicas)
    matriz_origem_sinonimos = vectorizer_sinonimos.transform(descricoes_origem_canonicas)
    similaridade_sinonimos = cosine_similarity(matriz_origem_sinonimos, matriz_master_sinonimos)

    return (peso_sinonimos * similaridade_sinonimos) + (peso_caracteres * similaridade_caracteres)


# -----------------------------------------------------------------------------
# Falsos amigos: penalização de termos parecidos na escrita mas com sentidos diferentes
# -----------------------------------------------------------------------------
def carregar_falsos_amigos(caminho_falsos_amigos: Path) -> dict:
    """Carrega a planilha de falsos amigos e mapeia cada par de termos ao peso de penalização já atribuído nela."""
    if not caminho_falsos_amigos.exists():
        print(f"[AVISO] Arquivo de falsos amigos não encontrado: {caminho_falsos_amigos}. Penalização desativada.")
        return {}

    df_falsos_amigos = pd.read_excel(caminho_falsos_amigos)

    mapa_falsos_amigos: dict = {}
    for _, linha in df_falsos_amigos.iterrows():
        termo1 = normalize_text(linha.get("Termo 1"))
        termo2 = normalize_text(linha.get("Termo 2"))
        peso = linha.get("Similaridade")
        if not termo1 or not termo2 or pd.isna(peso):
            continue
        mapa_falsos_amigos[frozenset((termo1, termo2))] = float(peso)

    return mapa_falsos_amigos


def aplicar_penalidade_falsos_amigos(
    descricao_origem_normalizada: str,
    descricao_master_normalizada: str,
    mapa_falsos_amigos: dict,
    score: float,
) -> float:
    """Reduz o score quando a descrição de origem e a do master encontrada contêm um par de termos marcado
    como falso amigo (ex.: 'rede' x 'parede'), usando o peso já atribuído para essa dupla na planilha."""
    if not mapa_falsos_amigos:
        return score

    palavras_origem = set(descricao_origem_normalizada.split())
    palavras_master = set(descricao_master_normalizada.split())

    penalidade_maxima = 0.0
    for par, peso in mapa_falsos_amigos.items():
        termo1, termo2 = tuple(par)
        termos_cruzados = (
            (termo1 in palavras_origem and termo2 in palavras_master)
            or (termo2 in palavras_origem and termo1 in palavras_master)
        )
        if termos_cruzados:
            penalidade_maxima = max(penalidade_maxima, peso)

    if penalidade_maxima <= 0:
        return score

    return score * (1 - penalidade_maxima)


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
