from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import selecionar_arquivo_cliente, gerar_caminho_saida_versionado

# -----------------------------------------------------------------------------
# Configuração dos caminhos do projeto
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DIR_RAW = BASE_DIR / "data" / "01_raw"
DIR_MASTER = BASE_DIR / "data" / "02_master"
DIR_PROCESSED = BASE_DIR / "data" / "03_processed"

ARQUIVO_CPU = DIR_MASTER / "listaCod.xlsx"

# -----------------------------------------------------------------------------
# Funções utilitárias
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


def ensure_processed_dir():
    """Cria a pasta de arquivos processados caso não exista."""
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Diretório de saída confirmado: {DIR_PROCESSED}")


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
def prepare_client_dataframe(df_cliente: pd.DataFrame) -> pd.DataFrame:
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

    if qtd_col_cliente is not None:
        df_cliente["_quantidade_limpa"] = df_cliente[qtd_col_cliente].fillna("").astype(str).str.strip()
    else:
        df_cliente["_quantidade_limpa"] = ""

    if unidade_col_cliente is not None:
        df_cliente["_unidade_limpa"] = df_cliente[unidade_col_cliente].fillna("").astype(str).str.strip()
    else:
        df_cliente["_unidade_limpa"] = ""

    df_cliente["_is_agrupador"] = (
        df_cliente["_quantidade_limpa"].eq("") & df_cliente["_unidade_limpa"].eq("")
    )

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
# Mapeamento automático por similaridade TF-IDF
# -----------------------------------------------------------------------------
def classify_similarity(score: float) -> str:
    """Define o status de automação pela taxa de similaridade."""
    if score >= 0.68:
        return "Alto Grau de Confiança"
    if score >= 0.50:
        return "Revisão Recomendada"
    return "Baixa Similaridade"


def process_similarity_mapping(df_cliente: pd.DataFrame, df_master: pd.DataFrame, cliente_meta: dict, master_meta: dict):
    """Compara descrições do cliente com a base master usando TF-IDF com n-grams por caracteres."""
    print("[INFO] Executando comparação por similaridade TF-IDF...")

    cliente_desc = df_cliente.loc[~df_cliente["_is_agrupador"], "_descricao_normalizada"]
    master_desc = df_master["_descricao_normalizada"]

    if cliente_desc.empty:
        print("[INFO] Nenhuma linha de serviço válida para processar. Verificando agrupadores/cabeçalhos.")
        return df_cliente

    if master_desc.empty:
        raise ValueError("A base mestre CPU está vazia após a limpeza de descrições. Verifique a planilha de catálogo.")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    matriz_master = vectorizer.fit_transform(master_desc)
    matriz_cliente = vectorizer.transform(cliente_desc)

    semelhancas = cosine_similarity(matriz_cliente, matriz_master)
    melhor_idx = semelhancas.argmax(axis=1)
    melhor_score = semelhancas[np.arange(len(semelhancas)), melhor_idx]

    for idx_pos, pos_original in enumerate(df_cliente.index[df_cliente["_is_agrupador"] == False]):
        score = float(melhor_score[idx_pos])
        melhor_master_idx = int(melhor_idx[idx_pos])

        codigo_mestre = df_master.iloc[melhor_master_idx][master_meta["codigo_col"]]
        descricao_mestre = df_master.iloc[melhor_master_idx][master_meta["descricao_col"]]

        df_cliente.at[pos_original, cliente_meta["codigo_col"]] = codigo_mestre
        df_cliente.at[pos_original, "Descrição_Mestre_Encontrada"] = descricao_mestre
        df_cliente.at[pos_original, "Similaridade_Pct"] = round(score * 100, 2)
        df_cliente.at[pos_original, "Status_Automacao"] = classify_similarity(score)

    return df_cliente


def build_executive_report(df_cliente: pd.DataFrame) -> pd.DataFrame:
    """Cria a tabela resumida de métricas por status para a aba executiva."""
    total_itens = len(df_cliente)
    if total_itens == 0:
        return pd.DataFrame(
            {
                "Status": [
                    "Alto Grau de Confiança",
                    "Revisão Recomendada",
                    "Baixa Similaridade",
                    "Agrupador / Cabeçalho",
                ],
                "Quantidade_de_Itens": [0, 0, 0, 0],
                "%_do_Total": [0.0, 0.0, 0.0, 0.0],
            }
        )

    status_counts = df_cliente["Status_Automacao"].value_counts(dropna=False).rename_axis("Status").reset_index(name="Quantidade_de_Itens")
    status_counts["%_do_Total"] = (status_counts["Quantidade_de_Itens"] / total_itens * 100).round(2)

    status_order = [
        "Alto Grau de Confiança",
        "Revisão Recomendada",
        "Baixa Similaridade",
        "Agrupador / Cabeçalho",
    ]

    relatorio = status_counts[status_counts["Status"].isin(status_order)].copy()
    relatorio = relatorio.set_index("Status").reindex(status_order).reset_index().rename(columns={"index": "Status"})

    if relatorio.empty:
        relatorio = pd.DataFrame(
            {
                "Status": status_order,
                "Quantidade_de_Itens": [0, 0, 0, 0],
                "%_do_Total": [0.0, 0.0, 0.0, 0.0],
            }
        )

    for col in ["Quantidade_de_Itens", "%_do_Total"]:
        relatorio[col] = relatorio[col].fillna(0)

    relatorio["%_do_Total"] = relatorio["%_do_Total"].map(lambda x: f"{x:.2f}%")
    return relatorio


def print_dashboard(df_cliente: pd.DataFrame):
    """Exibe um resumo final no terminal em formato de dashboard."""
    total_itens = len(df_cliente)
    if total_itens == 0:
        print("\n[RESUMO FINAL]")
        print("  - Total de linhas processadas: 0")
        print("  - Alto Grau de Confiança: 0 (0.00%)")
        print("  - Revisão Recomendada: 0 (0.00%)")
        print("  - Baixa Similaridade: 0 (0.00%)")
        print("  - Agrupadores / Cabeçalhos: 0 (0.00%)")
        return

    status_counts = df_cliente["Status_Automacao"].value_counts(dropna=False)
    status_order = [
        "Alto Grau de Confiança",
        "Revisão Recomendada",
        "Baixa Similaridade",
        "Agrupador / Cabeçalho",
    ]
    for status in status_order:
        count = int(status_counts.get(status, 0))
        pct = (count / total_itens) * 100 if total_itens else 0
        print(f"  - {status}: {count} ({pct:.2f}%)")

    print(f"\n[INFO] Total de linhas processadas: {total_itens}")


def export_excel_with_report(df_cliente: pd.DataFrame, output_path: Path):
    """Exporta a planilha final em duas abas: orçamento e relatório executivo."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_cliente.to_excel(writer, sheet_name="Orcamento_Mapeado", index=False)

        relatorio = build_executive_report(df_cliente)
        relatorio.to_excel(writer, sheet_name="Relatorio_Executivo", index=False)

        print(f"\n[INFO] Arquivo Excel exportado com duas abas em: {output_path}")


# -----------------------------------------------------------------------------
# Execução principal
# -----------------------------------------------------------------------------
def main():
    """Fluxo principal da automação de mapeamento de códigos de serviços."""
    print("\n=== INÍCIO DA AUTOMATIZAÇÃO ===")
    ensure_processed_dir()

    try:
        arquivo_cliente = selecionar_arquivo_cliente(DIR_RAW)
    except FileNotFoundError as exc:
        print(f"\n[ERRO] {exc}\n")
        return

    nome_base = f"{arquivo_cliente.stem}_orcamento_processado"
    arquivo_saida = gerar_caminho_saida_versionado(DIR_PROCESSED, arquivo_cliente.stem, nome_base)

    try:
        df_cliente = load_excel_file(arquivo_cliente, "cliente")
        df_master = load_excel_file(ARQUIVO_CPU, "CPU")
    except FileNotFoundError as exc:
        print(f"\n[ERRO] {exc}\n")
        print("[DICA] Verifique se os arquivos existem nos diretórios esperados:")
        print(f"  - Cliente: {arquivo_cliente}")
        print(f"  - CPU: {ARQUIVO_CPU}")
        return

    try:
        cliente_meta = prepare_client_dataframe(df_cliente)
        master_meta = prepare_master_dataframe(df_master)

        df_cliente = cliente_meta["df"]
        df_master = master_meta["df"]

        df_cliente["Descrição_Mestre_Encontrada"] = ""
        df_cliente["Similaridade_Pct"] = 0.0
        df_cliente["Status_Automacao"] = ""

        for idx in df_cliente.index[df_cliente["_is_agrupador"]]:
            df_cliente.at[idx, "Descrição_Mestre_Encontrada"] = ""
            df_cliente.at[idx, "Similaridade_Pct"] = 0.0
            df_cliente.at[idx, "Status_Automacao"] = "Agrupador / Cabeçalho"

        df_cliente = process_similarity_mapping(df_cliente, df_master, cliente_meta, master_meta)

        # Garantindo que linhas agrupadoras permaneçam com código vazio e sem busca executada.
        for idx in df_cliente.index[df_cliente["_is_agrupador"]]:
            df_cliente.at[idx, cliente_meta["codigo_col"]] = ""
            df_cliente.at[idx, "Descrição_Mestre_Encontrada"] = ""
            df_cliente.at[idx, "Similaridade_Pct"] = 0.0
            df_cliente.at[idx, "Status_Automacao"] = "Agrupador / Cabeçalho"

        colunas_finais = [
            col for col in df_cliente.columns if not col.startswith("_")
        ] + [
            "Descrição_Mestre_Encontrada",
            "Similaridade_Pct",
            "Status_Automacao",
        ]

        # Ajusta a ordem das colunas para manter os campos principais e os de auditoria ao final.
        colunas_saida = []
        seen = set()
        for col in df_cliente.columns:
            if col in {"Descrição_Mestre_Encontrada", "Similaridade_Pct", "Status_Automacao"}:
                continue
            if col not in seen:
                colunas_saida.append(col)
                seen.add(col)

        for col in ["Descrição_Mestre_Encontrada", "Similaridade_Pct", "Status_Automacao"]:
            if col in df_cliente.columns:
                colunas_saida.append(col)

        df_cliente = df_cliente[colunas_saida]
        export_excel_with_report(df_cliente, arquivo_saida)

        print("\n[RESUMO FINAL]")
        print(f"  - Total de linhas processadas: {len(df_cliente)}")
        print_dashboard(df_cliente)
        print("\n=== FIM DA AUTOMATIZAÇÃO ===\n")

    except ValueError as exc:
        print(f"\n[ERRO] {exc}\n")
        return
    except Exception as exc:
        print(f"\n[ERRO INESPERADO] {type(exc).__name__}: {exc}\n")
        return


if __name__ == "__main__":
    main()
