from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    selecionar_arquivo_cliente,
    gerar_caminho_saida_versionado,
    ensure_processed_dir,
    load_excel_file,
    prepare_client_dataframe,
    prepare_master_dataframe,
    classify_similarity,
    carregar_dicionario_sinonimos,
    calcular_matriz_similaridade_ponderada,
)

# -----------------------------------------------------------------------------
# Configuração dos caminhos do projeto
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DIR_RAW = BASE_DIR / "data" / "01_raw"
DIR_MASTER = BASE_DIR / "data" / "02_master"
DIR_PROCESSED = BASE_DIR / "data" / "03_processed"

ARQUIVO_CPU = DIR_MASTER / "lista_cod_main.xlsx"
ARQUIVO_DICIONARIO_SINONIMOS = DIR_MASTER / "dicionario_sinonimos" / "dicionario_sinonimo_1.0.xlsx"


# -----------------------------------------------------------------------------
# Mapeamento automático por similaridade ponderada (caracteres + sinônimos)
# -----------------------------------------------------------------------------
def process_similarity_mapping(
    df_cliente: pd.DataFrame,
    df_master: pd.DataFrame,
    cliente_meta: dict,
    master_meta: dict,
    mapa_sinonimos: dict,
):
    """Compara descrições do cliente com a base master usando similaridade ponderada por sinônimos."""
    print("[INFO] Executando comparação por similaridade ponderada...")

    cliente_desc = df_cliente.loc[~df_cliente["_is_agrupador"], "_descricao_normalizada"]
    master_desc = df_master["_descricao_normalizada"]

    if cliente_desc.empty:
        print("[INFO] Nenhuma linha de serviço válida para processar. Verificando agrupadores/cabeçalhos.")
        return df_cliente

    if master_desc.empty:
        raise ValueError("A base mestre CPU está vazia após a limpeza de descrições. Verifique a planilha de catálogo.")

    semelhancas = calcular_matriz_similaridade_ponderada(cliente_desc, master_desc, mapa_sinonimos)
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


def carregar_bases_master_lista_cod(dir_master: Path) -> pd.DataFrame:
    """Carrega e combina todos os arquivos 'lista_cod*' da pasta master em uma única base de comparação."""
    arquivos_master = sorted(
        p for p in dir_master.glob("lista_cod*")
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}
    )

    frames_master = []
    for arquivo in arquivos_master:
        df_master_arquivo = load_excel_file(arquivo, f"CPU ({arquivo.name})")
        master_meta_arquivo = prepare_master_dataframe(df_master_arquivo)
        df_padronizado = master_meta_arquivo["df"][["_descricao_normalizada"]].copy()
        df_padronizado["_codigo"] = master_meta_arquivo["df"][master_meta_arquivo["codigo_col"]]
        df_padronizado["_descricao"] = master_meta_arquivo["df"][master_meta_arquivo["descricao_col"]]
        frames_master.append(df_padronizado)

    if not frames_master:
        return pd.DataFrame(columns=["_descricao_normalizada", "_codigo", "_descricao"])

    return pd.concat(frames_master, ignore_index=True)


def process_aux_similarity_mapping(df_cliente: pd.DataFrame, cliente_meta: dict, mapa_sinonimos: dict) -> pd.DataFrame:
    """Reavalia itens com baixa similaridade/revisão recomendada buscando a melhor similaridade entre
    todos os arquivos 'lista_cod*' da pasta master de uma só vez (sem repassar arquivo por arquivo)."""
    print("[INFO] Executando busca complementar combinando todos os arquivos lista_cod* da pasta master...")

    mask_revisao = df_cliente["Status_Automacao"].isin(["Baixa Similaridade", "Revisão Recomendada"])
    if not mask_revisao.any():
        return df_cliente

    # Remove o código sugerido pela base principal para os itens que serão reavaliados.
    df_cliente.loc[mask_revisao, cliente_meta["codigo_col"]] = ""

    df_master_combinado = carregar_bases_master_lista_cod(DIR_MASTER)
    master_desc = df_master_combinado["_descricao_normalizada"]
    cliente_desc = df_cliente.loc[mask_revisao, "_descricao_normalizada"]

    if master_desc.empty:
        print(f"[AVISO] Nenhum arquivo lista_cod* encontrado em: {DIR_MASTER}. Etapa de busca complementar ignorada.")
        return df_cliente

    if cliente_desc.empty:
        return df_cliente

    semelhancas = calcular_matriz_similaridade_ponderada(cliente_desc, master_desc, mapa_sinonimos)
    melhor_idx = semelhancas.argmax(axis=1)
    melhor_score = semelhancas[np.arange(len(semelhancas)), melhor_idx]

    for idx_pos, pos_original in enumerate(df_cliente.index[mask_revisao]):
        score = float(melhor_score[idx_pos])
        melhor_master_idx = int(melhor_idx[idx_pos])

        codigo_encontrado = df_master_combinado.iloc[melhor_master_idx]["_codigo"]
        descricao_encontrada = df_master_combinado.iloc[melhor_master_idx]["_descricao"]

        df_cliente.at[pos_original, "Similaridade_Pct"] = round(score * 100, 2)
        df_cliente.at[pos_original, "Descrição_Mestre_Encontrada"] = descricao_encontrada

        if score >= 0.50:
            df_cliente.at[pos_original, cliente_meta["codigo_col"]] = codigo_encontrado
            df_cliente.at[pos_original, "Status_Automacao"] = "Encontrado na Base Auxiliar"
        else:
            df_cliente.at[pos_original, cliente_meta["codigo_col"]] = ""
            df_cliente.at[pos_original, "Status_Automacao"] = "Não Encontrado - Baixa Similaridade"

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
        "Encontrado na Base Auxiliar",
        "Não Encontrado - Baixa Similaridade",
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
        "Encontrado na Base Auxiliar",
        "Não Encontrado - Baixa Similaridade",
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
    ensure_processed_dir(DIR_PROCESSED)
    mapa_sinonimos = carregar_dicionario_sinonimos(ARQUIVO_DICIONARIO_SINONIMOS)

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

        df_cliente = process_similarity_mapping(df_cliente, df_master, cliente_meta, master_meta, mapa_sinonimos)
        df_cliente = process_aux_similarity_mapping(df_cliente, cliente_meta, mapa_sinonimos)

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
