# Automação de códigos de serviços

Programa para associar descrições de serviços de arquivos de orçamento aos códigos existentes nas planilhas master. A comparação usa TF-IDF, dicionário de sinônimos e uma lista de falsos amigos para reduzir associações causadas apenas por palavras visualmente parecidas.

## Estrutura de diretórios

```text
Projeto_Automacao_Napoli/
├── main.py
├── requirements.txt
├── src/
│   ├── utils.py
│   └── palavras_diferentes
└── data/
	├── 01_raw/
	│   └── arquivos de orçamento do cliente
	├── 02_master/
	│   ├── lista_cod_main.xlsx
	│   ├── lista_cod_aux.xlsx
	│   ├── lista_cod_aux_2.xlsx
	│   ├── falsos_amigos_candidatos.xlsx
	│   └── dicionario_sinonimos/
	│       └── dicionario_sinonimo_1.0.xlsx
	└── 03_processed/
		└── uma pasta para cada arquivo processado
```

## Pré-requisitos

Instale as dependências do projeto no ambiente virtual:

```bash
pip install -r requirements.txt
```

As planilhas são lidas e gravadas com `pandas`, utilizando `openpyxl` para os arquivos Excel. O cálculo de similaridade usa `scikit-learn`.

## Arquivos de entrada

### Arquivos do cliente

Coloque os arquivos de orçamento em `data/01_raw/`. O programa procura automaticamente uma coluna de descrição usando nomes como:

- `descricao`
- `descrição`
- `descricao dos servicos`
- `descrição dos serviços`
- `servico`
- `item`

Se não existir uma coluna de código, o programa cria a coluna `Código`. Se ela existir, seus valores são limpos antes do novo mapeamento.

As colunas de quantidade e unidade são opcionais. Quando existem, elas são usadas apenas para identificar agrupadores/cabeçalhos: uma linha sem quantidade e sem unidade não recebe código nem passa pela busca de similaridade.

### Catálogo principal

`data/02_master/lista_cod_main.xlsx` é a primeira base pesquisada. Ela deve conter uma coluna de código, reconhecida por nomes como `codigo`, `cod` ou `cod_servico`, e uma coluna de descrição, reconhecida por nomes como `descricao`, `descricao completa` ou `descricao do servico`.

### Catálogos complementares

Depois da primeira pesquisa, os itens classificados como `Baixa Similaridade` ou `Revisão Recomendada` são pesquisados novamente. Nessa etapa, o programa carrega de uma vez todos os arquivos da pasta `data/02_master/` cujo nome começa com `lista_cod` e cuja extensão é `.xlsx` ou `.xls`.

Exemplos incluídos no conjunto:

- `lista_cod_main.xlsx`
- `lista_cod_aux.xlsx`
- `lista_cod_aux_2.xlsx`

Todos os registros são combinados em uma única base e o melhor resultado é escolhido entre todos eles. O programa não pesquisa os arquivos auxiliares em uma sequência que pare no primeiro resultado.

### Dicionário de sinônimos

O arquivo `data/02_master/dicionario_sinonimos/dicionario_sinonimo_1.0.xlsx` contém grupos de palavras semelhantes, normalmente separados por vírgulas em uma coluna. Por exemplo:

```text
instalacao, instalacoes, instalado, instalar
```

A primeira palavra do grupo é usada como termo canônico. Assim, todas as palavras do grupo passam a ser comparadas como o mesmo conceito.

### Falsos amigos

O arquivo `data/02_master/falsos_amigos_candidatos.xlsx` contém pares de palavras parecidas na escrita, mas que podem representar conceitos diferentes. As colunas usadas pelo programa são:

- `Termo 1`
- `Termo 2`
- `Similaridade`

O valor de `Similaridade` é usado como intensidade da penalidade. Por exemplo, se uma descrição do cliente contém `rede` e a descrição master encontrada contém `parede`, o par é localizado na planilha e o score é reduzido.

## Execução passo a passo

Execute:

```bash
python main.py
```

O fluxo é:

1. Define as pastas `data/01_raw`, `data/02_master` e `data/03_processed`.
2. Cria `data/03_processed` caso ela ainda não exista.
3. Carrega o dicionário de sinônimos e a lista de falsos amigos.
4. Lista os arquivos disponíveis em `data/01_raw`.
5. Exibe um menu numérico, como:

   ```text
   arquivo 1: cliente_a.xlsx - digite 1 para ler esse arquivo
   arquivo 2: cliente_b.xlsx - digite 2 para ler esse arquivo
   ```

6. O usuário informa o número correspondente ao arquivo que deseja processar. Essa é a seleção/switch de opções de arquivo do programa.
7. Cria a pasta de saída específica para o arquivo escolhido.
8. Carrega o arquivo do cliente e `lista_cod_main.xlsx`.
9. Localiza e padroniza as colunas de código e descrição.
10. Normaliza os textos para comparação: converte para minúsculas, remove acentos, pontuação e espaços extras.
11. Marca como `Agrupador / Cabeçalho` as linhas sem quantidade e unidade.
12. Calcula a similaridade entre as demais descrições e o catálogo principal.
13. Aplica a penalidade de falsos amigos ao resultado escolhido.
14. Para os itens com baixa similaridade ou revisão recomendada, pesquisa todos os arquivos `lista_cod*` combinados.
15. Aplica novamente a penalidade de falsos amigos na segunda busca.
16. Preenche o código somente quando o score final atende ao limite exigido.
17. Exporta o orçamento mapeado e o relatório executivo.

## Como o cálculo de similaridade funciona

Para cada descrição do cliente, são calculados dois sinais:

### 1. Similaridade por caracteres

O programa usa `TfidfVectorizer` com:

```python
analyzer="char_wb"
ngram_range=(3, 5)
```

Isso divide as palavras em grupos de 3 a 5 caracteres e calcula a similaridade cosseno entre a descrição do cliente e cada descrição master. Essa etapa ajuda a reconhecer pequenas variações de escrita, plural e flexões.

### 2. Similaridade por palavras e sinônimos

Antes da comparação por palavras, cada termo é substituído pelo termo canônico do dicionário de sinônimos. Essa etapa usa TF-IDF por palavras e pares de palavras:

```python
analyzer="word"
ngram_range=(1, 2)
```

Palavras que estão no mesmo grupo do dicionário recebem o mesmo tratamento semântico. Por exemplo, `instalacao` e `instalacoes` podem contribuir para a mesma correspondência.

### 3. Pesos combinados

O score inicial é calculado pela fórmula:

```text
score_inicial = (0,60 × similaridade_de_sinonimos)
			   + (0,40 × similaridade_de_caracteres)
```

Portanto, o significado indicado pelo dicionário tem mais influência que a semelhança puramente visual.

Uma palavra apenas parecida na grafia não é considerada sinônimo automaticamente. Por exemplo, `rede` e `parede` só são tratadas como conceitos equivalentes se estiverem no mesmo grupo do dicionário, o que não deve ocorrer nesse caso.

## Penalidade de falsos amigos

Depois que o melhor candidato é encontrado, o programa compara as palavras da descrição original com as palavras da descrição master selecionada.

Se encontrar um par cruzado na lista de falsos amigos, aplica:

```text
score_final = score_inicial × (1 - peso_do_par)
```

Exemplo:

```text
Descrição do cliente: fornecimento de rede eletrica
Descrição master:     retirada de parede em alvenaria
Score inicial:        0,70
Peso do par:          0,535
Score final:          0,3255
```

O maior peso encontrado entre os pares conflitantes é usado uma única vez. Isso evita reduzir repetidamente o score quando várias palavras da mesma descrição produzem alertas.

Essa penalidade é aplicada tanto na busca de `lista_cod_main.xlsx` quanto na busca complementar dos arquivos `lista_cod*`.

## Classificação dos resultados

O score considerado abaixo já inclui a penalidade de falsos amigos:

| Score final | Status | Comportamento |
|---:|---|---|
| `>= 0,75` | `Alto Grau de Confiança` | Preenche o código encontrado na busca principal. |
| `>= 0,50` e `< 0,75` | `Revisão Recomendada` | Entra na busca complementar; se continuar acima de 50%, recebe código da base combinada. |
| `< 0,50` | `Baixa Similaridade` | Entra na busca complementar; se continuar abaixo de 50%, o código permanece vazio. |
| `>= 0,50` na busca complementar | `Encontrado na Base Auxiliar` | Preenche o código encontrado entre todos os arquivos `lista_cod*`. |
| `< 0,50` na busca complementar | `Não Encontrado - Baixa Similaridade` | Não preenche o campo de código. |
| Linha sem quantidade e unidade | `Agrupador / Cabeçalho` | Não executa busca e mantém o código vazio. |

## Saída e controle de versão

Para cada arquivo selecionado, o programa cria uma pasta com o mesmo nome-base dentro de `data/03_processed/`.

Exemplo para `orcamento_cliente.xlsx`:

```text
data/03_processed/
└── orcamento_cliente/
	├── orcamento_cliente_orcamento_processado.xlsx
	├── orcamento_cliente_orcamento_processado_v2.xlsx
	└── orcamento_cliente_orcamento_processado_v3.xlsx
```

O controle funciona assim:

1. Se o primeiro arquivo de saída não existir, ele recebe o nome base.
2. Se já existir, o programa tenta `_v2`.
3. Se `_v2` também existir, tenta `_v3`, e assim por diante.
4. Nenhum resultado anterior é sobrescrito.

Cada arquivo exportado contém duas abas:

- `Orcamento_Mapeado`: dados originais do cliente, código atribuído, descrição master encontrada, similaridade final e status.
- `Relatorio_Executivo`: quantidade e percentual de linhas por status.

Além dos arquivos gerados por `main.py`, o diagnóstico [src/palavras_diferentes](src/palavras_diferentes) gera ou atualiza `data/03_processed/falsos_amigos_candidatos.xlsx`. Essa planilha deve ser revisada antes de ser usada como fonte de penalidades e normalmente é copiada para `data/02_master/`.

## Componentes principais

### `main.py`

Coordena o fluxo da aplicação:

- seleção do arquivo do cliente;
- carregamento das bases;
- busca principal;
- busca complementar;
- aplicação das penalidades;
- montagem do relatório;
- exportação versionada.

### `src/utils.py`

Concentra as funções reutilizáveis:

- normalização de texto;
- identificação de colunas;
- leitura de planilhas;
- preparação de cliente e catálogo;
- carregamento do dicionário de sinônimos;
- cálculo da matriz de similaridade ponderada;
- carregamento da lista de falsos amigos;
- aplicação das penalidades;
- seleção interativa e versionamento do caminho de saída.

## Limitações e cuidados

- Os nomes das colunas precisam ser compatíveis com os aliases reconhecidos pelo programa.
- O dicionário de sinônimos deve conter apenas palavras que realmente representam o mesmo conceito.
- A planilha de falsos amigos é uma lista de candidatos; os pares devem ser revisados para evitar penalizações indevidas.
- O código só é preenchido na busca complementar quando o score final permanece em pelo menos `0,50`.
- A seleção do arquivo é interativa e exige que o usuário informe um número válido no terminal.
