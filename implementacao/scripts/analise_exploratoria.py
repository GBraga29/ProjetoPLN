# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome: 
# RA:
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

# Constantes

CATEGORY_MAP = {
    -1: "Desconhecido",
    0: "Acórdão",
    1: "ARE",
    2: "Despacho",
    3: "RE",
    4: "Sentença",
}

CATEGORY_ORDER = list(CATEGORY_MAP.values())
COLORS = sns.color_palette("Set2", len(CATEGORY_MAP))
STOP_WORDS_PT = set(stopwords.words("portuguese"))


# Carregamento

def carregar_dados(path_train, path_test, path_submission):
    train_df = pd.read_csv(path_train, encoding='utf-8')
    test_df = pd.read_csv(path_test, encoding='utf-8')
    sample_submission = pd.read_csv(path_submission, encoding='utf-8')

    print(f"Treinamento : {train_df.shape}")
    print(f"Teste       : {test_df.shape}")
    print(f"Submissão   : {sample_submission.shape}")

    return train_df, test_df, sample_submission

def filtrar_categorias_validas(train_df):
    """Remove linhas com categorias fora de CATEGORY_MAP e adiciona coluna de nome."""
    df = train_df[train_df["Category"].isin(CATEGORY_MAP)].copy()
    df["Category_name"] = df["Category"].map(CATEGORY_MAP)
    return df


# Inspeção geral

def exibir_estrutura(train_df, test_df):
    # Mostra head, dtypes e info dos conjuntos.
    print("=== Primeiras linhas — treino ===")
    display(train_df.head())

    print("\n=== Informações gerais — treino ===")
    train_df.info()

    print("\n=== Primeiras linhas — teste ===")
    display(test_df.head())


def verificar_qualidade(train_df, test_df):
    # Exibe valores ausentes, duplicatas e contagem de categorias.
    print("=== Valores ausentes — treino ===")
    print(train_df.isnull().sum())

    print("\n=== Valores ausentes — teste ===")
    print(test_df.isnull().sum())

    dup_treino = train_df.duplicated(subset="Body").sum()
    dup_teste = test_df.duplicated(subset="Body").sum()
    print(f"\nBody duplicado no treino : {dup_treino}")
    print(f"Body duplicado no teste  : {dup_teste}")

    print("\n=== Categorias no treino ===")
    print(train_df["Category"].value_counts().sort_index())


# Distribuição de classes

def plotar_distribuicao_classes(train_valid, save_path="figs/distribuicao_classes.png"):
    # Barra + pizza com distribuição das categorias."""
    class_counts = (
        train_valid["Category_name"].value_counts().reindex(CATEGORY_ORDER)
    )
    class_pct = (class_counts / class_counts.sum() * 100).round(2)

    dist_df = pd.DataFrame({"Contagem": class_counts, "Percentual (%)": class_pct})
    print("=== Distribuição das classes ===")
    display(dist_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(class_counts.index, class_counts.values, color=COLORS, edgecolor="black")
    axes[0].set_title("Distribuição das Classes (contagem absoluta)")
    axes[0].set_xlabel("Categoria")
    axes[0].set_ylabel("Número de documentos")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        colors=COLORS,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Distribuição das Classes (percentual)")

    plt.suptitle(
        "Base de Dados - Conjunto de Treinamento",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nTotal de amostras válidas: {len(train_valid):,}")
    print(
        f"Razão desbalanceamento (maior/menor): "
        f"{class_counts.max() / class_counts.min():.2f}x"
    )

    return class_counts, class_pct


# Comprimento dos textos

def adicionar_metricas_comprimento(df):
    # Adiciona colunas num_chars, num_words e num_sentences ao DataFrame.
    df = df.copy()
    df["num_chars"] = df["Body"].str.len()
    df["num_words"] = df["Body"].str.split().str.len()
    df["num_sentences"] = df["Body"].str.count(r"[.!?]+")
    return df


def plotar_comprimento_textos(train_valid, save_path="figs/comprimento_textos.png"):
    # Estatísticas descritivas + boxplot + histograma de comprimento.
    print("=== Estatísticas de comprimento dos textos ===")
    display(
        train_valid[["num_chars", "num_words", "num_sentences"]].describe().round(1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(
        data=train_valid,
        x="Category_name",
        y="num_words",
        order=CATEGORY_ORDER,
        palette="Set2",
        ax=axes[0],
    )
    axes[0].set_title("Número de Palavras por Categoria")
    axes[0].set_xlabel("Categoria")
    axes[0].set_ylabel("Número de palavras")

    mediana = train_valid["num_words"].median()
    media = train_valid["num_words"].mean()
    axes[1].hist(train_valid["num_words"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(mediana, color="red", linestyle="--", label=f"Mediana: {mediana:.0f}")
    axes[1].axvline(media, color="orange", linestyle="--", label=f"Média: {media:.0f}")
    axes[1].set_title("Distribuição do Número de Palavras")
    axes[1].set_xlabel("Número de palavras")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== Média de palavras por categoria ===")
    display(
        train_valid.groupby("Category_name")[["num_chars", "num_words", "num_sentences"]]
        .mean()
        .round(1)
        .reindex(CATEGORY_ORDER)
    )


# Vocabulário e frequência de termos

def tokenizar_simples(texto):
    # Tokenização básica: lowercase, apenas letras ≥3 chars, sem stopwords.
    tokens = re.findall(r"\b[a-záéíóúâêîôûãõàèìòùç]{3,}\b", str(texto).lower())
    return [t for t in tokens if t not in STOP_WORDS_PT]


def calcular_frequencia_global(train_valid):
    # Retorna Counter com todos os tokens do corpus de treinamento.
    todos_tokens = []
    for texto in train_valid["Body"]:
        todos_tokens.extend(tokenizar_simples(texto))
    freq = Counter(todos_tokens)
    print(f"Vocabulário (sem stopwords): {len(freq):,} termos únicos")
    print(f"Total de tokens            : {len(todos_tokens):,}")
    return freq


def plotar_top_termos_global(freq_global, n=20, save_path="figs/top20_termos.png"):
    # Gráfico de barras horizontais com os N termos mais frequentes.
    top_n = freq_global.most_common(n)
    termos, freqs = zip(*top_n)

    plt.figure(figsize=(12, 5))
    bars = plt.barh(termos[::-1], freqs[::-1], color=sns.color_palette("Blues_r", n))
    plt.title(f"Top {n} Termos Mais Frequentes no Corpus (sem stopwords)")
    plt.xlabel("Frequência")
    for bar, freq in zip(bars, freqs[::-1]):
        plt.text(
            bar.get_width() + 200,
            bar.get_y() + bar.get_height() / 2,
            f"{freq:,}",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plotar_top_termos_por_categoria(train_valid, n=10, save_path="figs/top10_termos_categoria.png"):
    # Grid com os N termos mais frequentes de cada categoria.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (cat_id, cat_name) in enumerate(CATEGORY_MAP.items()):
        subset = train_valid[train_valid["Category"] == cat_id]
        tokens_cat = []
        for texto in subset["Body"]:
            tokens_cat.extend(tokenizar_simples(texto))

        top_n = Counter(tokens_cat).most_common(n)
        if not top_n:
            continue

        termos_cat, freqs_cat = zip(*top_n)
        axes[idx].barh(termos_cat[::-1], freqs_cat[::-1], color=COLORS[idx], edgecolor="white")
        axes[idx].set_title(f"{cat_name} (n={len(subset):,})", fontweight="bold")
        axes[idx].set_xlabel("Frequência")

    axes[-1].axis("off")
    plt.suptitle(f"Top {n} Termos por Categoria (sem stopwords)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Exemplos por categoria

def exibir_exemplos_por_categoria(train_valid, n_chars=400):
    # Imprime um trecho de texto de exemplo para cada categoria.
    print("=== Exemplos de textos por categoria ===\n")
    for cat_id, cat_name in CATEGORY_MAP.items():
        subset = train_valid[train_valid["Category"] == cat_id]
        if len(subset) == 0:
            continue
        exemplo = subset["Body"].iloc[0]
        print(f"--- {cat_name} (Categoria {cat_id}) ---")
        print(str(exemplo)[:n_chars].strip())
        print("...\n")


# Ruído de OCR

def calcular_metricas_ruido(texto):
    # Retorna proporções de caracteres especiais, dígitos e maiúsculas.
    texto = str(texto)
    n = len(texto)
    if n == 0:
        return 0.0, 0.0, 0.0
    prop_especiais = len(
        re.findall(r"[^\w\sáéíóúâêîôûãõàèìòùç.,;:!?()\-\'\"°%/]", texto, re.I)
    ) / n
    prop_digitos = len(re.findall(r"\d", texto)) / n
    prop_maiusculas = len(re.findall(r"[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]", texto)) / n
    return prop_especiais, prop_digitos, prop_maiusculas


def adicionar_metricas_ruido(df):
    # Adiciona colunas prop_especiais, prop_digitos e prop_maiusculas ao DataFrame.
    df = df.copy()
    metricas = df["Body"].apply(calcular_metricas_ruido)
    df[["prop_especiais", "prop_digitos", "prop_maiusculas"]] = pd.DataFrame(
        metricas.tolist(), index=df.index
    )
    return df


def plotar_ruido_textual(train_valid, save_path="figs/ruido_textual.png"):
    # Tabela resumo + boxplots de métricas de ruído por categoria.
    print("=== Métricas de ruído textual por categoria ===")
    display(
        train_valid.groupby("Category_name")[
            ["prop_especiais", "prop_digitos", "prop_maiusculas"]
        ]
        .mean()
        .round(4)
        .reindex(CATEGORY_ORDER)
    )

    metricas_cols = ["prop_especiais", "prop_digitos", "prop_maiusculas"]
    titulos = [
        "Proporção de Caracteres Especiais",
        "Proporção de Dígitos",
        "Proporção de Maiúsculas",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, titulo in zip(axes, metricas_cols, titulos):
        sns.boxplot(
            data=train_valid,
            x="Category_name",
            y=col,
            order=CATEGORY_ORDER,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Categoria")
        ax.set_ylabel("")

    plt.suptitle(
        "Indicadores de Ruído Textual por Categoria", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Resumo final

def exibir_resumo(train_valid, test_df, freq_global, class_counts, class_pct):
    sep = "=" * 60
    print(sep)
    print("         RESUMO DA ANÁLISE EXPLORATÓRIA")
    print(sep)
    print(f"Amostras de treinamento (válidas) : {len(train_valid):,}")
    print(f"Amostras de teste                 : {len(test_df):,}")
    print(f"Número de classes                 : {len(CATEGORY_MAP)}")
    print(f"Tamanho do vocabulário            : {len(freq_global):,} termos únicos")
    print(f"Média de palavras por documento   : {train_valid['num_words'].mean():.0f}")
    print(f"Mediana de palavras por documento : {train_valid['num_words'].median():.0f}")
    print()
    print("Classes mais e menos representadas:")
    print(
        f"  Maior : {class_counts.idxmax()} "
        f"({class_counts.max():,} amostras, {class_pct.max():.1f}%)"
    )
    print(
        f"  Menor : {class_counts.idxmin()} "
        f"({class_counts.min():,} amostras, {class_pct.min():.1f}%)"
    )
    print()
    print("Observações para o pré-processamento:")
    print("  - Dataset desbalanceado: RE domina com ~56% dos dados.")
    print("  - Textos via OCR apresentam ruído (caracteres especiais).")
    print("  - Carimbos e assinaturas adicionam tokens irrelevantes.")
    print("  - Alto percentual de maiúsculas em cabeçalhos.")
    print("  - Limpeza e normalização são etapas críticas.")
    print(sep)
    
    