# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
# Nome: Eric Lopes, Guilherme Braga, Guilherme Saggion
# RA:   822873,    823161,          823159
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento.
#
# Sao implementados dois niveis de limpeza, com base em Martins (2024)
# e Araujo et al. (2020) para o dataset VICTOR:
#
#   limpeza_basica   — correcao de encoding (mojibake), extracao do
#                      wrapper JSON, normalizacao unicode, minusculas,
#                      remocao de URLs/e-mails/referencias de folhas.
#                      Aplicada a TODOS os modelos.
#
#   limpeza_completa — limpeza_basica + remocao de ruido OCR + remocao
#                      de tokens numericos isolados + remocao de
#                      stopwords (NLTK pt).
#                      Aplicada aos modelos baseados em TF-IDF
#                      (SVM, Regressao Logistica).
#
# Justificativa da divisao:
#   Modelos neurais (Word2Vec, FastText, BERT, BiLSTM) aprendem relacoes
#   contextuais — remover stopwords eliminaria informacao sequencial
#   importante (ex.: "nao provimento"). Modelos bag-of-words (TF-IDF)
#   nao usam contexto, portanto stopwords sao puro ruido.

import re
import json
import unicodedata
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords


# ---------------------------------------------------------------------------
# Constantes de dominio
# ---------------------------------------------------------------------------

LABEL_MAP = {0: 'Acordao', 1: 'ARE', 2: 'Despacho', 3: 'RE', 4: 'Sentenca'}
CORES     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


# ---------------------------------------------------------------------------
# Expressoes regulares (compiladas uma unica vez)
# ---------------------------------------------------------------------------

# URLs (http, https, www)
_RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

# Enderecos de e-mail
_RE_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', re.IGNORECASE)

# Referencias de folhas/paginas (fl., fls., pg.)
_RE_FOLHA = re.compile(r'\b(f\.?l\.?s?\.?|p\.?g\.?)\s*\d*\b', re.IGNORECASE)

# Ruido OCR: mantem letras portuguesas, digitos, espacos e underscore.
# O underscore e preservado para manter tokens LEI_X e ARTIGO_X intactos.
_RE_RUIDO = re.compile(r'[^a-z\u00e0-\u00fc\s\d_]')

# Token puramente numerico (ex.: "1234", "42") — usado em filtragem por token,
# nao como substituicao de caracteres, para evitar corrupcao de LEI_X/ARTIGO_X.
_RE_NUMERO_PURO = re.compile(r'^\d+$')

# Espacos multiplos
_RE_ESPACOS = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

def _carregar_stopwords() -> set:
    """
    Carrega stopwords do NLTK para portugues e adiciona termos
    especificos do dominio juridico sem valor discriminativo.
    """
    try:
        sw = set(stopwords.words('portuguese'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        sw = set(stopwords.words('portuguese'))

    extras = {
        'stf', 'stj', 'trf', 'tst', 'tse', 'tj', 'tre',
        'fl', 'fls', 'pg',
        'email', 'cep', 'fone', 'fax',
    }
    return sw | extras


_STOPWORDS_PT = _carregar_stopwords()


# ===========================================================================
# FUNCOES INTERNAS
# ===========================================================================

def _extrair_corpo(texto: str) -> str:
    """
    Remove o wrapper JSON {"..."} presente no campo Body do dataset VICTOR.
    O AILAB disponibiliza o texto no formato: {"conteudo do documento"}.
    """
    if pd.isna(texto) or str(texto).strip() == '':
        return ''

    texto = str(texto).strip()

    if texto.startswith('{') and texto.endswith('}'):
        try:
            parsed = json.loads(texto)
            if isinstance(parsed, str):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback manual
        interno = texto[1:-1].strip()
        if interno.startswith('"') and interno.endswith('"'):
            interno = interno[1:-1]
        interno = interno.replace('""', '"')
        return interno

    return texto


def _corrigir_encoding(texto: str) -> str:
    """
    Corrige mojibake do conjunto de treino (UTF-8 lido como latin-1).
    Exemplos: 'conclusAo' -> 'conclusao', 'serAo' -> 'serao'.
    O conjunto de teste ja esta correto; a funcao trata ambos sem erro.
    """
    try:
        return texto.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto


# ===========================================================================
# NIVEIS DE LIMPEZA
# ===========================================================================

def limpeza_basica(texto: str) -> str:
    """
    Limpeza basica — indicada para modelos neurais (Word2Vec, FastText,
    BERT, BiLSTM) que dependem do contexto sequencial das palavras.

    Etapas
    ------
    1. Extrai conteudo do wrapper JSON {"..."}
    2. Corrige encoding (mojibake latin-1 -> UTF-8)
    3. Normaliza Unicode para NFC
    4. Converte para minusculas
    5. Remove URLs e enderecos de e-mail
    6. Remove referencias de folhas/paginas (fl., fls., pg.)
    7. Normaliza espacos multiplos

    Tokens LEI_X e ARTIGO_X (substituicoes do AILAB) sao preservados.
    """
    if pd.isna(texto) or str(texto).strip() == '':
        return ''

    texto = _extrair_corpo(texto)
    texto = _corrigir_encoding(texto)
    texto = unicodedata.normalize('NFC', texto)
    texto = texto.lower()
    texto = _RE_URL.sub(' ', texto)
    texto = _RE_EMAIL.sub(' ', texto)
    texto = _RE_FOLHA.sub(' ', texto)
    texto = _RE_ESPACOS.sub(' ', texto).strip()

    return texto


def limpeza_completa(texto: str) -> str:
    """
    Limpeza completa — indicada para modelos bag-of-words (TF-IDF + SVM,
    TF-IDF + Regressao Logistica), onde stopwords sao ruido puro.

    Etapas adicionais (alem da limpeza_basica)
    -------------------------------------------
    5. Remove caracteres especiais / ruido de OCR
    6. Remove tokens puramente numericos (ex.: "1234", "42")
       OBS: filtragem por token inteiro — LEI_102 e ARTIGO_195
       sao preservados porque contem letras e underscore.
    7. Remove stopwords em portugues (NLTK + extras juridicos)
    8. Descarta tokens com 1 caractere ou menos
    """
    texto = limpeza_basica(texto)

    if not texto:
        return ''

    # Remove ruido OCR (preserva underscore para LEI_X / ARTIGO_X)
    texto = _RE_RUIDO.sub(' ', texto)

    # Filtragem por token: remove numericos puros, stopwords e tokens curtos.
    # A checagem e feita no token inteiro, nao em substrings de caracteres,
    # evitando corrupcao de tokens como artigo_102 ou lei_8080.
    tokens = [
        t for t in texto.split()
        if not _RE_NUMERO_PURO.match(t)  # remove "1234", "42", mas nao "artigo_102"
        and t not in _STOPWORDS_PT
        and len(t) > 1
    ]

    return ' '.join(tokens)


# ===========================================================================
# PIPELINE DE APLICACAO AO DATAFRAME
# ===========================================================================

def preprocessar(df: pd.DataFrame,
                 nivel: str = 'basica',
                 coluna: str = 'Body') -> pd.DataFrame:
    """
    Aplica o nivel de limpeza escolhido a todos os documentos do DataFrame.

    Parametros
    ----------
    df    : DataFrame com a coluna de texto.
    nivel : 'basica' ou 'completa'.
    coluna: nome da coluna de texto bruto (padrao: 'Body').

    Retorna
    -------
    Copia do DataFrame com nova coluna Body_basica ou Body_completa.
    """
    nivel = nivel.lower()
    if nivel not in ('basica', 'completa'):
        raise ValueError(f"Nivel '{nivel}' invalido. Use 'basica' ou 'completa'.")

    df        = df.copy()
    col_saida = f'Body_{nivel}'
    funcao    = limpeza_basica if nivel == 'basica' else limpeza_completa

    print(f'{"─" * 50}')
    print(f' Limpeza {nivel} — {len(df):,} documentos')
    print(f'{"─" * 50}')

    df[col_saida] = df[coluna].apply(funcao)

    media_antes  = df[coluna].dropna().apply(lambda x: len(str(x).split())).mean()
    media_depois = df[col_saida].apply(lambda x: len(str(x).split())).mean()
    reducao      = (1 - media_depois / media_antes) * 100

    print(f'  Coluna criada  : {col_saida}')
    print(f'  Tokens (media) : {media_antes:.0f} -> {media_depois:.0f}  '
          f'(reducao de {reducao:.1f}%)\n')

    return df


# ===========================================================================
# VISUALIZACOES
# ===========================================================================

def mostrar_exemplo(texto_bruto: str, n_chars: int = 280):
    """Exibe o efeito de cada nivel de limpeza em um documento de exemplo."""
    basica   = limpeza_basica(texto_bruto)
    completa = limpeza_completa(texto_bruto)
    sep      = '─' * 62

    print(sep)
    print('EFEITO DO PRE-PROCESSAMENTO (exemplo)')
    print(sep)
    print(f'\nORIGINAL  ({len(texto_bruto.split())} tokens):')
    print(f'  {texto_bruto[:n_chars]}...')
    print(f'\nLIMPEZA BASICA  ({len(basica.split())} tokens):')
    print(f'  {basica[:n_chars]}...')
    print(f'\nLIMPEZA COMPLETA  ({len(completa.split())} tokens):')
    print(f'  {completa[:n_chars]}...')
    print(sep)


def plotar_tokens(df: pd.DataFrame, percentil: int = 99):
    """Histogramas comparando distribuicao de tokens entre os niveis."""
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa']
               if c in df.columns]

    titulos = {
        'Body':          'Original',
        'Body_basica':   'Limpeza Basica',
        'Body_completa': 'Limpeza Completa',
    }

    fig, axes = plt.subplots(1, len(colunas),
                             figsize=(5.5 * len(colunas), 4))
    if len(colunas) == 1:
        axes = [axes]

    for ax, col in zip(axes, colunas):
        n_tok = df[col].dropna().apply(lambda x: len(str(x).split()))
        lim   = int(np.percentile(n_tok, percentil))
        ax.hist(n_tok.clip(upper=lim), bins=40,
                color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(n_tok.median(), color='crimson',
                   linestyle='--', linewidth=1.5,
                   label=f'Mediana: {n_tok.median():.0f}')
        ax.axvline(n_tok.mean(), color='darkorange',
                   linestyle=':', linewidth=1.5,
                   label=f'Media: {n_tok.mean():.0f}')
        ax.set_title(titulos.get(col, col), fontsize=11, fontweight='bold')
        ax.set_xlabel('Numero de tokens')
        ax.set_ylabel('Frequencia')
        ax.legend(fontsize=9)

    plt.suptitle('Distribuicao de Tokens por Nivel de Limpeza',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def tabela_comparativa(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com estatisticas de tokens e vocabulario por nivel."""
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa']
               if c in df.columns]

    rotulos = {
        'Body':          'Original',
        'Body_basica':   'Limpeza Basica',
        'Body_completa': 'Limpeza Completa',
    }

    linhas    = []
    ref_media = None
    ref_vocab = None

    for col in colunas:
        n_tok = df[col].dropna().apply(lambda x: len(str(x).split()))
        vocab = set(w for t in df[col].dropna() for w in str(t).split())

        if ref_media is None:
            ref_media = n_tok.mean()
        if ref_vocab is None:
            ref_vocab = len(vocab)

        linhas.append({
            'Nivel':           rotulos.get(col, col),
            'Media tokens':    round(n_tok.mean(), 1),
            'Mediana tokens':  round(n_tok.median(), 1),
            'Vocabulario':     len(vocab),
            'Red. tokens (%)': round((1 - n_tok.mean() / ref_media) * 100, 1),
            'Red. vocab. (%)': round((1 - len(vocab) / ref_vocab) * 100, 1),
        })

    return pd.DataFrame(linhas).set_index('Nivel')


def plotar_vocabulario(df: pd.DataFrame):
    """Grafico de barras comparando tamanho do vocabulario entre os niveis."""
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa']
               if c in df.columns]

    rotulos = {
        'Body':          'Original',
        'Body_basica':   'Limpeza\nBasica',
        'Body_completa': 'Limpeza\nCompleta',
    }

    valores = [len(set(w for t in df[c].dropna()
                       for w in str(t).split())) for c in colunas]
    labels  = [rotulos.get(c, c) for c in colunas]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, valores,
                  color=['#4C72B0', '#55A868', '#DD8452'][:len(colunas)],
                  edgecolor='white', width=0.45)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(valores) * 0.01,
                f'{val:,}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_title('Tamanho do Vocabulario por Nivel de Limpeza',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Termos unicos')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    sns.despine()
    plt.tight_layout()
    plt.show()


def plotar_top_palavras(df: pd.DataFrame,
                        coluna: str = 'Body_completa',
                        top_n: int = 20):
    """Palavras mais frequentes no corpus apos limpeza."""
    contador = Counter(w for t in df[coluna].dropna()
                       for w in str(t).split())
    palavras, freqs = zip(*contador.most_common(top_n))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(top_n), freqs[::-1], color='steelblue', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(palavras[::-1], fontsize=10)
    ax.set_xlabel('Frequencia')
    ax.set_title(f'Top {top_n} palavras mais frequentes ({coluna})',
                 fontsize=12, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.show()


def plotar_top_palavras_por_classe(df: pd.DataFrame,
                                   coluna: str = 'Body_completa',
                                   coluna_label: str = 'Category',
                                   top_n: int = 10):
    """
    Palavras mais frequentes por classe.
    Util para identificar termos discriminativos entre os tipos de documento.
    """
    classes   = sorted(df[coluna_label].dropna().unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes,
                             figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        subset = df[df[coluna_label] == cls][coluna].dropna()
        cont   = Counter(w for t in subset for w in str(t).split())
        if not cont:
            ax.set_visible(False)
            continue
        palavras, freqs = zip(*cont.most_common(top_n))
        ax.barh(range(len(palavras)), freqs[::-1],
                color=CORES[int(cls) % len(CORES)], edgecolor='white')
        ax.set_yticks(range(len(palavras)))
        ax.set_yticklabels(palavras[::-1], fontsize=8)
        ax.set_title(LABEL_MAP.get(int(cls), f'Classe {cls}'),
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Freq.')

    plt.suptitle(f'Top {top_n} palavras por classe ({coluna})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def exibir_resumo(df_orig: pd.DataFrame, df_proc: pd.DataFrame):
    """Resumo textual comparando original e processado."""
    print(f'\n{"=" * 52}')
    print('  RESUMO DO PRE-PROCESSAMENTO')
    print(f'{"=" * 52}')
    print(f'  Documentos processados : {len(df_proc):,}')

    for col in ['Body_basica', 'Body_completa']:
        if col not in df_proc.columns:
            continue
        nivel = col.replace('Body_', '').capitalize()
        orig  = df_orig['Body'].dropna().apply(
            lambda x: len(str(x).split())).mean()
        proc  = df_proc[col].apply(
            lambda x: len(str(x).split())).mean()
        vocab = len(set(w for t in df_proc[col].dropna()
                        for w in str(t).split()))
        print(f'\n  [{nivel}]')
        print(f'    Tokens medios  : {orig:.0f} -> {proc:.0f}  '
              f'(reducao de {(1 - proc/orig)*100:.1f}%)')
        print(f'    Vocabulario    : {vocab:,} termos unicos')

    print(f'\n{"=" * 52}\n')