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
# ETAPAS COBERTAS (exigidas no enunciado):
#   1. Limpeza de ruido de OCR ....... limpeza_completa / limpeza_lematizada
#   2. Tokenizacao ................... tokenizar()
#   3. Remocao de stopwords .......... limpeza_completa / limpeza_lematizada
#   4. Lematizacao ................... lematizar() / limpeza_lematizada()
#   5. Representacao computacional ... vetorizar_tfidf() (esparsa) +
#                                      vetorizar_densa() (embeddings)
#   6. Vetorizacao / dados prontos ... salvar_representacoes()
#
# NIVEIS DE LIMPEZA (base: Martins 2024 e Araujo et al. 2020 - dataset VICTOR):
#
#   limpeza_basica     — correcao de encoding (mojibake), extracao do wrapper
#                        JSON, normalizacao unicode, minusculas, remocao de
#                        URLs/e-mails/referencias de folhas.
#                        Aplicada a TODOS os modelos.
#
#   limpeza_completa   — limpeza_basica + remocao de ruido OCR + remocao de
#                        tokens numericos isolados + remocao de stopwords.
#                        Aplicada aos modelos TF-IDF (SVM, Reg. Logistica).
#
#   limpeza_lematizada — limpeza_completa + lematizacao (spaCy pt). Reduz o
#                        vocabulario agrupando flexoes (recursos -> recurso).
#                        Variante para os modelos bag-of-words.
#
# Justificativa da divisao:
#   Modelos neurais (Word2Vec, FastText, BERT, BiLSTM) aprendem relacoes
#   contextuais — remover stopwords/lematizar eliminaria informacao
#   sequencial importante (ex.: "nao provimento"). Modelos bag-of-words
#   (TF-IDF) nao usam contexto, portanto stopwords sao puro ruido e a
#   lematizacao reduz a esparsidade da matriz.

import re
import os
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

_RE_URL    = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_RE_EMAIL  = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', re.IGNORECASE)
_RE_FOLHA  = re.compile(r'\b(f\.?l\.?s?\.?|p\.?g\.?)\s*\d*\b', re.IGNORECASE)
_RE_RUIDO  = re.compile(r'[^a-z\u00e0-\u00fc\s\d_]')
_RE_NUMERO_PURO = re.compile(r'^\d+$')
_RE_ESPACOS = re.compile(r'\s+')

# Tokens especiais do AILAB (LEI_102, ARTIGO_195) — nao devem ser lematizados.
_RE_ESPECIAL = re.compile(r'^(lei|artigo)_\d+$', re.IGNORECASE)

# Tokenizador: sequencias de letras (com acento) opcionalmente seguidas de
# _digitos. Captura "recurso" e tambem "artigo_543"/"lei_8080" como token unico.
_RE_TOKEN = re.compile(r'[a-z\u00e0-\u00fc]+(?:_\d+)?', re.IGNORECASE)


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

        interno = texto[1:-1].strip()
        if interno.startswith('"') and interno.endswith('"'):
            interno = interno[1:-1]
        interno = interno.replace('""', '"')
        return interno

    return texto


def _corrigir_encoding(texto: str) -> str:
    """
    Corrige mojibake do conjunto de treino (UTF-8 lido como latin-1).
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
    - Remove caracteres especiais / RUIDO DE OCR
    - Remove tokens puramente numericos (ex.: "1234", "42")
      OBS: filtragem por token inteiro — LEI_102 e ARTIGO_195 sao
      preservados porque contem letras e underscore.
    - Remove stopwords em portugues (NLTK + extras juridicos)
    - Descarta tokens com 1 caractere ou menos
    """
    texto = limpeza_basica(texto)

    if not texto:
        return ''

    texto = _RE_RUIDO.sub(' ', texto)

    tokens = [
        t for t in texto.split()
        if not _RE_NUMERO_PURO.match(t)
        and t not in _STOPWORDS_PT
        and len(t) > 1
    ]

    return ' '.join(tokens)


# ===========================================================================
# 2. TOKENIZACAO (explicita)
# ===========================================================================

def tokenizar(texto: str) -> list:
    """
    Tokenizacao baseada em expressao regular, adequada ao dominio juridico.

    Diferente de um simples split(), esta funcao:
      - isola sequencias de letras (com acentuacao portuguesa);
      - mantem tokens especiais do AILAB intactos (artigo_543, lei_8080);
      - descarta pontuacao e simbolos automaticamente.

    Parametros
    ----------
    texto : string ja limpa (recomenda-se aplicar limpeza_basica antes).

    Retorna
    -------
    Lista de tokens (strings).
    """
    if pd.isna(texto) or str(texto).strip() == '':
        return []
    return _RE_TOKEN.findall(str(texto))


# ===========================================================================
# 4. LEMATIZACAO
# ===========================================================================
#
# Estrategia principal: spaCy (modelo pt_core_news_sm), recomendado no
# enunciado. Caso o modelo nao esteja instalado, ha fallback automatico para
# 'simplemma' (lematizador leve, offline), de modo que o pipeline rode em
# qualquer ambiente.
#
#   Instalacao do modelo spaCy (recomendado):
#       python -m spacy download pt_core_news_sm
#   Fallback (opcional):
#       pip install simplemma

_LEMATIZADOR = None
_MODO_LEMA   = None


def _carregar_lematizador():
    """Carrega o lematizador uma unica vez (spaCy; fallback simplemma)."""
    global _LEMATIZADOR, _MODO_LEMA
    if _LEMATIZADOR is not None:
        return

    try:
        import spacy
        # parser e ner sao desnecessarios para lematizar -> mais rapido.
        # O tagger/morphologizer e MANTIDO porque o lematizador pt depende dele.
        _LEMATIZADOR = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
        _LEMATIZADOR.max_length = 2_000_000   # textos juridicos podem ser longos
        _MODO_LEMA   = 'spacy'
    except Exception:
        try:
            import simplemma
            _LEMATIZADOR = simplemma
            _MODO_LEMA   = 'simplemma'
        except Exception as e:
            raise ImportError(
                "Nenhum lematizador disponivel. Instale o modelo spaCy com "
                "'python -m spacy download pt_core_news_sm' ou "
                "'pip install simplemma'."
            ) from e


def usar_lematizador(modo: str = 'auto'):
    """
    Força o lematizador a ser usado. Útil para acelerar o pipeline.

      'auto'      : spaCy se disponível, senão simplemma (padrao).
      'spacy'     : qualidade maior, mais lento.
      'simplemma' : lookup em dicionario, MUITO mais rapido (offline).

    Ex.: pp.usar_lematizador('simplemma')  # antes de preprocessar(nivel='lematizada')
    """
    global _LEMATIZADOR, _MODO_LEMA
    _LEMATIZADOR, _MODO_LEMA = None, None
    if modo == 'simplemma':
        import simplemma
        _LEMATIZADOR, _MODO_LEMA = simplemma, 'simplemma'
    elif modo == 'spacy':
        import spacy
        _LEMATIZADOR = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
        _LEMATIZADOR.max_length = 2_000_000
        _MODO_LEMA = 'spacy'
    elif modo != 'auto':
        raise ValueError("modo deve ser 'auto', 'spacy' ou 'simplemma'.")
    print(f"  Lematizador definido: {modo}")


def _lematizar_doc_spacy(doc) -> str:
    """Extrai os lemas de um Doc spaCy ja processado (preserva artigo_X/lei_X)."""
    saida = []
    for tok in doc:
        forma = tok.text
        if _RE_ESPECIAL.match(forma):
            saida.append(forma)
        elif tok.is_space or tok.is_punct:
            continue
        else:
            saida.append(tok.lemma_.lower())
    return ' '.join(saida)


def _lematizar_lote(textos, batch_size: int = 256, n_process: int = 1) -> list:
    """
    Lematiza uma LISTA de textos de uma vez. Esta e a versao rapida:
    no spaCy usa nlp.pipe() (em lote), evitando o custo fixo de chamar o
    modelo documento a documento.

    n_process > 1 ativa paralelismo (cuidado no Windows/Jupyter).
    """
    _carregar_lematizador()
    textos = [t if isinstance(t, str) else '' for t in textos]

    if _MODO_LEMA == 'spacy':
        return [
            _lematizar_doc_spacy(doc)
            for doc in _LEMATIZADOR.pipe(textos, batch_size=batch_size,
                                         n_process=n_process)
        ]

    # simplemma: lookup token a token (ja e rapido)
    return [lematizar(t) for t in textos]


def lematizar(texto: str) -> str:
    """
    Reduz cada token a sua forma canonica (lema).
    Ex.: 'recursos negados' -> 'recurso negar'.

    Tokens especiais (artigo_543, lei_8080) sao preservados sem alteracao.

    Parametros
    ----------
    texto : string ja limpa.

    Retorna
    -------
    String com os tokens lematizados, separados por espaco.
    """
    _carregar_lematizador()

    if pd.isna(texto) or str(texto).strip() == '':
        return ''
    texto = str(texto)

    if _MODO_LEMA == 'spacy':
        saida = []
        for tok in _LEMATIZADOR(texto):
            forma = tok.text
            if _RE_ESPECIAL.match(forma):
                saida.append(forma)              # preserva artigo_X / lei_X
            elif tok.is_space or tok.is_punct:
                continue
            else:
                saida.append(tok.lemma_.lower())
        return ' '.join(saida)

    # Fallback simplemma (lematiza token a token)
    saida = []
    for t in tokenizar(texto):
        if _RE_ESPECIAL.match(t):
            saida.append(t)
        else:
            saida.append(_LEMATIZADOR.lemmatize(t, lang='pt'))
    return ' '.join(saida)


def _pos_filtro_lema(texto: str) -> str:
    """Aplica, apos a lematizacao, a remocao de ruido OCR + stopwords + tokens curtos."""
    if not texto:
        return ''
    texto = _RE_RUIDO.sub(' ', texto)
    tokens = [
        t for t in texto.split()
        if not _RE_NUMERO_PURO.match(t)
        and t not in _STOPWORDS_PT
        and len(t) > 1
    ]
    return ' '.join(tokens)


def limpeza_lematizada(texto: str) -> str:
    """
    Variante da limpeza_completa que adiciona LEMATIZACAO (um documento).

    Ordem das etapas
    -----------------
    1. limpeza_basica (contexto preservado -> melhor qualidade do lema)
    2. lematizacao
    3. remocao de ruido OCR
    4. remocao de numericos puros, stopwords e tokens curtos

    OBS: para processar um DataFrame inteiro, use preprocessar(nivel='lematizada'),
    que lematiza em LOTE (muito mais rapido que chamar esta funcao em cada linha).
    """
    texto = limpeza_basica(texto)
    if not texto:
        return ''
    return _pos_filtro_lema(lematizar(texto))


# ===========================================================================
# PIPELINE DE APLICACAO AO DATAFRAME
# ===========================================================================

_FUNCOES_LIMPEZA = {
    'basica':     limpeza_basica,
    'completa':   limpeza_completa,
    'lematizada': limpeza_lematizada,
}


def preprocessar(df: pd.DataFrame,
                 nivel: str = 'basica',
                 coluna: str = 'Body',
                 batch_size: int = 256,
                 n_process: int = 1) -> pd.DataFrame:
    """
    Aplica o nivel de limpeza escolhido a todos os documentos do DataFrame.

    Parametros
    ----------
    df         : DataFrame com a coluna de texto.
    nivel      : 'basica', 'completa' ou 'lematizada'.
    coluna     : nome da coluna de texto bruto (padrao: 'Body').
    batch_size : (so 'lematizada') tamanho do lote do spaCy nlp.pipe().
    n_process  : (so 'lematizada') nucleos de CPU; >1 paraleliza
                 (evite em Jupyter no Windows).

    Retorna
    -------
    Copia do DataFrame com nova coluna Body_<nivel>.
    """
    nivel = nivel.lower()
    if nivel not in _FUNCOES_LIMPEZA:
        raise ValueError(
            f"Nivel '{nivel}' invalido. Use {list(_FUNCOES_LIMPEZA)}."
        )

    df        = df.copy()
    col_saida = f'Body_{nivel}'

    print(f'{"─" * 50}')
    print(f' Limpeza {nivel} — {len(df):,} documentos')
    print(f'{"─" * 50}')

    if nivel == 'lematizada':
        # Caminho RAPIDO: limpa em vetor, lematiza em LOTE, filtra em vetor.
        base  = df[coluna].apply(limpeza_basica).tolist()
        lemas = _lematizar_lote(base, batch_size=batch_size, n_process=n_process)
        df[col_saida] = [_pos_filtro_lema(t) for t in lemas]
    else:
        df[col_saida] = df[coluna].apply(_FUNCOES_LIMPEZA[nivel])

    media_antes  = df[coluna].dropna().apply(lambda x: len(str(x).split())).mean()
    media_depois = df[col_saida].apply(lambda x: len(str(x).split())).mean()
    reducao      = (1 - media_depois / media_antes) * 100

    print(f'  Coluna criada  : {col_saida}')
    print(f'  Tokens (media) : {media_antes:.0f} -> {media_depois:.0f}  '
          f'(reducao de {reducao:.1f}%)\n')

    return df


# ===========================================================================
# 5. REPRESENTACAO COMPUTACIONAL — VETORIAL ESPARSA (TF-IDF)
# ===========================================================================

def vetorizar_tfidf(textos_treino,
                    textos_teste=None,
                    max_features: int = 20000,
                    ngram_range: tuple = (1, 2),
                    min_df: int = 2,
                    max_df: float = 0.95,
                    sublinear_tf: bool = True):
    """
    Representacao VETORIAL ESPARSA via TF-IDF (modelos classicos:
    SVM, Regressao Logistica, Naive Bayes).

    IMPORTANTE: o vetorizador e ajustado (fit) APENAS no treino e aplicado
    (transform) ao teste, evitando vazamento de dados (data leakage).

    Retorna
    -------
    (X_treino, X_teste, vetorizador)
      X_treino : matriz esparsa (scipy.sparse) do treino
      X_teste  : matriz esparsa do teste (ou None)
      vetorizador : TfidfVectorizer ajustado (reutilizavel)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vetorizador = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r'(?u)\b\w+\b',   # mantem tokens com underscore (artigo_543)
    )

    X_treino = vetorizador.fit_transform(textos_treino)
    X_teste  = vetorizador.transform(textos_teste) if textos_teste is not None else None

    print(f'  TF-IDF (esparsa)')
    print(f'    Matriz treino : {X_treino.shape}')
    if X_teste is not None:
        print(f'    Matriz teste  : {X_teste.shape}')
    print(f'    Vocabulario   : {len(vetorizador.vocabulary_):,} termos')
    densidade = 100 * X_treino.nnz / (X_treino.shape[0] * X_treino.shape[1])
    print(f'    Densidade     : {densidade:.3f}% (esparsidade {100 - densidade:.3f}%)\n')

    return X_treino, X_teste, vetorizador


# ===========================================================================
# 5. REPRESENTACAO COMPUTACIONAL — VETORIAL DENSA (EMBEDDINGS)
# ===========================================================================

def treinar_word2vec(textos_treino,
                     vector_size: int = 100,
                     window: int = 5,
                     min_count: int = 2,
                     sg: int = 1,
                     epochs: int = 10,
                     workers: int = 4):
    """
    Treina um Word2Vec sobre o corpus de treino (representacao VETORIAL
    DENSA / embeddings, util para BiLSTM e baselines densos).

    sg : 1 = skip-gram, 0 = CBOW.

    Retorna
    -------
    Modelo gensim.models.Word2Vec treinado.
    """
    from gensim.models import Word2Vec

    corpus = [tokenizar(t) for t in textos_treino]
    modelo = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
    )
    print(f'  Word2Vec (densa)')
    print(f'    Vocabulario : {len(modelo.wv.key_to_index):,} termos')
    print(f'    Dimensao    : {vector_size}\n')
    return modelo


def documentos_para_vetores(textos, modelo) -> np.ndarray:
    """
    Converte cada documento em um vetor denso pela MEDIA dos embeddings
    das suas palavras (mean pooling). Documentos sem palavras conhecidas
    recebem um vetor de zeros.

    Retorna
    -------
    np.ndarray de shape (n_documentos, vector_size).
    """
    dim = modelo.vector_size
    vetores = []
    for texto in textos:
        tokens = [t for t in tokenizar(texto) if t in modelo.wv.key_to_index]
        if tokens:
            vetores.append(np.mean(modelo.wv[tokens], axis=0))
        else:
            vetores.append(np.zeros(dim))
    return np.vstack(vetores)


def vetorizar_densa(textos_treino, textos_teste=None, **kw):
    """
    Atalho: treina o Word2Vec no treino e devolve as matrizes densas de
    treino e teste, alem do modelo.

    Retorna
    -------
    (D_treino, D_teste, modelo)
    """
    modelo   = treinar_word2vec(textos_treino, **kw)
    D_treino = documentos_para_vetores(textos_treino, modelo)
    D_teste  = documentos_para_vetores(textos_teste, modelo) if textos_teste is not None else None
    return D_treino, D_teste, modelo


# ===========================================================================
# 6. SALVAMENTO DAS REPRESENTACOES (dados prontos para os modelos)
# ===========================================================================

def salvar_representacoes(diretorio: str,
                          X_tfidf_treino=None, X_tfidf_teste=None,
                          vetorizador_tfidf=None,
                          D_densa_treino=None, D_densa_teste=None,
                          modelo_w2v=None,
                          y_treino=None,
                          ids_teste=None):
    """
    Persiste em disco tudo o que a fase de modelagem precisa, deixando os
    dados PRONTOS para a proxima etapa (treino/teste dos modelos).

    Salva (quando fornecido):
      - tfidf_treino.npz / tfidf_teste.npz : matrizes esparsas
      - tfidf_vectorizer.joblib            : vetorizador ajustado
      - densa_treino.npy / densa_teste.npy : matrizes densas (embeddings)
      - word2vec.model                     : modelo Word2Vec
      - y_treino.npy                       : rotulos (Category)
      - ids_teste.npy                      : Ids do teste (para submissao)
    """
    import joblib
    from scipy import sparse

    os.makedirs(diretorio, exist_ok=True)
    salvos = []

    if X_tfidf_treino is not None:
        sparse.save_npz(os.path.join(diretorio, 'tfidf_treino.npz'), X_tfidf_treino)
        salvos.append('tfidf_treino.npz')
    if X_tfidf_teste is not None:
        sparse.save_npz(os.path.join(diretorio, 'tfidf_teste.npz'), X_tfidf_teste)
        salvos.append('tfidf_teste.npz')
    if vetorizador_tfidf is not None:
        joblib.dump(vetorizador_tfidf, os.path.join(diretorio, 'tfidf_vectorizer.joblib'))
        salvos.append('tfidf_vectorizer.joblib')
    if D_densa_treino is not None:
        np.save(os.path.join(diretorio, 'densa_treino.npy'), D_densa_treino)
        salvos.append('densa_treino.npy')
    if D_densa_teste is not None:
        np.save(os.path.join(diretorio, 'densa_teste.npy'), D_densa_teste)
        salvos.append('densa_teste.npy')
    if modelo_w2v is not None:
        modelo_w2v.save(os.path.join(diretorio, 'word2vec.model'))
        salvos.append('word2vec.model')
    if y_treino is not None:
        np.save(os.path.join(diretorio, 'y_treino.npy'), np.asarray(y_treino))
        salvos.append('y_treino.npy')
    if ids_teste is not None:
        np.save(os.path.join(diretorio, 'ids_teste.npy'), np.asarray(ids_teste))
        salvos.append('ids_teste.npy')

    print(f'  {len(salvos)} artefato(s) salvo(s) em {diretorio}:')
    for s in salvos:
        print(f'    - {s}')
    return salvos


# ===========================================================================
# VISUALIZACOES
# ===========================================================================

def mostrar_exemplo(texto_bruto: str, n_chars: int = 280):
    """Exibe o efeito de cada nivel de limpeza em um documento de exemplo."""
    basica     = limpeza_basica(texto_bruto)
    completa   = limpeza_completa(texto_bruto)
    lematizada = limpeza_lematizada(texto_bruto)
    sep        = '─' * 62

    print(sep)
    print('EFEITO DO PRE-PROCESSAMENTO (exemplo)')
    print(sep)
    print(f'\nORIGINAL  ({len(str(texto_bruto).split())} tokens):')
    print(f'  {str(texto_bruto)[:n_chars]}...')
    print(f'\nLIMPEZA BASICA  ({len(basica.split())} tokens):')
    print(f'  {basica[:n_chars]}...')
    print(f'\nLIMPEZA COMPLETA  ({len(completa.split())} tokens):')
    print(f'  {completa[:n_chars]}...')
    print(f'\nLIMPEZA LEMATIZADA  ({len(lematizada.split())} tokens):')
    print(f'  {lematizada[:n_chars]}...')
    print(sep)


def plotar_tokens(df: pd.DataFrame, percentil: int = 99):
    """Histogramas comparando distribuicao de tokens entre os niveis."""
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa', 'Body_lematizada']
               if c in df.columns]

    titulos = {
        'Body':            'Original',
        'Body_basica':     'Limpeza Basica',
        'Body_completa':   'Limpeza Completa',
        'Body_lematizada': 'Limpeza Lematizada',
    }

    fig, axes = plt.subplots(1, len(colunas), figsize=(5.0 * len(colunas), 4))
    if len(colunas) == 1:
        axes = [axes]

    for ax, col in zip(axes, colunas):
        n_tok = df[col].dropna().apply(lambda x: len(str(x).split()))
        lim   = int(np.percentile(n_tok, percentil))
        ax.hist(n_tok.clip(upper=lim), bins=40,
                color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(n_tok.median(), color='crimson', linestyle='--',
                   linewidth=1.5, label=f'Mediana: {n_tok.median():.0f}')
        ax.axvline(n_tok.mean(), color='darkorange', linestyle=':',
                   linewidth=1.5, label=f'Media: {n_tok.mean():.0f}')
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
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa', 'Body_lematizada']
               if c in df.columns]

    rotulos = {
        'Body':            'Original',
        'Body_basica':     'Limpeza Basica',
        'Body_completa':   'Limpeza Completa',
        'Body_lematizada': 'Limpeza Lematizada',
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
    colunas = [c for c in ['Body', 'Body_basica', 'Body_completa', 'Body_lematizada']
               if c in df.columns]

    rotulos = {
        'Body':            'Original',
        'Body_basica':     'Limpeza\nBasica',
        'Body_completa':   'Limpeza\nCompleta',
        'Body_lematizada': 'Limpeza\nLematizada',
    }

    valores = [len(set(w for t in df[c].dropna() for w in str(t).split()))
               for c in colunas]
    labels  = [rotulos.get(c, c) for c in colunas]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(labels, valores,
                  color=['#4C72B0', '#55A868', '#DD8452', '#C44E52'][:len(colunas)],
                  edgecolor='white', width=0.5)

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
    contador = Counter(w for t in df[coluna].dropna() for w in str(t).split())
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

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
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

    for col in ['Body_basica', 'Body_completa', 'Body_lematizada']:
        if col not in df_proc.columns:
            continue
        nivel = col.replace('Body_', '').capitalize()
        orig  = df_orig['Body'].dropna().apply(lambda x: len(str(x).split())).mean()
        proc  = df_proc[col].apply(lambda x: len(str(x).split())).mean()
        vocab = len(set(w for t in df_proc[col].dropna() for w in str(t).split()))
        print(f'\n  [{nivel}]')
        print(f'    Tokens medios  : {orig:.0f} -> {proc:.0f}  '
              f'(reducao de {(1 - proc/orig)*100:.1f}%)')
        print(f'    Vocabulario    : {vocab:,} termos unicos')

    print(f'\n{"=" * 52}\n')