# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome: Eric Lopes, Guilherme Braga, Guilherme Saggion
# RA: 822873, 823161, 823159
# ################################################################

# Arquivo com todas as funcoes e codigos referentes aos experimentos

# ─── Importações ──────────────────────────────────────────────────────────────
import os
import time
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─── 0. Configuração global ───────────────────────────────────────────────────

SEED        = 42
NUM_CLASSES = 5
CLASS_NAMES  = {0: "Acordao", 1: "ARE", 2: "Despacho", 3: "RE", 4: "Sentenca"}
CLASS_LABELS = list(CLASS_NAMES.values())
DATA_DIR     = Path("../dados")


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS E REPRESENTAÇÕES
# ═════════════════════════════════════════════════════════════════════════════

def carregar_representacoes_tfidf():
    """
    Carrega o vectorizer TF-IDF salvo no pré-processamento e transforma os
    CSVs de treino/teste em matrizes esparsas prontas para os modelos clássicos.

    Retorna
    -------
    X_train_tfidf, X_test_tfidf : scipy.sparse matrix
    y_train                     : np.ndarray  (rótulos inteiros 0-4)
    tfidf_vectorizer            : TfidfVectorizer ajustado
    train_completo, test_completo : pd.DataFrame
    """
    with open(DATA_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    train_completo = pd.read_csv(DATA_DIR / "train_completo.csv")
    test_completo  = pd.read_csv(DATA_DIR / "test_completo.csv")

    # --- TRAVA DE SEGURANÇA ---
    # Elimina a classe extra (-1) antes de gerar as matrizes
    train_completo = train_completo[train_completo["Category"] != -1].copy()

    X_train_tfidf = tfidf_vectorizer.transform(
        train_completo["Body_completa"].fillna("")
    )
    X_test_tfidf = tfidf_vectorizer.transform(
        test_completo["Body_completa"].fillna("")
    )
    y_train = train_completo["Category"].values

    print(f"X_train_tfidf : {X_train_tfidf.shape}")
    print(f"X_test_tfidf  : {X_test_tfidf.shape}")
    print(f"y_train       : {y_train.shape}  | classes: {np.unique(y_train)}")

    return (
        X_train_tfidf, X_test_tfidf,
        y_train,
        tfidf_vectorizer,
        train_completo, test_completo,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. UTILITÁRIOS DE AVALIAÇÃO COMPARTILHADOS
# ═════════════════════════════════════════════════════════════════════════════

def avaliar_modelo(nome, modelo, X_tr, y_tr, cv=5, verbose=True):
    """
    Avalia um estimador scikit-learn com validação cruzada estratificada.

    Métrica principal: F1-macro (robusta ao desbalanceamento extremo das classes).

    Parâmetros
    ----------
    nome    : rótulo do modelo para exibição e tabela de resultados.
    modelo  : estimador compatível com scikit-learn.
    X_tr    : matriz de features (densa ou esparsa).
    y_tr    : vetor de rótulos inteiros.
    cv      : número de folds (padrão 5).
    verbose : se True, imprime uma linha com as métricas.

    Retorna
    -------
    dict com Modelo, F1-Macro, F1-Weighted, Acuracia e Tempo CV.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scoring = {
        "f1_macro"   : make_scorer(f1_score, average="macro",    zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "accuracy"   : "accuracy",
    }
    t0     = time.time()
    scores = cross_validate(
        modelo, X_tr, y_tr,
        cv=skf, scoring=scoring,
        n_jobs=-1, return_train_score=False,
    )
    elapsed = time.time() - t0

    result = {
        "Modelo"             : nome,
        "F1-Macro (media)"   : scores["test_f1_macro"].mean(),
        "F1-Macro (std)"     : scores["test_f1_macro"].std(),
        "F1-Weighted (media)": scores["test_f1_weighted"].mean(),
        "Acuracia (media)"   : scores["test_accuracy"].mean(),
        "Tempo CV (s)"       : round(elapsed, 1),
    }
    if verbose:
        print(
            f"  {nome:<38}  "
            f"F1-macro={result['F1-Macro (media)']:.4f} "
            f"(+/-{result['F1-Macro (std)']:.4f})  "
            f"acc={result['Acuracia (media)']:.4f}  "
            f"tempo={result['Tempo CV (s)']}s"
        )
    return result


def plotar_matriz_confusao(nome, modelo, X_tr, y_tr, cmap="Blues"):
    """
    Treina o modelo em X_tr completo e plota a matriz de confusão normalizada
    por linha (recall por classe), seguida do classification_report detalhado.
    """
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_tr)
    cm = confusion_matrix(y_tr, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap=cmap,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
    )
    ax.set_title(f"Matriz de Confusão (treino) — {nome}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.show()
    print(classification_report(y_tr, y_pred,
                                target_names=CLASS_LABELS, zero_division=0))


def gerar_submissao_classico(nome_arquivo, modelo, X_tr, y_tr, X_te, ids_te):
    """
    Treina o modelo no conjunto completo de treino e gera o CSV de submissão
    no formato Id / Category exigido pelo Kaggle.
    """
    modelo.fit(X_tr, y_tr)
    y_pred  = modelo.predict(X_te)
    sub     = pd.DataFrame({"Id": ids_te, "Category": y_pred})
    caminho = DATA_DIR / nome_arquivo
    sub.to_csv(caminho, index=False)
    print(f"  Submissão salva em: {caminho}  ({len(sub)} registros)")
    return sub


def tabela_comparativa_classicos(resultados: list) -> pd.DataFrame:
    """
    Monta e exibe o DataFrame comparativo dos modelos clássicos,
    formata os valores numéricos e salva em CSV.

    Parâmetros
    ----------
    resultados : lista de dicts retornados por avaliar_modelo().

    Retorna
    -------
    pd.DataFrame indexado por Modelo.
    """
    df = pd.DataFrame(resultados).set_index("Modelo")

    disp = df.copy()
    for col in ["F1-Macro (media)", "F1-Macro (std)",
                "F1-Weighted (media)", "Acuracia (media)"]:
        disp[col] = disp[col].apply(lambda x: f"{x:.4f}")
    disp["Tempo CV (s)"] = disp["Tempo CV (s)"].apply(lambda x: f"{x:.1f}s")

    print("\n" + "=" * 65)
    print("COMPARATIVO — MODELOS CLÁSSICOS (TF-IDF + Limpeza Completa)")
    print("=" * 65)

    df.to_csv(DATA_DIR / "resultados_classicos.csv")
    return df, disp


def plotar_comparativo_classicos(df_resultados: pd.DataFrame):
    """Gráfico de barras com F1-macro ± std para os modelos clássicos."""
    nomes   = df_resultados.index.tolist()
    f1_mean = df_resultados["F1-Macro (media)"].values
    f1_std  = df_resultados["F1-Macro (std)"].values
    cores   = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(nomes, f1_mean, yerr=f1_std, capsize=6,
                  color=cores, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, f1_mean):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, min(1.0, f1_mean.max() + 0.12))
    ax.set_ylabel("F1-Macro (média 5-fold)")
    ax.set_title("Comparativo de Modelos Clássicos — F1-macro (5-fold CV)")
    ax.axhline(f1_mean.max(), color="gray", linestyle="--",
               linewidth=0.8, label="Melhor")
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()


def treinar_e_salvar_classico(nome_arquivo, modelo, X, y, subdir="modelos_classicos"):
    """Treina em todo o conjunto e persiste o modelo com pickle."""
    destino = DATA_DIR / subdir
    destino.mkdir(exist_ok=True)
    modelo.fit(X, y)
    caminho = destino / nome_arquivo
    with open(caminho, "wb") as f:
        pickle.dump(modelo, f)
    print(f"  Salvo: {caminho}")
    return modelo


# ═════════════════════════════════════════════════════════════════════════════
# 3. MODELOS CLÁSSICOS
# ═════════════════════════════════════════════════════════════════════════════

def treinar_naive_bayes(X_train, y_train, cv_inner=3):
    """
    Busca o melhor alpha para o Complement NB via GridSearchCV (3-fold interno)
    e avalia o modelo otimizado com CV externo de 5 folds.

    Retorna
    -------
    gs          : GridSearchCV ajustado.
    resultado   : dict de métricas (para tabela comparativa).
    best_model  : ComplementNB com o melhor alpha.
    """
    from sklearn.naive_bayes import ComplementNB

    scorer_f1 = make_scorer(f1_score, average="macro", zero_division=0)
    skf_inner = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=SEED)

    print("=" * 65)
    print("NAÏVE BAYES — busca de hiperparâmetros (GridSearchCV)")
    print("=" * 65)

    gs = GridSearchCV(
        ComplementNB(),
        {"alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]},
        cv=skf_inner, scoring=scorer_f1, n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  Melhor alpha  : {gs.best_params_['alpha']}")
    print(f"  F1-macro (CV) : {gs.best_score_:.4f}")

    best_model = ComplementNB(alpha=gs.best_params_["alpha"])
    resultado  = avaliar_modelo(
        "Complement NB (melhor alpha)", best_model, X_train, y_train, cv=5
    )
    return gs, resultado, best_model


def plotar_curva_alpha_nb(gs_nb):
    """Plota a curva de F1-macro por valor de alpha para o Complement NB."""
    cv_res = pd.DataFrame(gs_nb.cv_results_)
    plt.figure(figsize=(8, 4))
    plt.plot(cv_res["param_alpha"].astype(float),
             cv_res["mean_test_score"], marker="o", color="steelblue")
    plt.fill_between(
        cv_res["param_alpha"].astype(float),
        cv_res["mean_test_score"] - cv_res["std_test_score"],
        cv_res["mean_test_score"] + cv_res["std_test_score"],
        alpha=0.2, color="steelblue",
    )
    plt.xscale("log")
    plt.xlabel("alpha (escala log)")
    plt.ylabel("F1-macro (CV)")
    plt.title("Complement NB — Efeito do alpha na F1-macro")
    plt.tight_layout()
    plt.show()


def treinar_regressao_logistica(X_train, y_train, cv_inner=3):
    """
    Busca o melhor C para a Regressão Logística (L2, balanced, lbfgs)
    via GridSearchCV e avalia com CV externo de 5 folds.

    Retorna
    -------
    gs, resultado, best_model
    """
    from sklearn.linear_model import LogisticRegression

    scorer_f1 = make_scorer(f1_score, average="macro", zero_division=0)

    print("=" * 65)
    print("REGRESSÃO LOGÍSTICA — busca de hiperparâmetros (GridSearchCV)")
    print("=" * 65)

    gs = GridSearchCV(
        LogisticRegression(
            class_weight="balanced", 
            solver="lbfgs",       # MUDANÇA 1: lbfgs é muito mais rápido para L2 + TF-IDF
            max_iter=1000, 
            random_state=SEED, 
            n_jobs=1              # MUDANÇA 2: Deixa o paralelismo para o GridSearchCV
        ),
        {"C": [0.01, 0.1, 1.0, 5.0, 10.0], "penalty": ["l2"]},
        cv=StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=SEED),
        scoring=scorer_f1, 
        n_jobs=-1,                # MUDANÇA 3: Roda as diferentes combinações em paralelo
        verbose=1,                # BÔNUS: Adicionado verbose=1 para você ver o progresso (barrinha) e não achar que travou
    )
    
    gs.fit(X_train, y_train)
    
    print(f"  Melhores params : {gs.best_params_}")
    print(f"  F1-macro (CV)   : {gs.best_score_:.4f}")

    # Constrói o modelo final com os melhores parâmetros
    best_model = LogisticRegression(
        C=gs.best_params_["C"], 
        penalty="l2",
        class_weight="balanced", 
        solver="lbfgs",           # Lembre-se de manter o mesmo solver aqui
        max_iter=1000, 
        random_state=SEED, 
        n_jobs=-1,
    )
    
    resultado = avaliar_modelo(
        "Regressão Logística (melhor C)", best_model, X_train, y_train, cv=5
    )
    
    return gs, resultado, best_model


def plotar_coeficientes_lr(modelo_lr, tfidf_vectorizer, top_n=12):
    """
    Plota os termos com maior coeficiente positivo e negativo por classe
    para a Regressão Logística, revelando as features mais discriminativas.
    """
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for idx, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        coef    = modelo_lr.coef_[idx]
        top_pos = np.argsort(coef)[-top_n:][::-1]
        top_neg = np.argsort(coef)[:top_n]
        terms   = np.concatenate([feature_names[top_pos], feature_names[top_neg]])
        vals    = np.concatenate([coef[top_pos], coef[top_neg]])
        cores   = ["#2166ac"] * top_n + ["#d6604d"] * top_n
        axes[idx].barh(terms[::-1], vals[::-1], color=cores[::-1])
        axes[idx].axvline(0, color="black", linewidth=0.8)
        axes[idx].set_title(cls_name, fontsize=10)
        axes[idx].tick_params(labelsize=7)

    fig.suptitle(
        "Termos mais discriminativos por classe — Regressão Logística",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def treinar_svm(X_train, y_train, cv_inner=3):
    """
    Busca o melhor (alpha, class_weight) para o SGDClassifier loss='hinge'
    (SVM linear) via GridSearchCV e avalia com CV externo de 5 folds.

    Retorna
    -------
    gs, resultado, best_model
    """
    from sklearn.linear_model import SGDClassifier

    scorer_f1 = make_scorer(f1_score, average="macro", zero_division=0)

    print("=" * 65)
    print("SVM LINEAR (SGD) — busca de hiperparâmetros (GridSearchCV)")
    print("=" * 65)

    gs = GridSearchCV(
        SGDClassifier(
            loss="hinge", penalty="l2",
            max_iter=200, tol=1e-3,
            n_jobs=-1, random_state=SEED,
        ),
        {
            "alpha"       : [1e-5, 1e-4, 1e-3, 1e-2],
            "class_weight": ["balanced", None],
        },
        cv=StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=SEED),
        scoring=scorer_f1, n_jobs=1, verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  Melhores params : {gs.best_params_}")
    print(f"  F1-macro (CV)   : {gs.best_score_:.4f}")

    best_model = _build_svm(gs.best_params_)
    resultado  = avaliar_modelo(
        "SVM Linear/SGD (melhores params)", best_model, X_train, y_train, cv=5
    )
    return gs, resultado, best_model


def _build_nb(params):
    """Constrói um ComplementNB com os parâmetros do GridSearch."""
    from sklearn.naive_bayes import ComplementNB
    return ComplementNB(alpha=params["alpha"])


def _build_lr(params):
    """Constrói uma LogisticRegression com os parâmetros do GridSearch."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=params["C"], penalty="l2",
        class_weight="balanced", solver="saga",
        max_iter=1000, random_state=SEED, n_jobs=-1,
    )


def _build_svm(params):
    """Constrói um SGDClassifier com os parâmetros do GridSearch."""
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(
        loss="hinge", penalty="l2",
        alpha=params["alpha"],
        class_weight=params["class_weight"],
        max_iter=200, tol=1e-3,
        n_jobs=-1, random_state=SEED,
    )


def plotar_heatmap_svm(gs_svm):
    """Mapa de calor de F1-macro por (alpha, class_weight) para o SVM."""
    cv_df = pd.DataFrame(gs_svm.cv_results_)
    pivot = cv_df.pivot_table(
        index="param_alpha",
        columns="param_class_weight",
        values="mean_test_score",
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("SVM — F1-macro por alpha e class_weight")
    plt.xlabel("class_weight")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 4. MODELOS PROFUNDOS (BiLSTM e TextCNN)
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1 Setup e vocabulário ───────────────────────────────────────────────────

def configurar_tensorflow():
    """Importa e configura TensorFlow; retorna o device disponível."""
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    print(f"TensorFlow   : {tf.__version__}")
    print(f"GPUs visíveis: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU threads  : {os.cpu_count()}")
    return tf


def construir_vocabulario(train_basico: pd.DataFrame, max_vocab=30_000):
    """
    Constrói vocabulário a partir dos textos de treino (limpeza básica).

    Retorna
    -------
    word2idx  : dict {token: índice_inteiro}
    vocab_size : int
    """
    all_tokens = [
        tok
        for text in train_basico["Body_basica"].fillna("").tolist()
        for tok in str(text).split()
    ]
    counter   = Counter(all_tokens)
    vocab     = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(max_vocab - 2)]
    word2idx  = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulário  : {vocab_size:,} tokens")
    return word2idx, vocab_size


def encode_e_pad(texts, word2idx, max_len=300):
    """
    Converte lista de textos em sequências de inteiros preenchidas com zeros.

    Parâmetros
    ----------
    texts    : pd.Series ou lista de strings.
    word2idx : dict {token: índice}.
    max_len  : comprimento fixo de saída (trunca ou preenche com 0).

    Retorna
    -------
    np.ndarray de shape (N, max_len).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = [[word2idx.get(tok, 1) for tok in str(t).split()] for t in texts]
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def carregar_dados_neurais(word2idx, max_len=300):
    """
    Carrega os CSVs de limpeza básica, constrói matrizes de sequências e o
    split hold-out estratificado (85% treino / 15% validação).

    Retorna
    -------
    X_seq_train, X_seq_test, y_nn,
    X_tr_nn, X_val_nn, y_tr_nn, y_val_nn
    """
    train_basico = pd.read_csv(DATA_DIR / "train_basico.csv")
    test_basico  = pd.read_csv(DATA_DIR / "test_basico.csv")

    # --- TRAVA DE SEGURANÇA ---
    train_basico = train_basico[train_basico["Category"] != -1].copy()

    X_seq_train = encode_e_pad(train_basico["Body_basica"], word2idx, max_len)
    X_seq_test  = encode_e_pad(test_basico["Body_basica"],  word2idx, max_len)
    y_nn        = train_basico["Category"].values

    X_tr_nn, X_val_nn, y_tr_nn, y_val_nn = train_test_split(
        X_seq_train, y_nn,
        test_size=0.15, random_state=SEED, stratify=y_nn,
    )
    print(f"X_seq_train  : {X_seq_train.shape}")
    print(f"X_seq_test   : {X_seq_test.shape}")
    print(f"Hold-out     — treino: {X_tr_nn.shape}, val: {X_val_nn.shape}")
    return X_seq_train, X_seq_test, y_nn, X_tr_nn, X_val_nn, y_tr_nn, y_val_nn


def carregar_embedding_matrix(word2idx, vocab_size, embed_dim=100):
    """
    Carrega o Word2Vec salvo no pré-processamento e monta a matriz de embeddings.
    Se o arquivo não existir, treina um novo modelo on-the-fly.

    Retorna
    -------
    embed_matrix : np.ndarray de shape (vocab_size, embed_dim)
    embed_dim    : dimensão real (pode diferir se o W2V tiver dim diferente)
    """
    from gensim.models import Word2Vec

    w2v_path = DATA_DIR / "representacoes" / "word2vec.model"
    if w2v_path.exists():
        modelo_w2v = Word2Vec.load(str(w2v_path))
        embed_dim  = modelo_w2v.vector_size
        print(f"Word2Vec carregado — dim={embed_dim}, vocab={len(modelo_w2v.wv):,}")
    else:
        print("Modelo W2V não encontrado — treinando on-the-fly...")
        train_basico = pd.read_csv(DATA_DIR / "train_basico.csv")
        
        # --- TRAVA DE SEGURANÇA ---
        # Remove textos não rotulados para não enviesar o embedding
        train_basico = train_basico[train_basico["Category"] != -1].copy()
        
        sentences    = [str(t).split() for t in train_basico["Body_basica"].fillna("")]
        modelo_w2v   = Word2Vec(
            sentences, vector_size=embed_dim, window=5,
            min_count=2, sg=1, workers=4, seed=SEED,
        )
        print(f"Word2Vec treinado — dim={embed_dim}, vocab={len(modelo_w2v.wv):,}")

    embed_matrix = np.zeros((vocab_size, embed_dim), dtype="float32")
    hits = 0
    for word, idx in word2idx.items():
        if word in modelo_w2v.wv:
            embed_matrix[idx] = modelo_w2v.wv[word]
            hits += 1

    coverage = hits / vocab_size * 100
    print(f"Cobertura da matriz de embedding: {coverage:.1f}%  ({hits:,}/{vocab_size:,})")
    return embed_matrix, embed_dim


def calcular_pesos_classe(y):
    """
    Calcula pesos de classe inversamente proporcionais à frequência.
    Compatível com keras class_weight e sklearn compute_class_weight.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    print("Pesos de classe:", {CLASS_NAMES[k]: round(v, 3) for k, v in cw.items()})
    return cw


# ── 4.2 Arquiteturas neurais ──────────────────────────────────────────────────

def _embedding_layer(vocab_size, embed_dim, embed_matrix, max_len, trainable=False):
    """Camada de embedding inicializada com W2V."""
    from tensorflow.keras import layers
    return layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        weights=[embed_matrix],
        input_length=max_len,
        trainable=trainable,
        name="embedding_w2v",
    )


def construir_bilstm(vocab_size, embed_dim, embed_matrix, max_len,
                     lstm_units=128, dropout_rate=0.4,
                     l2_lambda=1e-4, embed_trainable=False):
    """
    Cria e compila uma rede BiLSTM de dois níveis com regularização L2.

    Arquitetura: Embedding → SpatialDropout → BiLSTM → BiLSTM → Dense → Dropout → Softmax
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inp = keras.Input(shape=(max_len,), name="input_seq")
    x   = _embedding_layer(vocab_size, embed_dim, embed_matrix, max_len, embed_trainable)(inp)
    x   = layers.SpatialDropout1D(0.3)(x)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units, return_sequences=True,
                          kernel_regularizer=regularizers.l2(l2_lambda)),
              name="bilstm_1",
          )(x)
    x   = layers.Bidirectional(
              layers.LSTM(lstm_units // 2,
                          kernel_regularizer=regularizers.l2(l2_lambda)),
              name="bilstm_2",
          )(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="BiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def construir_textcnn(vocab_size, embed_dim, embed_matrix, max_len,
                      filter_sizes=(2, 3, 4, 5), num_filters=128,
                      dropout_rate=0.5, embed_trainable=False):
    """
    Cria e compila uma rede TextCNN (Kim 2014) com filtros paralelos de múltiplos tamanhos.

    Arquitetura: Embedding → [Conv1D → GlobalMaxPool] × N → Concat → Dense → Dropout → Softmax
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inp   = keras.Input(shape=(max_len,), name="input_seq")
    embed = _embedding_layer(vocab_size, embed_dim, embed_matrix, max_len, embed_trainable)(inp)
    embed = layers.SpatialDropout1D(0.25)(embed)

    pooled = []
    for fs in filter_sizes:
        conv = layers.Conv1D(num_filters, fs, activation="relu",
                             padding="same", name=f"conv_{fs}gram")(embed)
        pool = layers.GlobalMaxPooling1D(name=f"pool_{fs}gram")(conv)
        pooled.append(pool)

    x   = layers.Concatenate(name="concat_pools")(pooled)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(dropout_rate)(x)
    x   = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="TextCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 4.3 Treinamento e avaliação neural ───────────────────────────────────────

def treinar_modelo_nn(model, X_tr, y_tr, X_val, y_val,
                      class_weights, epochs=20, batch_size=64,
                      patience_stop=3, patience_lr=2):
    """
    Treina um modelo Keras com EarlyStopping e ReduceLROnPlateau.

    Retorna
    -------
    history : keras History object.
    tempo   : float (segundos de treinamento).
    """
    from tensorflow.keras import callbacks

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_stop,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience_lr, min_lr=1e-6, verbose=1,
        ),
    ]
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )
    tempo = time.time() - t0
    print(f"Treinamento concluído em {tempo:.1f}s")
    return history, tempo


def plotar_historico(history, titulo):
    """Plota curvas de Loss e Accuracy (treino vs. validação) por época."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (m_tr, m_val), ylabel in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"],
    ):
        ax.plot(history.history[m_tr],  label="Treino")
        ax.plot(history.history[m_val], label="Validação")
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].set_title(f"{titulo} — Loss")
    axes[1].set_title(f"{titulo} — Accuracy")
    plt.tight_layout()
    plt.show()


def treinar_modelo_nn(model, X_tr, y_tr, X_val, y_val,
                      class_weights, epochs=20, batch_size=256,
                      patience_stop=3, patience_lr=2):
    """
    Treina um modelo Keras com tf.data.Dataset para máxima performance de I/O,
    utilizando EarlyStopping e ReduceLROnPlateau.

    Retorna
    -------
    history : keras History object.
    tempo   : float (segundos de treinamento).
    """
    import time
    import tensorflow as tf
    from tensorflow.keras import callbacks

    # 1. Criação de Pipeline de Dados Otimizado (Zero gargalo de memória)
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Dataset de treino com embaralhamento e prefetch
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    train_ds = train_ds.shuffle(buffer_size=len(X_tr), seed=42)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    
    # Dataset de validação
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # 2. Configuração de Callbacks
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_stop,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience_lr, min_lr=1e-6, verbose=1,
        ),
    ]
    
    t0 = time.time()
    
    # 3. Treinamento
    history = model.fit(
        train_ds,                      # Agora passamos o pipeline inteiro
        validation_data=val_ds,        
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=2,                     # verbose=2 retira a animação pesada do notebook
    )
    
    tempo = time.time() - t0
    print(f"Treinamento concluído em {tempo:.1f}s")
    
    return history, tempo


def plotar_matriz_confusao_nn(nome, model, X_val, y_val, cmap="Blues"):
    """Plota matriz de confusão normalizada para um modelo Keras."""
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm = confusion_matrix(y_val, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_title(f"Matriz de Confusão — {nome} (hold-out 15%)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.show()


def tabela_comparativa_profundos(resultados: list) -> pd.DataFrame:
    """Monta, exibe e salva o comparativo dos modelos profundos."""
    df   = pd.DataFrame(resultados).set_index("Modelo")
    disp = df.copy()
    for col in ["F1-Macro (media)", "F1-Weighted (media)", "Acuracia (media)"]:
        disp[col] = disp[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 60)
    print("COMPARATIVO — MODELOS PROFUNDOS (hold-out 15%)")
    print("=" * 60)

    df.to_csv(DATA_DIR / "resultados_profundos.csv")
    return df, disp


def plotar_comparativo_profundos(df_resultados: pd.DataFrame):
    """Gráfico de barras com F1-macro para BiLSTM e TextCNN."""
    nomes  = df_resultados.index.tolist()
    f1_nn  = df_resultados["F1-Macro (media)"].values
    cores  = ["#76b7b2", "#59a14f"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(nomes, f1_nn, color=cores, alpha=0.85,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, f1_nn):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(0, min(1.0, f1_nn.max() + 0.10))
    ax.set_ylabel("F1-Macro (hold-out 15%)")
    ax.set_title("Modelos Profundos — F1-macro (hold-out)")
    sns.despine()
    plt.tight_layout()
    plt.show()


def salvar_modelos_profundos(bilstm_model, cnn_model, word2idx):
    """Persiste os modelos Keras e o vocabulário no diretório de dados."""
    nn_dir = DATA_DIR / "modelos_profundos"
    nn_dir.mkdir(exist_ok=True)
    bilstm_model.save(nn_dir / "bilstm.keras")
    cnn_model.save(nn_dir / "textcnn.keras")
    with open(nn_dir / "word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    print(f"Modelos profundos salvos em: {nn_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. TRANSFORMERS (BERTimbau LoRA + Full Fine-tuning)
# ═════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "neuralmind/bert-base-portuguese-cased"
MAX_LEN_BERT = 512
TRANS_DIR    = DATA_DIR / "modelos_transformers"


def detectar_device():
    """Detecta e imprime o dispositivo disponível (CPU ou GPU)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo : {device}")
    if device == "cuda":
        import torch
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def preparar_datasets_bert(bert_batch=16):
    """
    Tokeniza os dados de treino/validação/teste com o BERTimbau tokenizer
    e retorna os HuggingFace DatasetDicts prontos para o Trainer.

    Retorna
    -------
    tok_ds       : DatasetDict com splits 'train', 'val', 'test'
    tokenizer    : AutoTokenizer ajustado
    data_collator: DataCollatorWithPadding
    df_tr, df_val: DataFrames para métricas externas
    """
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from datasets import Dataset, DatasetDict

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_basico = pd.read_csv(DATA_DIR / "train_basico.csv")
    test_basico  = pd.read_csv(DATA_DIR / "test_basico.csv")

    # --- TRAVA DE SEGURANÇA ---
    # Garante que o Transformer não receba rótulos inválidos
    train_basico = train_basico[train_basico["Category"] != -1].copy()

    df_tr, df_val = train_test_split(
        train_basico, test_size=0.15,
        random_state=SEED, stratify=train_basico["Category"],
    )

    def _make_ds(df, has_labels=True):
        d = {"text": df["Body_basica"].fillna("").tolist()}
        if has_labels:
            d["label"] = df["Category"].tolist()
        return Dataset.from_dict(d)

    raw_ds = DatasetDict({
        "train": _make_ds(df_tr),
        "val"  : _make_ds(df_val),
        "test" : _make_ds(test_basico, has_labels=False),
    })

    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    tok_ds = raw_ds.map(_tokenize, batched=True,
                        remove_columns=["text"], desc="Tokenizando")

    print("Splits tokenizados:")
    for split, ds in tok_ds.items():
        print(f"  {split:<6}: {len(ds):,} exemplos | colunas: {ds.column_names}")

    return tok_ds, tokenizer, data_collator, df_tr, df_val


def compute_metrics_bert(eval_pred):
    """Função de métricas para o HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro"   : f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "accuracy"   : float((labels == preds).mean()),
    }


def _make_weighted_trainer_class(class_weights_nn, device):
    """
    Fábrica que cria a classe WeightedTrainer com os pesos de classe corretos.
    Necessário porque a classe captura class_weights_tensor via closure.
    """
    import torch
    from transformers import Trainer

    class_weights_tensor = torch.tensor(
        [class_weights_nn[i] for i in range(NUM_CLASSES)], dtype=torch.float
    ).to(device)

    class WeightedTrainer(Trainer):
        """Trainer customizado com CrossEntropy ponderada pelo desbalanceamento."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss    = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def treinar_lora(tok_ds, tokenizer, data_collator,
                 class_weights_nn, device, bert_batch=16, max_train_samples=None):
    """
    Configura e treina o BERTimbau com PEFT/LoRA (r=16, target: query+value).

    Retorna
    -------
    trainer_lora : WeightedTrainer ajustado.
    lora_model   : PeftModel treinado.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments, EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    TRANS_DIR.mkdir(exist_ok=True)

    train_dataset = tok_ds["train"]
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=SEED).select(range(max_train_samples))
        print(f"Usando subset de treino: {max_train_samples:,} exemplos "
              f"(de {len(tok_ds['train']):,} originais)")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in CLASS_NAMES.items()},
        label2id={n: i for i, n in CLASS_NAMES.items()},
        ignore_mismatched_sizes=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1, bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    WeightedTrainer = _make_weighted_trainer_class(class_weights_nn, device)

    args = TrainingArguments(
        output_dir=str(TRANS_DIR / "lora_checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=bert_batch,
        per_device_eval_batch_size=bert_batch * 2,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=(device == "cuda"),
        dataloader_num_workers=2,
        seed=SEED,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=lora_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=tok_ds["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_bert,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    print("Iniciando treinamento LoRA...")
    t0 = time.time()
    trainer.train()
    print(f"Treinamento LoRA concluído em {time.time() - t0:.1f}s")
    return trainer, lora_model


def avaliar_transformer(nome, trainer, tok_ds, df_val, cmap="Oranges"):
    """
    Avalia um Trainer HuggingFace no split 'val', imprime o relatório
    detalhado e plota a matriz de confusão.

    Retorna
    -------
    resultado : dict com métricas para tabela comparativa.
    """
    metrics  = trainer.evaluate(tok_ds["val"])
    print(f"\nMétricas {nome} (validação):")
    for k, v in metrics.items():
        if not k.startswith("eval_runtime"):
            print(f"  {k:<30}: {v:.4f}")

    pred_out = trainer.predict(tok_ds["val"])
    y_pred   = np.argmax(pred_out.predictions, axis=-1)
    y_true   = df_val["Category"].values

    print(f"\nRelatório de classificação — {nome} (hold-out 15%):")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_title(f"Matriz de Confusão — {nome} (hold-out 15%)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.show()

    return {
        "Modelo"             : nome,
        "F1-Macro (media)"   : metrics.get("eval_f1_macro", 0),
        "F1-Weighted (media)": metrics.get("eval_f1_weighted", 0),
        "Acuracia (media)"   : metrics.get("eval_accuracy", 0),
    }


def salvar_lora(lora_model, tokenizer):
    """Salva o adaptador LoRA e o tokenizer no diretório de modelos."""
    caminho = TRANS_DIR / "bertimbau_lora_adapter"
    lora_model.save_pretrained(str(caminho))
    tokenizer.save_pretrained(str(caminho))
    print(f"Adaptador LoRA salvo em: {caminho}")
    print("# Para carregar:")
    print("#   base  = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)")
    print("#   model = PeftModel.from_pretrained(base, str(caminho))")


def treinar_finetuning_completo(tok_ds, tokenizer, data_collator,
                                class_weights_nn, device, bert_batch=16,
                                max_train_samples=None):
    if device != "cuda":
        print("GPU não disponível — pulando full fine-tuning.")
        return None

    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments, EarlyStoppingCallback,
    )

    TRANS_DIR.mkdir(exist_ok=True)
    WeightedTrainer = _make_weighted_trainer_class(class_weights_nn, device)

    train_dataset = tok_ds["train"]
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=SEED).select(range(max_train_samples))
        print(f"Usando subset de treino: {max_train_samples:,} exemplos "
              f"(de {len(tok_ds['train']):,} originais)")

    ft_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in CLASS_NAMES.items()},
        label2id={n: i for i, n in CLASS_NAMES.items()},
        ignore_mismatched_sizes=True,
    )
    args = TrainingArguments(
        output_dir=str(TRANS_DIR / "ft_checkpoints"),
        num_train_epochs=2,                  # era 4
        per_device_train_batch_size=bert_batch,
        per_device_eval_batch_size=bert_batch * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=2,
        seed=SEED,
        report_to="none",
    )
    trainer_ft = WeightedTrainer(
        model=ft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=tok_ds["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_bert,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    print("Iniciando full fine-tuning...")
    t0 = time.time()
    trainer_ft.train()
    print(f"Fine-tuning concluído em {time.time() - t0:.1f}s")

    ft_model.save_pretrained(str(TRANS_DIR / "bertimbau_finetuned"))
    tokenizer.save_pretrained(str(TRANS_DIR / "bertimbau_finetuned"))
    print("Modelo fine-tunado salvo.")
    return trainer_ft


def tabela_comparativa_transformers(resultados: list) -> pd.DataFrame:
    """Monta, exibe e salva o comparativo dos modelos Transformer."""
    df   = pd.DataFrame(resultados).set_index("Modelo")
    disp = df.copy()
    for col in ["F1-Macro (media)", "F1-Weighted (media)", "Acuracia (media)"]:
        disp[col] = disp[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 60)
    print("COMPARATIVO — TRANSFORMERS (hold-out 15%)")
    print("=" * 60)

    df.to_csv(DATA_DIR / "resultados_transformers.csv")
    return df, disp


# ═════════════════════════════════════════════════════════════════════════════
# 6. SUBMISSÃO FINAL
# ═════════════════════════════════════════════════════════════════════════════

def consolidar_resultados():
    """
    Carrega os CSVs de resultados salvos pelas seções anteriores e os une
    em um único DataFrame ordenado por F1-macro.

    Retorna
    -------
    todos_resultados : pd.DataFrame indexado por Modelo.
    """
    keep_cols = ["F1-Macro (media)", "F1-Weighted (media)", "Acuracia (media)"]

    def _load(path):
        if Path(path).exists():
            df = pd.read_csv(path, index_col="Modelo")
            return df[[c for c in keep_cols if c in df.columns]]
        return pd.DataFrame(columns=keep_cols)

    frames = [
        _load(DATA_DIR / "resultados_classicos.csv"),
        _load(DATA_DIR / "resultados_profundos.csv"),
        _load(DATA_DIR / "resultados_transformers.csv"),
    ]
    frames = [f for f in frames if not f.empty]
    todos  = pd.concat(frames) if frames else pd.DataFrame(columns=keep_cols)

    print("Tabela consolidada de resultados:")
    return todos


def plotar_ranking_geral(todos_resultados: pd.DataFrame):
    """
    Gráfico de barras horizontais com todos os modelos rankeados por F1-macro.
    Os dois primeiros (top-2 para submissão) são destacados em vermelho.
    """
    ranking = todos_resultados.sort_values("F1-Macro (media)", ascending=False)
    cores   = ["#d62728" if i < 2 else "#aec7e8" for i in range(len(ranking))]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(
        ranking.index[::-1],
        ranking["F1-Macro (media)"].values[::-1],
        color=cores[::-1], alpha=0.85, edgecolor="white",
    )
    for bar, val in zip(bars, ranking["F1-Macro (media)"].values[::-1]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )
    ax.set_xlabel("F1-Macro")
    ax.set_title(
        "Ranking Geral de Modelos — F1-macro  (vermelho = top-2 para submissão)"
    )
    ax.set_xlim(0, min(1.0, ranking["F1-Macro (media)"].max() + 0.08))
    sns.despine()
    plt.tight_layout()
    plt.show()
    return ranking


def gerar_submissao_final(
    nome_arquivo, ranking, posicao,
    # Modelos clássicos
    X_test_tfidf=None, ids_teste=None,
    # Modelos profundos
    bilstm_model=None, cnn_model=None, X_seq_test=None,
    # Transformers
    trainer_lora=None, tok_ds=None,
):
    """
    Gera o CSV de submissão para o modelo na posição indicada do ranking.
    Roteia automaticamente para o tipo correto (clássico / neural / Transformer).

    Parâmetros
    ----------
    posicao : 0 = primeiro colocado, 1 = segundo, etc.
    """
    modelos_pkl = {
        "Complement NB (melhor alpha)"    : DATA_DIR / "modelos_classicos" / "complement_nb.pkl",
        "Regressão Logística (melhor C)"  : DATA_DIR / "modelos_classicos" / "logistic_regression.pkl",
        "SVM Linear/SGD (melhores params)": DATA_DIR / "modelos_classicos" / "svm_sgd.pkl",
    }

    nome_modelo = ranking.index[posicao]
    print(f"Gerando submissão {posicao + 1}: {nome_modelo}")

    if nome_modelo in modelos_pkl:
        with open(modelos_pkl[nome_modelo], "rb") as f:
            m = pickle.load(f)
        y_pred = m.predict(X_test_tfidf)

    elif "BiLSTM" in nome_modelo and bilstm_model is not None:
        y_pred = np.argmax(bilstm_model.predict(X_seq_test, verbose=0), axis=1)

    elif "CNN" in nome_modelo and cnn_model is not None:
        y_pred = np.argmax(cnn_model.predict(X_seq_test, verbose=0), axis=1)

    elif trainer_lora is not None:
        pred_out = trainer_lora.predict(tok_ds["test"])
        y_pred   = np.argmax(pred_out.predictions, axis=-1)

    else:
        raise ValueError(f"Não foi possível encontrar o modelo para: {nome_modelo}")

    sub     = pd.DataFrame({"Id": ids_teste, "Category": y_pred})
    caminho = DATA_DIR / nome_arquivo
    sub.to_csv(caminho, index=False)
    print(f"  {nome_arquivo} salvo  ({len(sub)} registros)")
    print("  Distribuição predita:")
    print(sub["Category"].value_counts().rename(index=CLASS_NAMES).to_string())
    return sub


def validar_submissao(nome_arquivo, sub_df):
    """
    Verifica se o CSV de submissão está no formato correto para o Kaggle:
    colunas Id/Category, número de registros, classes válidas (0–4) e IDs corretos.
    """
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    erros      = []

    if set(sub_df.columns) != set(sample_sub.columns):
        erros.append(f"Colunas incorretas: {sub_df.columns.tolist()}")
    if len(sub_df) != len(sample_sub):
        erros.append(f"Tamanho incorreto: {len(sub_df)} (esperado {len(sample_sub)})")
    classes_pred   = set(sub_df["Category"].unique())
    classes_validas = {0, 1, 2, 3, 4}
    if not classes_pred.issubset(classes_validas):
        erros.append(f"Classes inválidas: {classes_pred - classes_validas}")
    if not sub_df["Id"].equals(sample_sub["Id"]):
        erros.append("IDs não coincidem com o sample_submission!")

    if erros:
        print(f"[ERRO] {nome_arquivo}:")
        for e in erros:
            print(f"  - {e}")
    else:
        print(f"[OK] {nome_arquivo} — formato válido para submissão!")
        print(f"     {len(sub_df)} registros | classes: {sorted(classes_pred)}")


def exibir_resumo_final(ranking, nomes_arquivos):
    """
    Imprime o resumo das submissões geradas e salva o ranking completo em CSV.
    """
    print("=" * 65)
    print("RESUMO FINAL — ARQUIVOS PRONTOS PARA SUBMISSÃO")
    print("=" * 65)

    for i, arq in enumerate(nomes_arquivos):
        nome_modelo = ranking.index[i]
        f1          = ranking["F1-Macro (media)"].iloc[i]
        print(f"  {arq}")
        print(f"    Modelo  : {nome_modelo}")
        print(f"    F1-macro: {f1:.4f} (CV/hold-out)")
        print(f"    Caminho : {DATA_DIR / arq}")
        print()

    ranking_final = ranking.copy().reset_index()
    ranking_final.insert(0, "Pos", range(1, len(ranking_final) + 1))
    for col in ["F1-Macro (media)", "F1-Weighted (media)", "Acuracia (media)"]:
        if col in ranking_final.columns:
            ranking_final[col] = ranking_final[col].apply(lambda x: f"{x:.4f}")

    print("Ranking completo:")
    ranking_final_idx = ranking_final.set_index("Pos")

    ranking_final.to_csv(DATA_DIR / "ranking_final.csv", index=False)
    print(f"\nranking_final.csv salvo em {DATA_DIR}")
    return ranking_final_idx

def avaliar_modelo_nn(nome, model, X_val, y_val):
    """
    Calcula F1-macro, F1-weighted e Acurácia no conjunto de validação,
    imprime o classification_report e plota a matriz de confusão.

    Retorna
    -------
    dict com métricas para a tabela comparativa.
    """
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    f1m = f1_score(y_val, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    acc = (y_val == y_pred).mean()

    print(f"  {nome:<36} F1-macro={f1m:.4f}  F1-weighted={f1w:.4f}  acc={acc:.4f}")
    print(f"\nRelatório de classificação — {nome} (hold-out):")
    print(classification_report(y_val, y_pred,
                                target_names=CLASS_LABELS, zero_division=0))

    return {
        "Modelo"             : nome,
        "F1-Macro (media)"   : f1m,
        "F1-Weighted (media)": f1w,
        "Acuracia (media)"   : acc,
    }