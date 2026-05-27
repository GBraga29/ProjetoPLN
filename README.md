#  Projeto Final PLN: Categorização Automática de Documentos Jurídicos 

---

## 👥 Integrantes do Time

* **Eric Lopes** (RA: 822873)
* **Guilherme Braga** (RA: 823161)
* **Guilherme Saggion** (RA: 823159)

---

## 📌 Overview do Projeto

O objetivo deste projeto é desenvolver modelos preditivos utilizando técnicas de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina para classificar automaticamente documentos jurídicos extraídos de processos reais. O sistema judiciário lida com volumes imensos de dados e a triagem automatizada é essencial para otimizar fluxos de trabalho na área de *Legal Tech*.

Utilizaremos a **base de dados VICTOR**, um conjunto de milhares de recursos judiciais do Supremo Tribunal Federal (STF). O desafio central é lidar com as nuances do vocabulário jurídico ("juridiquês"), além dos ruídos severos gerados por extração OCR, carimbos, assinaturas e textos extremamente longos.

Os modelos devem classificar cada documento em uma de **cinco categorias**:

1. Acórdão (0)
2. Agravo de Recurso Extraordinário - ARE (1)
3. Despacho (2)
4. Recurso Extraordinário - RE (3)
5. Sentença (4)

A implementação culminará em uma competição no Kaggle, visando o melhor desempenho preditivo no *Private Leaderboard*.

---

## 🚀 Etapas de Desenvolvimento

O trabalho deve ser conduzido de maneira modular e bem fundamentada, estruturado no arquivo `main.ipynb` e em scripts de suporte `.py`. Abaixo estão os passos cruciais:

### 1. Análise Exploratória de Dados (EDA)

* Carregar os dados de treino (`train.csv`) e teste (`test.csv`).
* Analisar a distribuição das classes para identificar desbalanceamentos (que são muito comuns em bases jurídicas).
* Mensurar o tamanho médio e máximo dos textos (contagem de palavras/tokens).
* Levantar o vocabulário mais frequente por classe para criar hipóteses sobre palavras-chave discriminativas.

### 2. Pré-processamento e Limpeza

* Implementar rotinas para limpar erros de OCR (caracteres estranhos, quebras de linha arbitrárias no meio das palavras).
* Normalizar textos (caixa baixa, remoção de acentos), aplicar tokenização rigorosa e remoção de *stop words* (mantendo termos jurídicos relevantes).
* Lematização focada no português estruturado do meio jurídico.

### 3. Representação Computacional e Features

* Extrair entidades nomeadas (NER), partes do discurso (POS) e outras tarefas estruturais relevantes.
* Gerar representações vetoriais esparsas (Bag of Words, TF-IDF com n-gramas).
* Extrair *embeddings* densos (Word2Vec, FastText) para capturar o contexto semântico.

### 4. Experimentos e Treinamento de Modelos

* **Abordagem Clássica:** Engenharia de atributos alimentando algoritmos como Regressão Logística, Naïve Bayes ou Máquinas de Vetores de Suporte (SVM).
* **Abordagem Profunda (Deep Learning):** Testar arquiteturas como Redes Convolucionais para texto (CNN) e Redes Recorrentes (como BiLSTM).
* **Transformers (Estado da Arte):** Empregar Modelos de Linguagem grandes pré-treinados, com foco nos adaptados ao português (ex: BERTimbau, Sabiá) ou adaptação via LoRA/PEFT para superar limitações de hardware.

### 5. Análise de Resultados

* Utilizar métricas de avaliação que penalizem de forma correta o desbalanceamento, com foco principal no **F1-Score (Macro)**.
* Gerar matrizes de confusão para entender entre quais tipos de documentos os modelos têm mais dificuldade em diferenciar (e.g., confundir "Despacho" com "Sentença").
* Documentar rigorosamente as configurações e comparar os algoritmos através de testes de significância estatística.

### 6. Relatório e Submissão

* Estruturar o relatório de, no máximo, 4 páginas (coluna dupla) em formato IEEE usando LaTeX.
* Submeter os resultados de teste para o Kaggle (formato `sample_submission.csv`).
* Compilar a solução reproduzível em um arquivo `.zip` seguindo a nomenclatura `RAs-PROJ-PLN.zip` para o AVA.

---

## 🛠️ Sugestões de Tratamento para os Dados Específicos

Os dados do STF extraídos via OCR e em formato longo trazem desafios crônicos. Aqui estão algumas abordagens altamente recomendadas para este projeto:

* **Estratégia de Truncamento para Textos Longos:** Modelos baseados em Transformers (como o BERT) têm um limite estrito de, geralmente, 512 tokens. Em documentos com centenas de páginas, o começo do documento costuma carregar o título e identificação, enquanto o final traz a assinatura/decisão. O artigo original que apresentou a base VICTOR identificou que utilizar apenas os primeiros 500 tokens do documento como entrada na rede produziu resultados rápidos e eficientes, economizando recursos de hardware. Considere usar *Head+Tail truncation* (ex: primeiros 128 tokens e últimos 384 tokens).
* **Expressões Regulares (RegEx) para Ruído de Extração:** Crie padrões de RegEx específicos para remover cabeçalhos de papel timbrado repetitivos, datas contínuas, números de processo (que não ajudam a generalizar o tipo de peça) e rodapés gerados pelos carimbos do processo judicial físico.
* **Técnicas de Desbalanceamento:** As classes "Sentença" e "Acórdão" podem aparecer em proporções muito diferentes em relação a "Despacho". Para evitar que o modelo preveja apenas a classe majoritária, utilize técnicas como `class_weight='balanced'` nativo do `scikit-learn`, ou métodos de subamostragem direcionada (*undersampling* inteligente).
* **Uso de Modelos Específicos de Domínio:** Em vez de *embeddings* genéricos, busque versões pré-treinadas no léxico jurídico brasileiro (como projetos focados no corpus do próprio STF, TCU, ou acórdãos públicos), o que ajuda o modelo computacional a não descartar o jargão do domínio.

---

## 📚 Materiais de Referência e Links Úteis

Para que possamos guiar nossos testes com uma base sólida de *Legal NLP*, os seguintes artigos de tarefas similares com este exato tipo de dado são vitais:

1. **O Artigo Original do Dataset (Dataset Paper):**
* *VICTOR: a Dataset for Brazilian Legal Documents Classification* (LREC 2020).
* Este é o documento fundador que construiu os dados que estamos usando. O artigo testou uma CNN em nível de palavras utilizando os primeiros 500 tokens de cada documento, o que os ajudou com questões de performance e memória da GPU. Link: [Artigo VICTOR (PDF)](https://aclanthology.org/2020.lrec-1.181.pdf)


2. **Revisões de Literatura no Domínio Jurídico:**
* Uma leitura interessante para justificar escolhas no Relatório é ver como outros trabalhos performaram na área jurídica. Exemplo: *Text Classification in Law Area: a Systematic Review*. Link: [Artigo JIDM](https://journals-sol.sbc.org.br/index.php/jidm/article/download/2547/2168/12669)


3. **Modelos de Linguagem para Português:**
* **BERTimbau (Neuralmind):** BERT pré-treinado no corpus BrWaC. Encontrado na biblioteca `transformers` da Hugging Face.
* **Ulysses Legal BERT:** Se houver disponibilidade, modelos treinados pela UnB e Senado Federal focados em textos legais brasileiros são ainda melhores para reconhecimento deste vocabulário específico.