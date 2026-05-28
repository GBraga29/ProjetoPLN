# Projeto Final PLN: Categorização Automática de Documentos Jurídicos

---

## 👥 Integrantes do Time
* **Eric Lopes** (RA: 822873)
* **Guilherme Braga** (RA: 823161)
* **Guilherme Saggion** (RA: 823159)

---

## 📌 Overview do Projeto

O objetivo deste projeto é desenvolver modelos preditivos utilizando técnicas de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina para classificar automaticamente documentos jurídicos extraídos de processos reais[cite: 10, 17]. O sistema judiciário lida com volumes imensos de dados, e a triagem automatizada é essencial para otimizar fluxos de trabalho[cite: 8].

Utilizaremos partições do conjunto VICTOR, uma base de milhares de recursos judiciais do Supremo Tribunal Federal (STF)[cite: 12, 13, 44]. O desafio central é lidar com as nuances do vocabulário jurídico, além dos ruídos severos gerados por extração OCR, carimbos, assinaturas e textos extremamente longos[cite: 18, 24, 26].

Os modelos devem classificar cada documento em uma das cinco categorias abaixo[cite: 25, 44]:

| Código | Categoria |
| :---: | :--- |
| **0** | Acórdão |
| **1** | Agravo de Recurso Extraordinário (ARE) |
| **2** | Despacho |
| **3** | Recurso Extraordinário (RE) |
| **4** | Sentença |

A implementação culminará em uma competição no Kaggle, visando o melhor desempenho preditivo no *Private Leaderboard*[cite: 53, 73, 96].

---

## Etapas de Desenvolvimento

O trabalho é conduzido de maneira modular no arquivo `main.ipynb` e em scripts de suporte `.py`[cite: 65, 66]. As etapas cruciais incluem:

### 1. Análise Exploratória de Dados (EDA)
* **Verificação de Integridade:** Busca e tratamento de valores nulos ou textos excessivamente curtos que indiquem falha de extração.
* **Análise de Distribuição:** Identificação de desbalanceamento de classes no arquivo `train.csv` para calibração de pesos no treinamento[cite: 50, 56].
* **Análise de Tamanho de Tokens:** Mensuração do tamanho médio e máximo dos textos para definir estratégias de truncamento em modelos de Deep Learning[cite: 56].
* **Levantamento de Vocabulário:** Extração de n-gramas e TF-IDF isolados por classe para identificar jargões discriminativos[cite: 56, 58].

### 2. Pré-processamento e Limpeza
* **Limpeza de Ruído Estrutural:** Uso de Expressões Regulares (RegEx) para remover quebras de linha errôneas, números de processos contínuos e cabeçalhos repetitivos[cite: 26, 33].
* **Stopwords Customizadas:** Criação de listas de remoção adaptadas ao domínio jurídico, preservando palavras de negação que invertem o sentido legal[cite: 26].
* **Tokenização e Lematização:** Aplicação de bibliotecas (como spaCy) para padronizar variações verbais e reduzir a dimensionalidade do vocabulário[cite: 26, 29].

### 3. Representação Computacional e Features
* **Features Linguísticas:** Extração de entidades nomeadas (NER), partes do discurso (POS) e contagem de palavras em caixa alta[cite: 35, 59].
* **Vetorização Esparsa:** Geração de representações Bag of Words e TF-IDF ajustadas com limites de frequência (min_df/max_df)[cite: 34, 58].
* **Embeddings Densos:** Utilização de modelos como Word2Vec ou FastText para capturar contexto semântico e mitigar erros de digitação e OCR[cite: 34, 58].

### 4. Experimentos e Treinamento de Modelos
* **Baseline Clássico:** Construção e treinamento de modelos iniciais utilizando algoritmos como Regressão Logística, Naïve Bayes ou SVM sobre atributos TF-IDF[cite: 36, 60].
* **Abordagem Profunda:** Teste de arquiteturas de Deep Learning, como Redes Convolucionais para texto (CNN) e Redes Recorrentes (BiLSTM)[cite: 37, 60].
* **Transformers (Estado da Arte):** Adaptação de Modelos de Linguagem para o português (ex: BERTimbau, Sabiá) com uso de PEFT/LoRA para viabilizar o treinamento em hardware limitado[cite: 38].

### 5. Análise de Resultados
* **Métricas de Avaliação:** Foco em métricas robustas ao desbalanceamento, prioritariamente o F1-Score[cite: 61].
* **Matriz de Confusão:** Avaliação visual dos maiores gargalos de classificação do modelo (ex: confusão entre Despachos e Sentenças).
* **Validação Cruzada:** Uso de técnicas estratificadas para garantir confiabilidade e significância estatística na comparação dos algoritmos[cite: 61].

### 6. Relatório e Submissão
* **Escrita Científica:** Documentação das decisões e resultados em um relatório de até 4 páginas (formato IEEE) utilizando LaTeX[cite: 75, 88].
* **Submissão Kaggle:** Inferência nos dados de `test.csv` e formatação das previsões baseadas no `sample_submission.csv`[cite: 51, 52].
* **Empacotamento:** Compilação do código reprodutível e relatório na estrutura `RAs-PROJ-PLN.zip` para envio no AVA[cite: 89, 109, 110].

---

## 🛠️ Sugestões de Tratamento para os Dados Específicos
* Os dados do STF extraídos via OCR e em formato longo trazem desafios crônicos. Aqui estão algumas abordagens altamente recomendadas para este projeto:

* Estratégia de Truncamento para Textos Longos: Modelos baseados em Transformers (como o BERT) têm um limite estrito de, geralmente, 512 tokens. Em documentos com centenas de páginas, o começo do documento costuma carregar o título e identificação, enquanto o final traz a assinatura/decisão. O artigo original que apresentou a base VICTOR identificou que utilizar apenas os primeiros 500 tokens do documento como entrada na rede produziu resultados rápidos e eficientes, economizando recursos de hardware. Considere usar Head+Tail truncation (ex: primeiros 128 tokens e últimos 384 tokens).

* Expressões Regulares (RegEx) para Ruído de Extração: Crie padrões de RegEx específicos para remover cabeçalhos de papel timbrado repetitivos, datas contínuas, números de processo (que não ajudam a generalizar o tipo de peça) e rodapés gerados pelos carimbos do processo judicial físico.

* Técnicas de Desbalanceamento: As classes "Sentença" e "Acórdão" podem aparecer em proporções muito diferentes em relação a "Despacho". Para evitar que o modelo preveja apenas a classe majoritária, utilize técnicas como class_weight='balanced' nativo do scikit-learn, ou métodos de subamostragem direcionada (undersampling inteligente).

* Uso de Modelos Específicos de Domínio: Em vez de embeddings genéricos, busque versões pré-treinadas no léxico jurídico brasileiro (como projetos focados no corpus do próprio STF, TCU, ou acórdãos públicos), o que ajuda o modelo computacional a não descartar o jargão do domínio.
---

📚 Materiais de Referência e Links Úteis
Para que possamos guiar nossos testes com uma base sólida de Legal NLP, os seguintes artigos de tarefas similares com este exato tipo de dado são vitais:

O Artigo Original do Dataset (Dataset Paper):

VICTOR: a Dataset for Brazilian Legal Documents Classification (LREC 2020).

Este é o documento fundador que construiu os dados que estamos usando. O artigo testou uma CNN em nível de palavras utilizando os primeiros 500 tokens de cada documento, o que os ajudou com questões de performance e memória da GPU. Link: Artigo VICTOR (PDF)

Revisões de Literatura no Domínio Jurídico:

Uma leitura interessante para justificar escolhas no Relatório é ver como outros trabalhos performaram na área jurídica. Exemplo: Text Classification in Law Area: a Systematic Review. Link: Artigo JIDM

Modelos de Linguagem para Português:

BERTimbau (Neuralmind): BERT pré-treinado no corpus BrWaC. Encontrado na biblioteca transformers da Hugging Face.

Ulysses Legal BERT: Se houver disponibilidade, modelos treinados pela UnB e Senado Federal focados em textos legais brasileiros são ainda melhores para reconhecimento deste vocabulário específico.