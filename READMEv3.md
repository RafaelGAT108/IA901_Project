# Classificação de Sons Pulmonares a partir de Espectrogramas

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA901 - Análise de Imagens e Reconhecimento de Padrões*, oferecida no primeiro semestre de 2026, na Unicamp, sob supervisão da Profa. Dra. Leticia Rittner, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

| Nome                    | RA     | Curso                                 |
|-------------------------|--------|---------------------------------------|
| Rafael Ávila dos Santos | 300905 | Doutorado em Engenharia Elétrica |
| Letícia Lopes Mendes Da Silva | 184423 | Graduação em Engenharia de Computação |
| Sofia Ballerini de Vasconcellos | 299904 | Doutorado em Engenharia Elétrica |

## Descrição do Projeto

O objetivo do projeto é utilizar sons pulmonares para classificação de doenças respiratórias, como asma, pneumonia, entre outras. Para isso, o sinal de áudio original é convertido em representações tempo-frequência (espectrogramas), como a Short-Time Fourier Transform (STFT), o Mel Spectrogram e os MFCCs, de forma a preservar simultaneamente as informações temporais e espectrais do sinal. Essas representações 2D são então utilizadas como entrada para redes neurais convolucionais (CNNs), treinadas para realizar a classificação multiclasse das patologias respiratórias. O projeto cobre, portanto, desde a obtenção e o pré-processamento dos dados brutos de áudio até o treinamento e avaliação dos classificadores.

## Bases de Dados

Os datasets públicos utilizados no projeto consistem em gravações de sons respiratórios obtidas em contexto clínico, acompanhadas de anotações e metadados associados. As bases foram escolhidas devido ao amplo uso em pesquisas de classificação automática de doenças pulmonares. A caracterização quantitativa apresentada a seguir foi obtida a partir da análise exploratória realizada em `2_audio_analysis.ipynb`.

Os áudios de ambas as bases estão disponíveis no formato `.wav`. As anotações, no entanto, são fornecidas em formatos distintos:

- **ICBHI 2017**: as anotações estão em arquivos de texto (`.txt`), sendo um deles responsável por relacionar o identificador de cada paciente ao respectivo diagnóstico (`ICBHI_Challenge_diagnosis.txt`), outro contendo informações demográficas (`ICBHI_Challenge_demographic_information.txt`), e um terceiro definindo a divisão oficial entre treino e teste proposta pelo desafio (`ICBHI_challenge_train_test.txt`).
- **KAUH (2021)**: as anotações estão concentradas em uma planilha Excel, contendo dados do paciente, informações sobre a aquisição do exame (modo de filtragem do estetoscópio, por exemplo) e o diagnóstico associado, quando houver.

### [ICBHI 2017](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

A base ICBHI 2017 é composta por **920 gravações respiratórias**, provenientes de **126 pacientes**, anotadas em **8 classes diagnósticas**. A tabela abaixo resume as principais características levantadas na análise exploratória:

| Característica | Valor |
|---|---|
| Total de gravações (amostras) | 920 |
| Pacientes únicos | 126 |
| Classes diagnósticas | 8 (COPD, Pneumonia, Healthy, URTI, Bronchiectasis, Bronchiolitis, LRTI, Asthma) |
| Classe majoritária | COPD — 793 amostras (≈ 86% da base) |
| Classe minoritária | Asthma — 1 amostra |
| Formato do áudio | `.wav` |
| Taxa de amostragem | Variável conforme o equipamento de aquisição (ex.: 44.100 Hz para o estetoscópio Meditron) |
| Duração mínima | 7,86 s |
| Duração máxima | 86,20 s |
| Duração média | 21,49 s |
| Tipo de anotação | Diagnóstico por paciente, dados demográficos (idade, sexo, IMC/peso/altura) e marcação manual de ciclos respiratórios, sibilos e estalos por especialistas |


### [KAUH (2021)](https://data.mendeley.com/datasets/jwyy9np4gv/3)

A base KAUH (2021), conforme efetivamente carregada e utilizada no pipeline, é composta por **324 gravações respiratórias**, anotadas em **8 classes diagnósticas** consideradas neste projeto (a publicação original reporta um total de 336 amostras provenientes de 112 pacientes; a diferença entre o total reportado na literatura e o número de amostras carregadas decorre do filtro de classes aplicado nesta análise). As principais características são resumidas a seguir:

| Característica | Valor |
|---|---|
| Total de gravações (amostras) | 324 |
| Pacientes (segundo a publicação original) | 112 |
| Classes diagnósticas consideradas | 8 (Healthy, Asthma, Heart Failure, COPD, Pneumonia, Lung Fibrosis, Bronchitis, Pleural Effusion) |
| Classe majoritária | Healthy — 105 amostras (≈ 32% da base) |
| Classe minoritária | Pleural Effusion — 6 amostras |
| Formato do áudio | `.wav` |
| Taxa de amostragem | 4.000 Hz (estetoscópio digital) |
| Duração mínima | 5,00 s |
| Duração máxima | 30,00 s |
| Tipo de anotação | Planilha Excel com dados do paciente, modo de filtragem acústica do estetoscópio, ponto de auscultação e diagnóstico associado |

### Base Combinada

Ao combinar as duas bases (1.244 amostras no total), o desbalanceamento entre classes se mantém como o principal desafio do projeto: a classe COPD passa a concentrar 820 amostras (≈ 66% do total combinado), enquanto classes como Bronchitis, Bronchiolitis, Lung Fibrosis, Pleural Effusion e LRTI somam poucas dezenas de amostras no total. Esse cenário motivou as estratégias de janelamento e data augmentation descritas na seção de Metodologia.

![combined_class_dist](assets/Label_Distribuition_by_Dataset.png)


## Metodologia

### Pré-processamento do Sinal de Áudio (1D)

Conforme identificado na análise exploratória, as gravações originais apresentam durações bastante heterogêneas (entre 7,86s e 86,20s, com média de 21,49s), além de forte desbalanceamento entre classes. A primeira abordagem testada para padronizar as durações consistiu em preencher os áudios mais curtos com zeros e truncar os mais longos, fixando todas as amostras em 20 segundos. Essa estratégia, entretanto, introduz um artefato: para os áudios mais curtos, surge uma região "vazia" no espectrograma resultante, que não carrega informação sobre a patologia, mas sim sobre o próprio pré-processamento.

Para contornar essa limitação, o pipeline atual substitui o zero-padding por uma estratégia de janelamento com sobreposição (*windowing*), aplicada da seguinte forma:

- **Reamostragem**: todos os sinais são reamostrados para uma taxa de amostragem comum de 22.050 Hz.
- **Janelamento**: cada gravação é dividida em janelas de 5 segundos. Para a maioria das classes, utiliza-se um *hop* de 2,5 segundos (50% de sobreposição), o que também funciona como uma forma de data augmentation, aumentando a quantidade de amostras disponíveis para as classes minoritárias. Para a classe COPD, utiliza-se um *hop* de 5 segundos (sem sobreposição), de forma a evitar a geração de janelas redundantes e, assim, reduzir o desbalanceamento entre classes.
- **Normalização**: cada janela de áudio é normalizada pela sua amplitude máxima.

Essa abordagem elimina o artefato de padding observado anteriormente e atua simultaneamente como mecanismo de data augmentation (para classes minoritárias) e de subamostragem (para a classe COPD).

### Extração de Características Tempo-Frequência (2D)

A partir do sinal 1D pré-processado, diferentes representações tempo-frequência são extraídas utilizando a biblioteca `Librosa`. Todas partem da Short-Time Fourier Transform (STFT), que divide o sinal em janelas curtas e calcula a Transformada de Fourier em cada uma:

$$X(m,k)=\sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$$

Sendo:

- x[n]: sinal de áudio
- w[n]: janela (geralmente Hann)
- N: tamanho da FFT
- H: hop length
- m: índice temporal
- k: bin de frequência

A partir do coeficiente complexo $X(m,k)$, são derivadas as oito representações efetivamente extraídas e utilizadas no treinamento dos classificadores (implementadas em `5_preprocess_features.ipynb`):

**Parte Real e Parte Imaginária da STFT (RealSTFT / ImagSTFT)**

$$\text{RealSTFT}(m,k) = \Re(X(m,k)) \qquad \text{ImagSTFT}(m,k) = \Im(X(m,k))$$

**Magnitude do Espectro (MagSTFT)**

A magnitude representa a energia/amplitude de cada frequência ao longo do tempo:

$$|X(m, k)| = \sqrt{\Re(X)^2 + \Im(X)^2}$$

**Fase Espectral (Phase)**

$$\phi(m, k) = \tan^{-1}\left(\frac{\Im(X(m, k))}{\Re(X(m, k))}\right)$$

**Mel Spectrogram**

Calculada em duas etapas. Primeiro o espectrograma de potência:

$$P(m, k) = |X(m, k)|^2$$

Depois aplica-se um banco de filtros Mel:

$$M(m,r)= \sum_k{H_r(k)P(m,k)}$$

Onde $H_r(k)$ é o filtro triangular Mel e r é o índice do filtro. Essa transformação comprime altas frequências para imitar a audição humana.

**Mel Frequency Cepstral Coefficients (MFCC)**

Aplica-se a Discrete Cosine Transform (DCT) sobre o log-Mel spectrogram:

$$L(m, r) = \log(M(m, r))$$

$$C(m, n) = \sum_k^R{}L(m, r)\cos\left[\frac{\pi n}{R} \left(r + \frac{1}{2}\right)\right]$$

Onde $C(m,n)$ é o coeficiente MFCC e R o número de filtros Mel.

**Delta MFCC (MFCCDelta)**

Derivada temporal dos coeficientes MFCC:

$$\Delta c_t= \frac{\sum_{n=1}^{N} n(c_{t+n}-c_{t-n})}{2\sum_{n=1}^{N} n^2}$$

**Chroma**

$$Chroma(p, t) = \sum_{k \in K_{p}}{|X(t, k)|}$$

Onde p é a classe cromática (C, C#, D, ...) e $K_p$ são os bins associados àquela nota.

Durante a etapa de análise exploratória das transformações (`3_transforms_analysis.ipynb`), outras representações também foram investigadas, como o Spectral Contrast e a Constant-Q Transform (CQT). Entretanto, o conjunto final de oito representações utilizado para a extração de features e o treinamento dos modelos é o listado acima.

**Combinações de canais (stacks)**

Além das oito representações individuais, também foram testadas combinações de 2 a 3 representações empilhadas em um único tensor multicanal, de forma análoga aos canais RGB de uma imagem natural. Os experimentos incluem as combinações: ImagSTFT + RealSTFT (2 canais), MagSTFT + Phase (2 canais), e MelSpectrogram + MFCC + Chroma (3 canais).

Após a extração, todas as representações (individuais e combinadas) são exportadas em arquivos `.npz`, formato binário do NumPy que permite armazenar múltiplos arrays de forma compacta, preservando a estrutura numérica das features e evitando custos adicionais de leitura/decodificação de imagem durante o treinamento dos modelos.

### Modelos de Classificação e Treinamento

A classificação é realizada utilizando três arquiteturas de CNN consagradas para classificação de imagens: **ResNet50**, **InceptionV3** e **DenseNet121**. O pipeline de treinamento foi implementado em PyTorch, com PyTorch Lightning estruturando o laço de treinamento.

Para cada arquitetura, o mesmo protocolo experimental é repetido para 9 configurações de features: as 6 representações individuais (MagSTFT, MelSpectrogram, MFCC, MFCCDelta, Chroma e Phase) e as 3 combinações de canais (ImagSTFT + RealSTFT, MagSTFT + Phase, e MelSpectrogram + MFCC + Chroma) descritas acima. RealSTFT e ImagSTFT, em particular, não são utilizadas isoladamente, apenas como parte da combinação de 2 canais. Isso totaliza 9 treinamentos por arquitetura, ao todo são realizados 10 treinamentos por arquitetura, pois a combinação MelSpectrogram + MFCC + Chroma é treinada duas vezes: uma vez do zero e outra reaproveitando a mesma configuração de features com pesos pré-treinados no ImageNet (`IMAGENET1K_V1`), permitindo comparar o efeito do pré-treino especificamente para essa combinação. Os principais hiperparâmetros utilizados são:

| Hiperparâmetro | Valor |
|---|---|
| Arquiteturas testadas | ResNet50, InceptionV3, DenseNet121 |
| Pesos pré-treinados | Sem pré-treino (`None`) na maioria dos experimentos; pesos ImageNet (`IMAGENET1K_V1`) utilizados apenas no treinamento adicional (retreino) da combinação MelSpectrogram + MFCC + Chroma, para comparação |
| Otimizador | AdamW |
| Taxa de aprendizado | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | StepLR (step_size=7, gamma=0.1) |
| Função de perda | CrossEntropyLoss |
| Batch size | 32 |
| Precisão de treinamento | Mixed precision (16-bit) |
| Épocas máximas | 100, com early stopping (paciência de 10 épocas, monitorando `val_f1_macro`) |
| Estratégia de amostragem | Sampler "equalizer" para balanceamento entre classes durante o treinamento, com limite de 1000 amostras por classe |
| Divisão dos dados | Holdout em treino/validação/teste, particionado por paciente para evitar que o mesmo paciente aparecesse em mais de um split (vazamento de dados), buscando também manter uma proporção semelhante de amostras por classe entre treino, validação e teste |
| Semente aleatória | 42, para garantir reprodutibilidade |
| Checkpoint do modelo | Melhor checkpoint salvo conforme `val_f1_macro` (modo max) |

O acompanhamento dos experimentos (curvas de perda, acurácia e demais métricas ao longo do treinamento) é feito via TensorBoard.

É importante destacar a diferença entre os dois mecanismos da tabela acima relacionados ao desbalanceamento de classes. A divisão por paciente em treino/validação/teste, com proporções semelhantes de amostras por classe entre os splits, não resolve o desbalanceamento em si. Esse processo apenas evita o vazamento de dados (mesmo paciente em mais de um split) e garante que esse desbalanceamento esteja igualmente refletido nos três conjuntos, sem favorecer artificialmente nenhum deles. Quem efetivamente trata o desbalanceamento entre classes **durante o treinamento** é o sampler "equalizer", que rebalanceia a frequência de amostragem das classes minoritárias e majoritárias a cada época, com limite de 1000 amostras por classe.

### Métricas de Avaliação

Como métricas de avaliação são utilizadas as métricas clássicas de classificação: acurácia, precision, recall e F1-Score, calculadas tanto de forma global quanto individualmente por classe. Para cada experimento (combinação de arquitetura + representação tempo-frequência), são gerados o relatório de classificação, curvas ROC e Precision-Recall por classe (estratégia one-vs-rest) e a matriz de confusão. Os resultados de cada experimento são salvos em arquivo CSV para análise posterior, incluindo rótulos verdadeiros, predições e probabilidades por classe.

## Ferramentas

Este projeto foi desenvolvido principalmente em Python, com a maior parte da exploração e dos experimentos documentados em notebooks.

- **Bibliotecas utilizadas:**
	- **NumPy:** operações numéricas e arrays.
	- **Librosa:** carregamento de áudio, transformações para o espaço 2D (STFT, Mel-spectrogram, MFCCs, etc.) e utilitários de áudio.
	- **SoundFile:** leitura e escrita dos arquivos de áudio pré-processados.
	- **Matplotlib / Seaborn:** visualização de espectrogramas, distribuições e gráficos.
	- **Pandas:** manipulação de tabelas, metadados e resultados.
	- **scikit-learn:** split de conjuntos, métricas e utilitários de avaliação.
	- **TensorFlow / Keras:** utilizado nos experimentos iniciais de treinamento de modelos de classificação (resultados preliminares).
	- **PyTorch / PyTorch Lightning:** utilizado no pipeline atual de treinamento, estruturando os módulos de dados (`DataModule`) e de modelo (`LightningModule`), bem como callbacks de checkpoint e early stopping.
	- **torchinfo:** sumarização da arquitetura e do número de parâmetros dos modelos.
	- **TensorBoard:** monitoramento dos experimentos, permitindo acompanhar métricas como perda, acurácia e curvas de treinamento entre diferentes execuções.

## Workflow

![workflow](assets/workflow.png)

## Experimentos e Resultados 

## Discussão


## Uso de IA Generativa

Utilizou-se IA generativa como apoio em comandos de markdown, ajustes de código e na reorganização e complementação da documentação deste README, a partir do conteúdo já presente nos notebooks do projeto.

## Referências

1. FRAIWAN, Mohammad; FRAIWAN, Luay; KHASSAWNEH, Basheer; IBNIAN, Ali.
   *A dataset of lung sounds recorded from the chest wall using an electronic stethoscope*.
   Data in Brief, v. 35, p. 106913, 2021.
   DOI: https://doi.org/10.1016/j.dib.2021.106913

2. ROCHA, Bruno M. et al.
   *An open access database for the evaluation of respiratory sound classification algorithms*.
   Physiological Measurement, v. 40, n. 3, p. 035001, 2019.
   DOI: https://doi.org/10.1088/1361-6579/ab03ea

3. WANASINGHE, Thinira; BANDARA, Sakuni; MADUSANKA, Supun; MEEDENIYA, Dulani Apeksha; BANDARA, Meelan; DE LA TORRE DÍEZ, Isabel.
   *Lung Sound Classification With Multi-Feature Integration Utilizing Lightweight CNN Model*.
   IEEE Access, v. 12, p. 21262–21276, 2024.

4. PARK, Jinho et al.
   *Lung Sound Classification Model for On-Device AI*.
   Applied Sciences, v. 15, n. 17, p. 9361, 2025.

5. MCFEE, Brian et al. librosa/librosa: 0.10. 0. zenodo, 2023.
