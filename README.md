# Classificação de Sons Pulmonares a partir de Espectrogramas

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA901 - Análise de Imagens e Reconhecimento de Padrões*, 
oferecida no primeiro semestre de 2026, na Unicamp, sob supervisão da Profa. Dra. Leticia Rittner, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Incluir nome RA e foco de especialização de cada membro do grupo. Os projetos devem ser desenvolvidos em duplas ou trios.

| Nome                    | RA     | Curso                                 |
|-------------------------|--------|---------------------------------------|
| Rafael Ávila dos Santos | 300905 | Doutorado em Engenharia Elétrica     |
| Letícia Lopes Mendes Da Silva  | 184423 | Graduação em Engenharia de Computação |
| Sofia Ballerini de Vasconcellos  | 299904 | Doutorado em                          |

## Descrição do Projeto

> Descrição do objetivo principal do projeto, incluindo contexto gerador, motivação, etc. Qual problema você pretende solucionar? Qual a relevância do problema e o impacto da solução do mesmo?

O objetivo final do projeto é utilizar sons pulmonares para classificação de doenças respiratórias, como asma, pneumonia, etc. Para isso, serão utilizadas técnicas de processamento de imagens para converter os sinais áudios originais em espectrogramas, como, por exemplo, a transformada curta de Fourier (stft), com a intenção de manter as informações em relação ao tempo e à frequência.

## Metodologia

> Proposta de metodologia incluindo especificação de quais técnicas pretende-se explorar. Espera-se que nesta entrega você já seja capaz de descrever de maneira mais específica (do que na Entrega 1) quais as técnicas a serem empregadas em cada etapa do projeto.

- Inicialmente, o sinal de áudio foi transformado para o domínio tempo-frequência utilizando a Short-Time Fourier Transform (STFT). A partir dessa representação, diferentes transformadas foram extraídas, incluindo espectrograma em escala logarítmica, Mel spectrogram, coeficientes cepstrais Mel-frequency (MFCC), derivadas temporais dos MFCCs (delta coefficients), características cromáticas (Chroma), contraste espectral, Constant-Q Transform (CQT) e fase espectral. As equações destas transformações são apresentadas abaixo:

### Short-Time Fourier Transform (STFT)

A STFT divide o sinal em janelas curtas e calcula a Transformada de Fourier em cada uma.

A equação é:

$$X(m,k)=\sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$$

Sendo:

- x[n]: sinal de áudio 
- w[n]: janela (geralmente Hann) 
- N: tamanho da FFT 
- H: hop length 
- m: índice temporal 
- k: bin de frequência

### Magnitude do Espectro
A magnitude representa a energia/amplitude de cada frequência ao longo do tempo. Ela é equacionada por:

$$|X(m, k)| = \sqrt{\Re(X)^2 \quad \text{+} \quad \Im(X)^2}$$

### Conversão para decibéis (Spectrogram dB)

A conversão para decibéis é util, tendo em vista que aproxima da percepção humana. Sua equação é dada por:

$$S_{dB} = 20\log_{10}{(\frac{S}{S_{ref}})}$$

Onde:

- S: magnitude espectral
- $S_{ref}$: referência (neste caso, o valor máximo).

### Mel Spectrogram

O resultado proveniente da transformada Mel Spectrogram se dá por duas etapas. Primeiro calcula-se o Espectrograma de Potência, conforme a equação:

$$
P(m, k) = |X(m, k)|^2
$$

Depois aplica-se um banco de filtros Mel:

$$
M(m,r)= \sum_k{H_r(k)P(m,k)}
$$

Onde:

- $H_r(k)$: filtro triangular Mel\
- r: índice do filtro Mel


Dessa forma, essa transformação comprime altas frequências para imitar a audição humana.

### Mel Frequency Cepstral Coefficients (MFCC)

Os MFCCs aplicam DCT no log-Mel spectrogram.

$$
L(m, r) = \log(M(m, r))
$$

Discrete Cosine Transform (DCT):
$$
C(m, n) = \sum_k^R{}L(m, r)\cos[\frac{\pi n}{R} (r + \frac{1}{2})]
$$

Onde:

- $C(m,n)$: coeficiente MFCC 
- R: número de filtros Mel

### Delta MFCC

$$
\Delta c_t= \frac{\sum_{n=1}^{N} n(c_{t+n}-c_{t-n})}{2\sum_{n=1}^{N} n^2}
$$

### Chroma
Essa transformação é equacionada por:

$$
Chroma(p, t) = \sum_{k \in K_{p}}{|X(t, k)|}
$$

Onde: p é a classe cromática (C, C#, D, ...) e $K_P$ é o bins associados àquela nota.

### Contrast

$$
SC_b(t) = 10\log_{10}(\frac{P_b(t)}{V_b(t)})
$$

Sendo $P_b(t)$ a média dos picos da banda e $V_b(t)$ a média dos vales da banda.

### Constant-Q Transform
$$
X(k,n)=
\frac{1}{N_k}
\sum_{m=0}^{N_k-1}
x[n-m],
w_k[m],
e^{-j2\pi Qm/N_k}
$$

$$
CQT_{dB} = 20\log(|X(k, n)|)
$$

### Fase Espectral
A fase espectral, por sua vez, é equacionada por:

$$
\phi(m, k) = \tan^{-1}(\frac{\Re(X(m, k))}{\Im(X(m, k))})
$$

Como métricas de avaliação será utilizado a acurácia do modelo, além das outras métricas como F1-Score, Recall e Precision tanto no aspecto de classificação global quanto individual por cada classe.
Para os resultados preliminares, o dataset será avaliado em três etapas: A primeira avaliação se dará utilizando o dataset ICBHI2017 com os audios de 20 segundos de duração. No segundo momento, esse dataset será complementado com amostras do dataset KAUH para as classes Asma, Saudável e Pneumonia. Esse procedimento foi feito manualmente levando em consideração as labels presentes no próprio nome dos audios, indicando a qual classe eles pertenciam. Por fim, a terceira etapa consiste em realizar as transformações no dataset mesclado, porém, para cada 6 segundo dos audios, fazendo assim o data-augmentation.
O treinamento se dará via Transfer Learning com a rede InceptionV3, treinada por 10 épocas e com a divisão Holdout 80% para treinamento e 20% para validação.

## Bases de Dados e Evolução

Os datasets públicos utilizados no projeto consistem em gravações de sons respiratórios obtidas em contexto clínico, acompanhadas de anotações e metadados associados. As bases foram escolhidas devido ao amplo uso em pesquisas de classificação automática de doenças pulmonares.


| Base de Dados | Endereço na Web | Resumo descritivo |
|-----|-----|-----|
| ICBHI Challenge (2017) | https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge | Dataset utilizado no ICBHI 2017 Respiratory Sound Challenge. Contém gravações respiratórias de pacientes, anotadas com 8 doenças pulmonares, além de anotações de ciclos respiratórios, sibilos e estalos realizadas por especialistas. Contém 920 amostras de 126 pacientes. |
|KAUH (2021) Respiratory Sound Dataset | https://data.mendeley.com/datasets/jwyy9np4gv/3 | Base composta por gravações respiratórias obtidas com estetoscópio digital e disponibilizadas em diferentes modos de filtragem acústica, e anotados com 11 doenças cardiopulmonares. Contém 336 amostras de 112 pacientes. |

Datasheet: [Datasheet (PDF)](data/datasheet.pdf)

## Ferramentas

Este projeto foi desenvolvido principalmente em Python, com a maior parte da exploração e dos experimentos documentados em notebooks. A seguir listamos as principais bibliotecas e arquivos de apoio que foram utilizados para pré-processamento de áudio, geração de espectrogramas, visualização e experimentos de classificação. 

- **Bibliotecas utilizadas:**
	- **NumPy:** operações numéricas e arrays.
	- **Librosa:** carregamento de áudio, transformações para o espaço 2D (STFT, Mel-spectrogram, MFCCs, etc.), e utilitários de áudio.
	- **Matplotlib:** visualização de espectrogramas e gráficos.
	- **Seaborn:** visualizações estatísticas e suporte estético aos gráficos.
	- **Pandas:** manipulação de tabelas, metadados e resultados.
	- **scikit-learn:** split de conjuntos, métricas e utilitários de avaliação.
	- **TensorFlow / Keras:** usado em experimentos de treinamento de modelos de classificação.

- **Ferramentas ainda a serem utilizadas:**
Pretende-se migrar parte dos experimentos para o ecossistema **PyTorch**, principalmente devido à maior flexibilidade para construção de pipelines experimentais, melhor integração com tarefas de deep learning e maior controle sobre etapas de treinamento.

Também pretende-se utilizar o **TensorBoard** para monitoramento dos experimentos, permitindo acompanhar métricas como perda, acurácia, curvas de treinamento e comparação entre diferentes execuções.

## Workflow

> Use uma ferramenta que permita desenhar o workflow e salvá-lo como uma imagem (Draw.io, por exemplo). Insira a imagem nesta seção.
> Você pode optar por usar um gerenciador de workflow (Sacred, Pachyderm, etc) e nesse caso use o gerenciador para gerar uma figura para você.
> Lembre-se que o objetivo de desenhar o workflow é ajudar a quem quiser reproduzir seus experimentos.
> Mais informações sobre o workflow podem ser encontradas nos materiais de apoio no Classroom (Reprodutibilidade em pesquisa computacional - workflow).

![workflow](workflow.png)




## Experimentos e Resultados preliminares

> Descreva de forma sucinta e organizada os experimentos realizados.
> Para cada experimento, apresente os principais resultados obtidos.
> Aponte os problemas encontrados nas soluções testadas até aqui.

> ### Análise Exploratória
- Esta etapa do projeto consistiu na realização de uma análise exploratória da distribuição das durações dos sinais de áudio respiratório.
O script percorre todas as pastas correspondentes às doenças respiratórias no dataset, identifica os arquivos .wav e calcula a duração de cada gravação utilizando a biblioteca Librosa. As durações e respectivas classes são armazenadas para posterior análise estatística. Foram analisadas 920 amostras de áudio:

| Métrica | Valor |
|---|---|
| Duração mínima | 7,86 s |
| Duração máxima | 86,20 s |
| Duração média | 21,49 s |

O histograma a seguir ilusta distribuição das durações dos sinais de áudio, evidenciando uma concentração de amostras com duração de 20 segundos. No entanto, observa-se a existência de gravações com durações discrepantes, alcançando valores significativamente inferiores e superiores à média do conjunto de dados. Essa heterogeneidade temporal pode dificultar o processo de treinamento dos modelos de aprendizado profundo, devido à inconsistência no comprimento das entradas. Como possível abordagem para minimizar esse problema, está sendo considerada a utilização apenas das amostras com durações próximas ao valor predominante observado no dataset, uma vez que essas representam a maior parte das gravações disponíveis. Alternativamente, também está em análise a aplicação de técnicas de padronização temporal durante o pré-processamento dos sinais.
<img src="results/distribuição de duração dos audios.png" width="700">

Adicionalmente, foram feitas análises individuais para cada classe do dataset, permitindo uma análise mais detalhada da distribuição das durações entre as diferentes patologias respiratórias. A análise individual das classes permitiu observar diferenças importantes tanto na quantidade de amostras quanto na distribuição das durações dos sinais de áudio.
Para a classe Asthma (Asma), observa-se apenas uma amostra, com duração concentrada em aproximadamente 20 segundos. De forma semelhante, as classes Bronchiectasis (Bronquiectasia) e Bronchiolitis (Bronquiolite) apresentam, respectivamente, 16 e 13 amostras, todas concentradas na mesma duração.
A classe COPD (Doença Pulmonar Obstrutiva Crônica) apresenta um comportamento significativamente diferente das demais, contendo mais de 700 amostras. Apesar de existir uma variação nas durações, aproximadamente entre 10 e 30 segundos, nota-se uma predominância expressiva de sinais próximos de 20 segundos, representando a maior parte das amostras dessa classe.
Para a classe Healthy (Saudável), observa-se aproximadamente 30 amostras, também concentradas próximas de 20 segundos, com pequenas variações temporais inferiores a 0,2 segundos. Já a classe LRTI (Infecção do Trato Respiratório Superior) apresenta apenas duas amostras, ambas próximas de 20 segundos.
Na classe Pneumonia (Pneumonia), observa-se cerca de 40 amostras concentradas em torno de 20 segundos de duração. Por fim, a classe URTI (Infecção do Trato Respiratório Inferior) apresenta aproximadamente 25 amostras próximas de 20 segundos, além de uma amostra com pequeno desvio temporal, em torno de 19,85 segundos.
A partir dessa análise, é possível observar não apenas diferenças nas durações dos sinais entre algumas amostras, mas principalmente um forte desbalanceamento entre as classes do conjunto de dados. Enquanto determinadas categorias possuem poucas amostras disponíveis, a classe COPD concentra a maior parte do dataset. Esse desbalanceamento pode impactar diretamente o treinamento dos modelos de classificação, favorecendo classes majoritárias e dificultando a generalização para classes com menor representatividade. Dessa forma, torna-se importante considerar estratégias de balanceamento ou seleção criteriosa das amostras durante as etapas posteriores do projeto.

Com o objetivo de garantir consistência entre as amostras, os sinais foram padronizados para uma taxa de amostragem de 18 kHz e duração fixa de 20 segundos. O pipeline de pré-processamento inclui:

-reamostragem dos sinais para 18 kHz;
-preenchimento com zeros para áudios com duração inferior ao alvo;
-truncamento para áudios com duração superior ao limite definido;
-normalização pela amplitude máxima.

Essa padronização reduz a variabilidade estrutural entre as amostras e viabiliza a extração consistente de características acústicas.

Após o pré-processamento, diferentes representações espectrais e temporais foram extraídas utilizando funções da biblioteca `Librosa`, incluindo:

- Espectrograma em dB;
- Mel Spectrogram;
- MFCC;
- MFCC Delta;
- Chroma STFT;
- Spectral Contrast;
- Constant-Q Transform (CQT);
- Espectrograma de Fase.

Entre as representações extraídas, o **Mel Spectrogram** foi utilizado para visualização qualitativa dos padrões espectrais presentes nos sinais respiratórios.

  <img src="results/mel_spectrogram.png" width="900">

A análise do Mel Spectrogram evidencia diferenças entre os padrões respiratórios das patologias e o sinal considerado saudável. No caso do sinal Healthy (Saudável), observa-se uma distribuição espectral homogênea e contínua, com predominância de energia concentrada nas baixas frequências, comportamento esperado em ciclos respiratórios fisiológicos. Há menor presença de eventos impulsivos e menor variabilidade temporal, indicando estabilidade do fluxo aéreo pulmonar. Já nos sinais patológicos, percebe-se aumento da complexidade espectral e mudanças importantes na distribuição temporal da energia:

-Asthma apresenta regiões intermitentes de energia distribuídas em faixas específicas de frequência, compatíveis com a presença de sibilos. 
-Bronchiectasis mostra um padrão bastante energético e repetitivo ao longo do tempo, com estruturas verticais recorrentes e ampla ocupação espectral. 
-Bronchiolitis apresenta distribuição espectral mais difusa, com maior densidade de componentes em médias frequências. 
-LRTI exibe eventos esparsos de energia intensa, indicando comportamento acústico menos regular que o saudável. 
-Pneumonia apresenta regiões isoladas de alta intensidade espectral, além de maior espalhamento energético em frequências médias e baixas. 
-URTI mostra aumento moderado da atividade espectral e maior irregularidade temporal em comparação ao saudável, embora com menor complexidade que patologias pulmonais mais severas.

No caso do COPD, observa-se um comportamento diferente dos demais. Parte do espectrograma contém atividade espectral normal do sinal respiratório, enquanto uma grande região escura aparece no restante da imagem. Isso ocorre devido ao zero padding aplicado durante o pré-processamento para padronizar todos os áudios em 20 segundos. Portanto, parte desse comportamento visual não está associada diretamente à doença, mas sim ao processo de padronização aplicado no dataset. Mesmo assim, na região correspondente ao áudio real, o COPD ainda apresenta um padrão mais irregular e fragmentado em comparação ao saudável, o que está de acordo com alterações respiratórias típicas da doença.


> ### Resultados Preliminares

Realizando as etapas descritas na Metodologia e no Workflow, obteu-se os resultados das métricas de avaliação para as 8 transformações analisadas, conforme apresentadas nas tabelas abaixo (Para Download, as Tabelas completas e em gráfico exemplos das tabelas parciais):

| feature    | class          |   precision |   recall |   f1_score |   support |   accuracy_global |   precision_global |   recall_global |   f1_global |
|:-----------|:---------------|------------:|---------:|-----------:|----------:|------------------:|-------------------:|----------------:|------------:|
| phase      | Bronchiectasis |    1        | 0.333333 |   0.5      |         3 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |
| phase      | Bronchiolitis  |    0        | 0        |   0        |         3 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |
| phase      | COPD           |    0.856287 | 1        |   0.922581 |       143 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |
| phase      | Healthy        |    0        | 0        |   0        |         7 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |
| phase      | Pneumonia      |    0        | 0        |   0        |         7 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |
| phase      | URTI           |    0        | 0        |   0        |         5 |          0.857143 |           0.746721 |        0.857143 |    0.79422  |


[Download results_ibchi.xlsx](results/results_icbhi.xlsx)

| feature            | class          | precision | recall | f1_score | support | accuracy_global | precision_global | recall_global | f1_global |
| ------------------ | -------------- | --------- | ------ | -------- | ------- | --------------- | ---------------- | ------------- | --------- |
| phase              | Asthma         | 1.0000    | 0.5000 | 0.6667   | 4       | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | Bronchiectasis | 1.0000    | 1.0000 | 1.0000   | 3       | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | Bronchiolitis  | 0.0000    | 0.0000 | 0.0000   | 3       | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | COPD           | 0.8854    | 0.9720 | 0.9267   | 143     | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | Healthy        | 0.4167    | 0.4167 | 0.4167   | 12      | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | Pneumonia      | 0.0000    | 0.0000 | 0.0000   | 7       | 0.8418          | 0.7831           | 0.8418        | 0.8089    |
| phase              | URTI           | 0.0000    | 0.0000 | 0.0000   | 5       | 0.8418          | 0.7831           | 0.8418        | 0.8089    |

[Download results_mixed.xlsx](results/results_mixed.xlsx)


| feature     | class          | precision | recall | f1_score | support | accuracy_global | precision_global | recall_global | f1_global |
| ----------- | -------------- | --------- | ------ | -------- | ------- | --------------- | ---------------- | ------------- | --------- |
| phase       | Asthma         | 0.3864    | 0.3542 | 0.3696   | 48      | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | Bronchiectasis | 0.0000    | 0.0000 | 0.0000   | 9       | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | Bronchiolitis  | 0.0000    | 0.0000 | 0.0000   | 8       | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | COPD           | 0.7822    | 0.9950 | 0.8758   | 397     | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | Healthy        | 0.4242    | 0.1867 | 0.2593   | 75      | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | Pneumonia      | 0.0000    | 0.0000 | 0.0000   | 16      | 0.7320          | 0.6201           | 0.7320        | 0.6613    |
| phase       | URTI           | 0.0000    | 0.0000 | 0.0000   | 22      | 0.7320          | 0.6201           | 0.7320        | 0.6613    |

[Download results_6sec.xlsx](results/results_6sec.xlsx)

Como pode-se observar na figura abaixo, a quantidade de amostras para a validação da classe COPD é muito superior as demais. Como a divisão entre treino e validação foi feito na proporção 80%-20%, isso implica que durante o treinamento o modelo também lida com muito mais dados da classe COPD do que de quaisquer outra classe, o que pode tornar o modelo enviesado a aprender apenas ela. Essa afirmação se fortalece ao analisar as métricas de avaliação por classe, conforme disposto nas Tabelas. Embora na média global o modelo se encontre com acurária em torno de 80% para as 8 transformações avaliadas, ao analisar os valores por classe vemos uma diferença substancial, e que predomina com valores equivalente apenas para classe de maior quantidade de amostras.
![confusion_icbhi](results/confusion_icbhi.png)

Após adicionar mais amostras para as classes Asma, Pneumonia e saudável, a desigualdade amostral ainda continuou altamente presente. Com isso, o mesmo comportamente tendencioso para a classe COPD visto anteriormente se repete aqui, tanto ao analisar a matriz de confusão quanto observado nas métricas de avaliação por classe, dado a tabela.
![confusion_mixed](results/confusion_mixed.png)

Ao realizar o data-augmentation fazendo o clip dos audios em 6 segundos ao invés de mantê-los em 20 segundos como nos casos anteriores, observa-se alguns impactos significativos. Embora a desigualdade amostral permaneça, agora é possível observar que houve diferenças dos resultados entre as 8 transformações analisadas, ao invés de todas se manterem em torno de 80% de acurácia como nos casos anteriores.
![confusion_6sec](results/confusion_6sec.png)

Sendo assim, é possível resumir os resultados como:

- Os resultados preliminares são baseados nos resultados obtidos a partir do treinamento de uma rede baseada na InceptionV3 via Transfer Learning. Os primeiros resultados dão-se utilizando como banco dados apenas as amostras do dataset nomeado ICBHI2017 e com as visualizações que partiram dos audios em 20 segundos. Posteriormente acrescentou-se amostras do dataset KAUH para as classes Saudável, Asma e Pneumonia, ainda com os audios em 20 segundos. O terceiro grupo de resultados deu-se utilizando agora o dataset composto pelos audios do ICBHI2017 e do KAUH, porém, sob a visualização dos clips em 6 segundos, o que aumentou consideravelmente o número de amostras por classe. A classe LRTI como apresentou poucas amostras, foi desconsiderada para o treinamento.
- É possível observar a partir dos resultados obtidos para o caso 1 e o caso 2 que não houve discrepância significativa entre os resultados obtidos a partir das 8 transformações do sinal de audio. Todavia, o resultado, que se apresenta em torno de 80% de acurácia em todos os casos, é enviesado pela alta quantidade de amostras da classe COPD. As outras métricas, principalmente analisadas individualmente por cada classe, reforçam isso.
- Destaca-se também a visualização "phase", onde se esperava que ela não seria relevante e não conteria informações que fosse possível caracterizar e diferenciar as doenças pulmonares, em contrapartida, o modelo de aprendizado de máquina foi capaz de utilizá-la e atingir resultados similares as demais análises.
- Por outro lado, os resultados obtidos a partir das visualizações com um conjunto maior de amostras começa-se a perceber variações dos resultados entre as formas de visualizações, demonstrando que algumas têm maior capacidade de caracterização das doenças do que outras.

## Próximos passos

Nas etapas até aqui as transformações do sinal original de audio (MFCC, melspectrogram, etc.) tem se mostrado eficazes para o uso posterior em aprendizado de máquina e classificação das doenças pulmonares. O principal gargalo encontrado até o momento é o desbalanceamento das amostras por classe, o que torna o classificador enviesado. Sendo assim, um dos nossos próximos passos é justamente buscar alternativas para lidar com esse problema. Para além, também buscaremos comparar os resultados utilizando outros algoritmos de aprendizado de máquina.

## Uso de IA Generativa

Utilizou-se a IA para auxílio em comandos do markdown, bem como em ajustes de código.

## Referências

1. Rocha BM et al. (2019) "An open access database for the evaluation of respiratory sound classification algorithms" Physiological Measurement 40 035001
2. PARK, Jinho et al. Lung Sound Classification Model for On-Device AI. Applied Sciences, v. 15, n. 17, p. 9361, 2025.
3. MCFEE, Brian et al. librosa/librosa: 0.10. 0. zenodo, 2023.
