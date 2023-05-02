# AVC Predictor

O código principal do projeto se encontra no arquivo **main.ipynb**

## Descrição do projeto
Neste projeto, vamos usar classificadores para identificar *quais são os fatores de risco para o acidente vascular cerebral (AVC)*.

Temos à nossa disposição um conjunto de dados para [predição de AVCs](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). O que faremos é:

1. Treinar dois classificadores para predizer se houve ou não houve AVCs.
2. Verificar a acurácia dos classificadores.
3. Identificar quais são os fatores que mais provavelmente estão ligados a ter AVCs, dados pelos modelos. E conferir se de fato são reconhecidos no meio acadêmico.


## Resumo do projeto

### Metodologia utilizada

**Pré-processamento dos dados**: o conjunto de dados é tratado para garantir que esteja em um formato adequado para a análise. Isso inclui:

- Remoção de valores nulos (NaNs)
- Remoção de colunas que poderiam causar viés no modelo
- Conversão de variáveis categóricas em variáveis numéricas

<br>

**Definição da função de perda**: é definida uma função de perda que mede a diferença entre a previsão do modelo e o valor real da variável de resposta. A função de perda é uma função de quatro parâmetros: os pesos (w), o viés (b), os pontos (ou dados) e os valores reais da variável de resposta e retorna o erro quadrático médio (EQM).

<br>

```python
def loss( parametros ):
    w, b, pontos, real_value = parametros
    prediction = w.T @ pontos + b
    eqm = np_.mean( (prediction - real_value)**2)
    return eqm
```

<br>

**Treinamento do modelo**: o modelo é treinado usando o algoritmo de gradiente descendente para minimizar a função de perda. A cada iteração, o gradiente da função de perda em relação aos parâmetros é calculado e os pesos w e o viés b são atualizados na direção oposta do gradiente, para que assim eles sejam ajustados para minimizar a função de perda.

<br>

**Teste do modelo**: o conjunto de dados é dividido em um conjunto de treinamento e um conjunto de teste. O modelo treinado é testado no conjunto de teste para avaliar sua precisão na previsão da variável de resposta e a acurácia do modelo é medida.

<br>

### Métodos de classificação utilizados

- **Classificador linear:** método que funciona através da atribuição de pesos (ou coeficientes) às features do conjunto de dados:

<br>

$$
f(x,y) = Ax + By + C
$$

<br>

Sendo que **x** e **y** são as features do conjunto de dados, **A** e **B** são os pesos e **C** é o viés.

<br>

- **Classificador de árvore de decisão:** método que funciona através da criação de uma árvore de decisão, onde cada nó representa uma feature do conjunto de dados e cada ramo representa um valor dessa feature.

<br>

## Principais resultados obtidos

De acordo com os resultados obtidos, os 3 fatores que mais influenciam na probabilidade de uma pessoa ter um AVC são:

- Doença cardíaca
- Hipertensão
- Ser do gênero masculino

<br>

De acordo com o site da **Sociedade Brasileira de AVC** (link no final do arquivo), dentre os fatores que são listados como potenciais causadores de AVC, se encontram os mesmos fatores que foram identificados pelo modelo:

- Doença cardíaca:

    - "As doenças do coração, especialmente as arritmias (batimentos cardíacos desregulados), aumentam o risco de AVC. A arritmia mais comum é a fibrilação atrial, que provoca batimentos irregulares no coração e facilita a formação de coágulos sanguíneos, que podem migrar para os vasos do cérebro, causando um AVC."

- Hipertensão:

    - "Conhecida como “pressão alta”, é um dos principais, senão o principal fator de risco facilmente modificável para se evitar o AVC."

- Ser do gênero masculino:

    - "Pessoas do sexo masculino e a raça negra exibem maior tendência ao desenvolvimento de AVC."


<br>

Desta maneira, pode-se concluir que **o modelo foi capaz de identificar os fatores de risco para AVCs de maneira satisfatória**, já que os fatores identificados pelo modelo como sendo os mais influenciam a probabilidade de uma pessoa ter um AVC são os mesmos que são listados como potenciais causadores de AVCs no site da **Sociedade Brasileira de AVC**.

<br>

## Referências

- [Sociedade Brasileira de AVC](https://avc.org.br/pacientes/fatores-de-risco-para-o-avc/)

- [Dataset utilizado](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Autores

- [Leonardo Scarlato](https://github.com/leoscarlato)
- [Tomás Alessi](https://github.com/alessitomas)
