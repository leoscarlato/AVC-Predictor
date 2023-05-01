# AVC-Predictor

## Descrição do projeto

Neste projeto, vamos usar classificadores para identificar *quais são os fatores de risco para o acidente vascular cerebral (AVC)*.

Temos à nossa disposição um conjunto de dados para [predição de AVCs](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). O que faremos é:

1. Treinar dois classificadores para predizer se houve ou não houve AVCs.
2. Verificar a acurácia dos classificadores.
3. Identificar quais são os fatores que mais provavelmente estão ligados a ter AVCs, dados pelos modelos. E conferir se de fato são reconhecidos no meio acadêmico.

## Metodologia utilizada

**Pré-processamento dos dados**: o conjunto de dados é tratado para garantir que esteja em um formato adequado para a análise. Isso inclui:

- Remoção de valores nulos (NaNs)
- Remoção de colunas que poderiam causar viés no modelo
- Conversão de variáveis categóricas em variáveis numéricas

<br>

## Modelos

### Modelo de classificação linear

---> Explicação aprofundada do modelo e sua implementação em demo.ipynb

Considerando: 

$$
f(x,y) = Ax + By + C
$$

Onde a equação f(x,y) = Ax + By + C é uma equação linear multivariável, onde x e y são as features, A e B são os pesos que multiplicam as features e C é o intercepto (ou "viés")

Podemos fazer formulação matricial da equação linear para entrada de i dimensões:


$$
f(\boldsymbol x) = b + \sum_n w_n x_n = \begin{bmatrix} w_0 & w_1 & ... & w_{N-1} \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ ... \\ x_{N-1} \end{bmatrix} + b = \boldsymbol w^T \boldsymbol x + b
$$

Onde x e y são as features passam a ser representados por i features no vetor ou matriz featurs:

$$
features = \begin{bmatrix}
x_0 \\ x_1 \\ ... \\ x_{N-1}
\end{bmatrix}
\hspace{0.5in}
$$

E os antigos pesos A e B, passam a ser representados pro i pesos no vetor ou matriz W

$$
W = \begin{bmatrix}
w_0 & w_1 & ... & w_{N-1}\\
\end{bmatrix}
\hspace{0.5in}
$$






X <-- Features que serão utilizadas no modelo
Y <-- Target 





### Modelo de classificação DecisionTreeClassifier


---> Explicação aprofundada do modelo e sua implementação em demo.ipynb


**Definição da função de perda**: é definida uma função de perda que mede a diferença entre a previsão do modelo e o valor real da variável de resposta. A função de perda é uma função de quatro parâmetros: os pesos (w), o viés (b), os pontos (ou dados) e os valores reais da variável de resposta e retorna o erro quadrático médio (EQM).

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

## Principais resultados encontrados


