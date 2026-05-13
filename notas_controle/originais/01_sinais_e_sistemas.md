# 🎛️ 1. Sinais e Sistemas Lineares

## 📘 O que é um sinal?

Um **sinal** é um conjunto de dados ou informação que pode ser processado por um sistema. Os sistemas podem modificar sinais ou extrair informações deles.

Um **sistema** é uma entidade que recebe um conjunto de sinais de **entrada** e produz um conjunto de sinais de **saída**. Pode ser implementado em:
- **Hardware**: componentes físicos, elétricos, mecânicos ou hidráulicos
- **Software**: algoritmo que calcula a saída a partir da entrada


---


## 📘 Classificação dos Sinais

Os sinais são classificados em duas dimensões:

**Quanto ao eixo do tempo (horizontal):**
- **Contínuo**: definido para todos os instantes de tempo
- **Discreto**: definido apenas em instantes específicos

**Quanto à amplitude (eixo vertical):**
- **Analógico**: amplitude varia continuamente
- **Digital**: amplitude assume apenas valores discretos

> [!tip]
> 💡 As combinações possíveis são: contínuo analógico, contínuo digital, discreto analógico e discreto digital.


---


## 📘 Tamanho de um sinal

Para sinais não-limitados no tempo, o tamanho é medido pela **energia média (potência)** — o valor médio quadrático da amplitude.


---


## 📘 Funções Singulares

Funções singulares são utilizadas na análise matemática de sistemas. Suas características:
- Possuem um **ponto singular** na origem (ponto onde a derivada não existe)
- São nulas em todas as demais posições
- Podem ser obtidas por diferenciação ou integração entre si

As principais funções singulares são:

| Função | Descrição |
| --------| -----------|
| Impulso unitário (δ) | "Spike" de área 1 na origem |
| Degrau unitário (u) | Vale 0 para t<0 e 1 para t≥0 |
| Rampa | Cresce linearmente a partir da origem |
| Parábola | Cresce quadraticamente |

### ⚡ Degrau Unitário u(t)

O degrau unitário é uma das funções singulares mais usadas em análise de sistemas:

$$u(t) = \begin{cases} 0 & t < 0 \\ 1 & t \geq 0 \end{cases}$$


---


## 📘 Classificação dos Sistemas

### ⚡ Quanto às entradas/saídas:
- **SISO** (Single-Input, Single-Output): uma entrada, uma saída
- **MIMO** (Multiple-Input, Multiple-Output): múltiplas entradas e saídas

### ⚡ Composição da resposta total:

$$\text{resposta total} = \underbrace{\text{resposta de entrada nula}}_{\text{(natural)}} + \underbrace{\text{resposta de estado nulo}}_{\text{(forçada)}}$$

### ⚡ Quanto ao tempo:
- **Sistemas em tempo contínuo**: entradas e saídas são sinais contínuos
- **Sistemas em tempo discreto**: entradas e saídas são sinais discretos
- **Sistemas analógicos**: sinais de entrada e saída são analógicos
- **Sistemas digitais**: sinais de entrada e saída são digitais


---


## 📘 Estabilidade de Sistemas Lineares Invariantes no Tempo

| Tipo | Comportamento da resposta natural |
| ------| ----------------------------------|
| **Estável** | Tende a zero quando t → ∞ |
| **Instável** | Cresce sem limites quando t → ∞ |
| **Marginalmente estável** | Permanece constante ou oscila quando t → ∞ |

> Quando o sistema é estável, apenas a **resposta forçada** permanece no regime permanente.


---


## 📘 Sistemas de Controle

Um sistema de controle reúne subsistemas e processos com o objetivo de obter uma **resposta desejada** para uma determinada entrada.

### ⚡ Malha Aberta
O controlador atua diretamente sobre o processo, sem verificar o resultado. Simples, mas sem correção automática de erros.

```
Entrada → Controlador → Atuador → Processo → Saída
```

### ⚡ Malha Fechada (Realimentação)
A saída é medida e comparada com a referência. O erro é usado para corrigir a ação do controlador.

```
Entrada → [+]→ Controlador → Atuador → Processo → Saída
           ↑                                          |
           └──────────── Sensor ←────────────────────┘
```

**Vantagens da malha fechada:**
- Rejeição de perturbações externas
- Redução de erros entre saída real e resposta desejada


---


## 📘 Objetivos de Análise e Projeto

| Objetivo | Descrição |
| ----------| -----------|
| **Resposta transitória** | Comportamento do sistema antes de atingir o regime permanente |
| **Resposta estacionária** | Comportamento após atingir o regime permanente (erro estacionário) |
| **Estabilidade** | O sistema deve ser estável |


---


## 📘 Sinais de Entrada para Análise de Desempenho

| Entrada | Uso principal |
| ---------| --------------|
| Impulso | Análise da resposta transitória |
| Degrau | Resposta transitória e erro de regime permanente |
| Rampa | Erro de regime permanente |
| Parábola | Erro de regime permanente |
| Senoidal | Resposta transitória e erro de regime permanente |


---


## 📘 Diagrama de Blocos

Ferramenta gráfica para representar sistemas complexos. Cada bloco representa um subsistema com sua relação entrada-saída.

### ⚡ Conexões possíveis:

**Em cascata (série):** a saída de um bloco é a entrada do próximo
$$Y(s) = H_1(s) \cdot H_2(s) \cdot X(s)$$

**Em paralelo:** os blocos recebem a mesma entrada e suas saídas são somadas
$$Y(s) = [H_1(s) + H_2(s)] \cdot X(s)$$

**Com realimentação:** a função de transferência em malha fechada é
$$T(s) = \frac{G(s)}{1 \pm G(s)H(s)}$$


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
