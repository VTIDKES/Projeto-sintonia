# 🎛️ 6. Estabilidade

## 📘 O que é Estabilidade?

Um sistema é **estável** se, após uma perturbação, ele retorna ao seu estado de equilíbrio. A estabilidade é um dos requisitos mais fundamentais no projeto de sistemas de controle.


---


## 📘 Estados de Equilíbrio

Analogia com uma bola em diferentes superfícies:

| Tipo | Comportamento | Exemplo físico |
| ------| --------------| ----------------|
| **Estável** | Quando perturbado, retorna ao equilíbrio | Bola no fundo de uma tigela |
| **Instável** | Quando perturbado, se afasta do equilíbrio | Bola no topo de uma colina |
| **Neutro (marginal)** | Quando perturbado, vai para novo equilíbrio | Bola em superfície plana |


---


## 📘 Estabilidade Interna vs. Externa

### ⚡ Estabilidade Externa (BIBO — Bounded Input, Bounded Output)
Um sistema é estável no sentido BIBO se toda entrada limitada produz saída limitada. É uma **descrição externa**.

### ⚡ Estabilidade Interna (Assintótica / Lyapunov)
Considera o comportamento interno do sistema (todos os estados). É uma **descrição mais completa**.

> [!warning]
> ⚠️ Um sistema pode ser estável externamente (BIBO) mas instável internamente (se for não controlável ou não observável). Por isso, a estabilidade interna é preferível.


---


## 📘 Critério de Estabilidade pelos Pólos

Para um sistema causal, linear e invariante no tempo:

| Condição dos pólos | Estabilidade |
| -------------------| -------------|
| Todos no **semiplano esquerdo** (SPE) | **Assintoticamente estável** |
| Pólos simples no eixo imaginário, nenhum no SPD | **Marginalmente estável** |
| **Qualquer pólo** no semiplano direito (SPD) | **Instável** |
| Pólos **repetidos** no eixo imaginário | **Instável** |


---


## 📘 Critério de Routh-Hurwitz

O critério de Routh-Hurwitz é um método **algébrico** para determinar a estabilidade de sistemas de **malha fechada** sem calcular explicitamente os polos.

### ⚡ Aplicação:
- Útil para sistemas de ordem maior que 2
- Permite determinar **quantos polos estão no SPD**
- Permite encontrar **faixas de ganho** para estabilidade

### ⚡ Como usar:

**Passo 1: Montar a tabela de Routh**

Para o polinômio $a_n s^n + a_{n-1} s^{n-1} + \cdots + a_1 s + a_0$:

```
s^n  |  a_n    a_{n-2}  a_{n-4}  ...
s^{n-1}|  a_{n-1}  a_{n-3}  a_{n-5}  ...
s^{n-2}|  b_1    b_2    b_3    ...
s^{n-3}|  c_1    c_2    c_3    ...
  ...
s^0  |  (*)
```

onde os elementos das novas linhas são calculados como:

$$b_1 = \frac{a_{n-1}\cdot a_{n-2} - a_n\cdot a_{n-3}}{a_{n-1}}$$

$$b_2 = \frac{a_{n-1}\cdot a_{n-4} - a_n\cdot a_{n-5}}{a_{n-1}}$$

e assim por diante.

**Passo 2: Interpretar a primeira coluna**

- **Número de mudanças de sinal** na 1ª coluna = número de polos no SPD
- Sem mudanças de sinal → sistema estável
- Zero na 1ª coluna → caso especial (substituir por $\varepsilon > 0$)
- Linha inteira de zeros → sistema marginalmente estável


---


## 📘 Exemplos

### ⚡ Exemplo 1 — Sistema estável

Planta: $G(s) = \frac{1000}{(s+2)(s+3)(s+5)}$ com realimentação unitária

Denominador em MF: $s^3 + 10s^2 + 31s + 1030$

```
s^3  |  1     31
s^2  |  10   1030
s^1  |  31 - 103 = -72  (negativo → mudança de sinal)
s^0  |  1030
```

**Resultado:** 2 mudanças de sinal → 2 polos no SPD → **sistema instável** com esse ganho.

```scilab
clear; clc; s=%s;
num1=1000; den1=(s+2)*(s+3)*(s+5);
G=syslin('c', num1, den1);
routh_t(G, 1)   % 1 = ganho de malha fechada
```


---


### ⚡ Exemplo 2 — Zero na primeira coluna

Para $H(s) = \frac{10}{s^5+2s^4+3s^3+6s^2+5s+3}$

Quando aparece zero na 1ª coluna, substitui-se por $\varepsilon$ (número positivo infinitesimal) e continua a tabela.

```scilab
num1=10; den1=s^5+2*s^4+3*s^3+6*s^2+5*s+3;
H=syslin('c', num1, den1);
routh_t(H, 1)
```


---


### ⚡ Exemplo 3 — Linha de zeros (sistema marginalmente estável)

Para $H(s) = \frac{10}{s^5+7s^4+6s^3+42s^2+8s+56}$

Quando aparece uma linha inteira de zeros, usa-se o **polinômio auxiliar** (derivada da linha anterior) para preencher e continuar a tabela.

A linha de zeros indica a presença de pólos **no eixo imaginário** (marginalmente estável).

```scilab
num1=10; den1=s^5+7*s^4+6*s^3+42*s^2+8*s+56;
H=syslin('c', num1, den1);
routh_t(H, 1)
```


---


### ⚡ Exemplo 4 — Região de estabilidade em função do ganho $k$

Para $G(s) = \frac{k}{s(s+7)(s+11)}$ com realimentação unitária:

Denominador em MF: $s^3 + 18s^2 + 77s + k$

Tabela de Routh:
```
s^3  |  1      77
s^2  |  18     k
s^1  |  (18×77 - k)/18 = (1386 - k)/18
s^0  |  k
```

Para estabilidade, todos os elementos da 1ª coluna devem ser positivos:
- $k > 0$
- $(1386 - k)/18 > 0 \Rightarrow k < 1386$

**Conclusão:** $0 < k < 1386$ para estabilidade.

```scilab
clear; clc; s=%s;
num1=1; den1=s*(s+7)*(s+11);
G=syslin('c', num1, den1);
routh_t(G, poly(0,'k'))   % análise simbólica com k
```


---


## 📘 Condição Necessária para Estabilidade

Uma condição necessária (mas não suficiente) para estabilidade é que **todos os coeficientes do polinômio característico sejam positivos e do mesmo sinal**. Se algum coeficiente for zero ou negativo, o sistema é definitivamente instável.


---


## 📘 Resumo

| Caso | Procedimento |
| ------| -------------|
| Nenhum zero na 1ª coluna | Contar mudanças de sinal |
| Zero isolado na 1ª coluna | Substituir por $\varepsilon > 0$ e continuar |
| Linha inteira de zeros | Usar polinômio auxiliar (derivada da linha anterior) |
| Todos positivos, sem mudança | **Sistema estável** |
| $N$ mudanças de sinal | $N$ polos no semiplano direito → **instável** |


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
