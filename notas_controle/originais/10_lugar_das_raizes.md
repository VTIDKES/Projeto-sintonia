# 🎛️ 10. Lugar das Raízes

## 📘 O que é o Lugar das Raízes

O **lugar das raízes** (*root locus*) é o desenho do caminho que os polos de **malha fechada** percorrem no plano $s$ à medida que um parâmetro varia, normalmente o ganho $k$.

Ele responde à pergunta central do projeto por realimentação: *até onde posso aumentar o ganho antes de estragar a resposta?*

Para o sistema em malha fechada com $G(s)H(s)$ na malha aberta, o polinômio característico é:

$$1 + k\,G(s)H(s) = 0$$

Cada valor de $k$ produz um conjunto de raízes. O lugar das raízes é o traço contínuo dessas raízes conforme $k$ vai de $0$ a $\infty$.


---


## 📘 Condições de Módulo e de Ângulo

Um ponto $s$ pertence ao lugar das raízes se satisfaz simultaneamente:

### ⚡ Condição de ângulo (define o traçado)

$$\angle G(s)H(s) = \pm 180°(2q+1), \quad q = 0, 1, 2, \ldots$$

### ⚡ Condição de módulo (define o valor de $k$)

$$k = \frac{1}{|G(s)H(s)|}$$

> [!tip]
> 💡 A condição de ângulo diz **onde** o lugar passa. A de módulo diz **com que ganho** o polo chega ali. Primeiro se desenha a curva, depois se calcula o ganho do ponto escolhido.


---


## 📘 Regras de Traçado

Para $G(s)H(s)$ com $n$ polos e $m$ zeros:

| # | Regra | Detalhe |
| ---| ------| --------|
| 1 | **Número de ramos** | Igual a $n$, a ordem do denominador |
| 2 | **Simetria** | O lugar é simétrico em relação ao eixo real |
| 3 | **Pontos de partida** ($k=0$) | Os $n$ polos de malha aberta |
| 4 | **Pontos de chegada** ($k\to\infty$) | Os $m$ zeros finitos, e $n-m$ ramos vão ao infinito |
| 5 | **Eixo real** | Um trecho pertence ao lugar se o total de polos e zeros à sua direita for **ímpar** |
| 6 | **Assíntotas** | São $n-m$ retas com ângulo $\theta_a$ |
| 7 | **Centroide** | Ponto de encontro das assíntotas no eixo real |
| 8 | **Pontos de separação** | Onde ramos deixam ou entram no eixo real |
| 9 | **Cruzamento do eixo $j\omega$** | Fronteira da estabilidade, achada por Routh |
| 10 | **Ângulos de partida e chegada** | Em polos e zeros complexos |

### ⚡ Ângulo das assíntotas

$$\theta_a = \frac{180°(2q+1)}{n-m}, \quad q = 0, 1, \ldots, n-m-1$$

### ⚡ Centroide

$$\sigma_a = \frac{\sum \text{polos} - \sum \text{zeros}}{n-m}$$

### ⚡ Pontos de separação

Escrevendo $k = -1/[G(s)H(s)]$ e derivando:

$$\frac{dk}{ds} = 0$$

As raízes reais dessa equação que caem sobre o lugar são os pontos onde os ramos se separam do eixo real.


---


## 📘 Exemplo Completo

Seja a malha aberta:

$$G(s)H(s) = \frac{k}{s(s+2)(s+3)}$$

### ⚡ Passo 1 — Polos e zeros

Polos em $s = 0$, $s = -2$ e $s = -3$. Nenhum zero finito, logo $n = 3$ e $m = 0$.

### ⚡ Passo 2 — Trechos do eixo real

Contando o que há à direita de cada ponto: pertencem ao lugar os intervalos $[-2, 0]$ e $(-\infty, -3]$.

### ⚡ Passo 3 — Assíntotas

Como $n - m = 3$:

$$\theta_a = 60°, \; 180°, \; 300°$$

$$\sigma_a = \frac{(0) + (-2) + (-3) - 0}{3} = -\frac{5}{3} \approx -1{,}67$$

### ⚡ Passo 4 — Ponto de separação

O polinômio característico é $s^3 + 5s^2 + 6s + k = 0$, então $k = -(s^3 + 5s^2 + 6s)$:

$$\frac{dk}{ds} = -(3s^2 + 10s + 6) = 0 \Rightarrow s = -0{,}785 \; \text{ou} \; s = -2{,}549$$

Apenas $s = -0{,}785$ está sobre o lugar. O ganho nesse ponto é $k \approx 2{,}05$.

### ⚡ Passo 5 — Cruzamento do eixo imaginário

Tabela de Routh para $s^3 + 5s^2 + 6s + k$:

```
s^3  |   1      6
s^2  |   5      k
s^1  |  (30 - k)/5
s^0  |   k
```

A primeira coluna troca de sinal quando $k > 30$. Portanto:

- **Faixa estável:** $0 < k < 30$
- **Ganho crítico:** $k = 30$
- **Frequência de cruzamento:** do polinômio auxiliar $5s^2 + 30 = 0$, vem $\omega = \sqrt{6} \approx 2{,}45$ rad/s

### ⚡ Passo 6 — Leitura do projeto

| Faixa de $k$ | Polos dominantes | Resposta |
| -------------| -----------------| ---------|
| $0 < k < 2{,}05$ | Reais | Sem oscilação |
| $2{,}05 < k < 30$ | Complexos, à esquerda | Oscila e converge |
| $k = 30$ | Sobre o eixo $j\omega$ | Oscilação permanente |
| $k > 30$ | No semiplano direito | Diverge |


---


## 📘 Projeto de Ganho a partir de uma Especificação

O caminho usual parte de um requisito de desempenho e chega ao ganho:

1. Traduza a especificação em $\zeta$. Por exemplo, $\%UP \leq 10\%$ exige $\zeta \geq 0{,}59$.
2. Trace no plano $s$ a reta que sai da origem com ângulo $\theta = \arccos\zeta$.
3. Marque onde essa reta cruza o lugar das raízes.
4. Aplique a condição de módulo nesse ponto para obter $k$.

$$\zeta = \cos\theta \qquad \Longleftrightarrow \qquad \theta = \arccos\zeta$$

> [!warning]
> ⚠️ As fórmulas de $\%UP$, $T_p$ e $T_s$ só valem se os polos escolhidos forem realmente **dominantes**. Um terceiro polo próximo, ou um zero na vizinhança, invalida a estimativa. A regra prática pede que os demais polos estejam pelo menos cinco vezes mais à esquerda.


---


## 📘 Efeito de Acrescentar Polos e Zeros

| Acréscimo | Efeito no lugar | Consequência prática |
| -----------| ---------------| ---------------------|
| **Polo no SPE** | Empurra os ramos **para a direita** | Menos estável, mais lento |
| **Zero no SPE** | Puxa os ramos **para a esquerda** | Mais estável, mais rápido |
| **Polo na origem** | Aumenta o tipo do sistema | Elimina erro ao degrau, reduz margem |
| **Zero no SPD** | Ramos migram para a direita | Fase não mínima, resposta com reversão inicial |

Essa é exatamente a lógica por trás dos compensadores: um zero bem posicionado reorganiza o lugar das raízes para que a especificação passe a ser alcançável.


---


## 📘 Lugar das Raízes de Ganho Negativo

Quando $k < 0$, a condição de ângulo muda para $0°$ em vez de $180°$, e a regra do eixo real inverte: passa a valer para um número **par** de polos e zeros à direita. Esse traçado complementar é chamado de lugar das raízes complementar e aparece em sistemas com realimentação positiva.


---


## 📘 Código Scilab

```scilab
clear; clc; clf();
s = %s;
G = syslin('c', 1, s*(s+2)*(s+3));
evans(G, 50)          // lugar das raizes ate k = 50
sgrid()               // curvas de zeta e wn constantes
```

Para achar o ganho de um ponto escolhido com o mouse:

```scilab
[k, s0] = krac2(G)    // clique sobre o lugar e leia o ganho
```

Verificação da faixa estável pelo critério de Routh:

```scilab
clear; clc; s = %s;
G = syslin('c', 1, s*(s+2)*(s+3));
routh_t(G, poly(0,'k'))
```


---


## 📘 Resumo

| Pergunta | Onde olhar |
| ---------| -----------|
| Quantos ramos existem? | Ordem do denominador de malha aberta |
| Onde os ramos começam? | Nos polos de malha aberta |
| Para onde vão? | Zeros finitos e assíntotas |
| Qual o ganho de um ponto? | Condição de módulo |
| Até onde o sistema é estável? | Cruzamento do eixo $j\omega$, por Routh |
| Que ganho atende ao $\%UP$ pedido? | Interseção da reta de $\zeta$ com o lugar |


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
