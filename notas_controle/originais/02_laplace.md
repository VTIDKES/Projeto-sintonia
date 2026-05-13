# 🎛️ 2. Transformada de Laplace

## 📘 Por que usar a Transformada de Laplace?

A Transformada de Laplace converte equações diferenciais (domínio do tempo) em equações algébricas (domínio da frequência complexa *s*), tornando a análise de sistemas muito mais simples.

$$\mathcal{L}\{f(t)\} = F(s) = \int_0^{\infty} f(t)\, e^{-st}\, dt$$

A transformada inversa recupera o sinal no domínio do tempo a partir de $F(s)$.


---


## 📘 Propriedades Fundamentais

| Propriedade | Expressão |
| -------------| -----------|
| **Linearidade (Homogeneidade)** | $\mathcal{L}\{af(t)\} = aF(s)$ |
| **Aditividade** | $\mathcal{L}\{f_1(t)+f_2(t)\} = F_1(s)+F_2(s)$ |
| **Deslocamento no tempo** | $\mathcal{L}\{f(t-a)u(t-a)\} = e^{-as}F(s)$ |
| **Deslocamento na frequência** | $\mathcal{L}\{e^{at}f(t)\} = F(s-a)$ |
| **Derivação** | $\mathcal{L}\{f'(t)\} = sF(s) - f(0^-)$ |
| **Integração** | $\mathcal{L}\left\{\int_0^t f(\tau)d\tau\right\} = \frac{F(s)}{s}$ |
| **Valor inicial** | $f(0^+) = \lim_{s\to\infty} sF(s)$ |
| **Valor final** | $\lim_{t\to\infty} f(t) = \lim_{s\to 0} sF(s)$ |

> [!warning]
> ⚠️ O **teorema do valor final** só é válido se os polos de $sF(s)$ estiverem no semiplano esquerdo.


---


## 📘 Tabela de Transformadas Comuns

| $f(t)$ | $F(s)$ |
| --------| --------|
| $\delta(t)$ (impulso) | $1$ |
| $u(t)$ (degrau) | $\dfrac{1}{s}$ |
| $t$ (rampa) | $\dfrac{1}{s^2}$ |
| $e^{-at}$ | $\dfrac{1}{s+a}$ |
| $\sin(\omega t)$ | $\dfrac{\omega}{s^2+\omega^2}$ |
| $\cos(\omega t)$ | $\dfrac{s}{s^2+\omega^2}$ |
| $e^{-at}\sin(\omega t)$ | $\dfrac{\omega}{(s+a)^2+\omega^2}$ |
| $e^{-at}\cos(\omega t)$ | $\dfrac{s+a}{(s+a)^2+\omega^2}$ |


---


## 📘 Convolução

A convolução é um operador linear que mede a sobreposição entre duas funções deslocadas:

**Convolução no tempo** → **Multiplicação no domínio de Laplace:**
$$\mathcal{L}\{f_1(t) * f_2(t)\} = F_1(s) \cdot F_2(s)$$

**Convolução na frequência** → **Multiplicação no domínio do tempo**


---


## 📘 Função de Transferência

A função de transferência $H(s)$ relaciona a saída à entrada no domínio de Laplace (com condições iniciais nulas):

$$H(s) = \frac{Y(s)}{X(s)} = \frac{b_m s^m + \cdots + b_1 s + b_0}{a_n s^n + \cdots + a_1 s + a_0}$$

### ⚡ Pólos e Zeros

- **Zeros**: valores de $s$ que tornam o numerador zero → $b(s) = 0$
- **Pólos**: valores de $s$ que tornam o denominador zero → $a(s) = 0$

A posição dos pólos no plano complexo determina a **estabilidade** e o **comportamento dinâmico** do sistema.


---


## 📘 Realizações de Sistemas

Uma **realização** é uma implementação do sistema via diagrama de blocos usando integradores. É **canônica** quando o número de integradores é igual à ordem da função de transferência.

### ⚡ Forma Direta
Implementa numerador e denominador de forma sequencial com realimentações e derivações.

### ⚡ Realização em Cascata
A função de transferência é fatorada como produto de funções de 1ª e 2ª ordem:
$$H(s) = H_1(s) \cdot H_2(s) \cdots H_k(s)$$

### ⚡ Realização em Paralelo
A função de transferência é expandida em frações parciais:
$$H(s) = H_1(s) + H_2(s) + \cdots + H_k(s)$$


---


## 📘 Expansão em Frações Parciais (Heaviside)

Permite decompor $F(s)$ em termos simples para aplicar a Laplace inversa.

### ⚡ Raízes Reais e Distintas

Para $F(s) = \dfrac{N(s)}{(s+p_1)(s+p_2)\cdots(s+p_n)}$:

$$F(s) = \frac{K_1}{s+p_1} + \frac{K_2}{s+p_2} + \cdots$$

onde $K_i = \lim_{s \to -p_i} (s+p_i)F(s)$

### ⚡ Raízes Complexas ou Imaginárias

Para pólos complexos conjugados $s = -\sigma \pm j\omega$, o resultado no tempo é uma **senóide amortecida**:

$$\mathcal{L}^{-1}\left\{\frac{K}{(s+\sigma)^2+\omega^2}\right\} = K\,e^{-\sigma t}\sin(\omega t)\,u(t)$$

### ⚡ Raízes Reais Repetidas

Para um pólo $s = -p$ com multiplicidade $r$:

$$F(s) = \frac{K_r}{(s+p)^r} + \frac{K_{r-1}}{(s+p)^{r-1}} + \cdots + \frac{K_1}{s+p}$$


---


## 📘 Resposta Total no Domínio do Tempo

$$\text{resposta total} = \underbrace{\text{resposta de entrada nula}}_{\text{(natural)}} + \underbrace{\text{resposta de estado nulo}}_{\text{(forçada)}}$$

A **resposta natural** depende dos pólos do sistema. A **resposta forçada** depende da entrada.


---


## 📘 Sistemas Não-Lineares

Na prática, muitos sistemas apresentam não-linearidades:

| Tipo | Classificação | Descrição |
| ------| --------------| -----------|
| **Atrito** | Magnitude | Força estática maior que cinética |
| **Zona morta** | Magnitude | Sistema só responde após certo valor de entrada |
| **Saturação** | Magnitude | Saída para de crescer após certo valor da entrada |
| **Folga** | Magnitude | Jogo mecânico entre componentes |
| **Histerese** | Magnitude | Curva diferente na ida e na volta |
| **Relé** | Magnitude | Chaveamento on/off |
| **Auto-excitação** | Frequência | Oscilação com entrada nula |
| **Harmônicas** | Frequência | Saída com frequências múltiplas da entrada |
| **Ciclo limite** | Frequência | Oscilação de amplitude e período fixos |
| **Caos/Bifurcação** | Frequência | Comportamento irregular dependente das condições iniciais |

### ⚡ Linearização

Quando as não-linearidades são pequenas, pode-se linearizar o sistema em torno de um **ponto de equilíbrio**. A linearização consiste em aproximar a função não-linear pela sua derivada nesse ponto (série de Taylor de 1ª ordem):

$$f(x) \approx f(x_0) + \frac{df}{dx}\bigg|_{x_0}(x - x_0)$$

### ⚡ Tempo Morto (Atraso de Transporte)

O atraso puro de transporte é representado por $e^{-\theta s}$ e pode ser aproximado pela **Aproximação de Padé** de 2ª ordem:

$$e^{-\theta s} \approx \frac{1 - \frac{\theta}{2}s + \frac{\theta^2}{12}s^2}{1 + \frac{\theta}{2}s + \frac{\theta^2}{12}s^2}$$


---


## 📘 Código Scilab — Frações Parciais

```scilab
clear; clc; s=%s;
num=1*s^3+2*s^2+6*s+7; den=(s^2+s+5); 
F=syslin('c', num, den);
fp=pfss(F);   % expansão em frações parciais
```


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
