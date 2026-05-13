# 🎛️ 3. Dinâmica — Sistemas de 1ª Ordem

## 📘 Análise no Domínio do Tempo

O domínio do tempo é a perspectiva mais intuitiva para analisar o comportamento de sistemas: observamos diretamente como a saída evolui ao longo do tempo em resposta a uma entrada.


---


## 📘 Sistemas de Primeira Ordem

A função de transferência canônica de um sistema de 1ª ordem é:

$$H(s) = \frac{k}{s + a} \quad \text{(grau relativo 1, sem zero)}$$

$$H(s) = \frac{k(s + b)}{s + a} \quad \text{(grau relativo 0, com zero)}$$

onde:
- $k$ é o **ganho**
- $a$ determina a **constante de tempo** $\tau = 1/a$ (posição do polo)
- $b$ é a posição do **zero** (quando existe)


---


## 📘 Resposta ao Degrau — Grau Relativo 1 (sem zero)

A resposta ao degrau de um sistema de 1ª ordem sem zero é uma exponencial que cresce (ou decai) suavemente:

$$y(t) = \frac{k}{a}\left(1 - e^{-at}\right)u(t)$$

### ⚡ Características típicas:
- **Sem ultrapassagem** (*overshoot*)
- **Inclinação inicial não nula**
- Atinge ~63% do valor final em $t = \tau$
- Considerado em regime permanente para $t > 4\tau$ (~98%)

### ⚡ Especificações de desempenho

| Especificação | Descrição | Como medir |
| --------------| -----------| -----------|
| **Constante de tempo (τ)** | Velocidade de resposta | Tempo para atingir 63% do valor final |
| **Tempo de subida (Tᵣ)** | Tempo de 10% a 90% do valor final | Medir diretamente na curva |
| **Tempo de acomodação (Tₛ)** | Tempo para entrar na faixa ±2% | Medir diretamente na curva |
| **Valor final y(∞)** | Ganho estático × amplitude do degrau | $y(\infty) = k/a$ |


---


## 📘 Efeito dos Parâmetros na Resposta

### ⚡ Variação do ganho $k$
- Altera o **valor final** da resposta: $y(\infty) = k/a$
- **Não altera** a constante de tempo $\tau = 1/a$

### ⚡ Variação do polo $a$
- Altera a **velocidade de resposta**: maior $a$ → resposta mais rápida
- $\tau = 1/a$ → quanto maior $a$, menor $\tau$
- **Não altera** o valor final relativo

### ⚡ Código Scilab — variação de $k$:
```scilab
clear; clc; clf();
s=%s; t=0:0.01:10;
for k = [1 2 3 4 5 6]
    num=k; den=s+k;
    H=syslin('c', num, den);
    y=csim('step',t,H);
    plot(t,y)
end
```


---


## 📘 Sistemas com Zero — Grau Relativo 0

Quando o sistema possui um zero $b$ além do polo $a$:

$$H(s) = \frac{k(s + b)}{s + a}$$

### ⚡ Efeito do zero:
- **Não altera** a constante de tempo do sistema
- **Altera** o valor inicial da saída (pode haver "salto" inicial)
- Zeros no **semiplano esquerdo** ($b > 0$): resposta ainda é monótona ou pode ter comportamento não mínimo de fase
- Zeros no **semiplano direito** ($b < 0$): resposta tem **comportamento de fase não mínima** — a saída começa indo na direção oposta antes de seguir a direção correta


---


## 📘 Polo no Semiplano Direito

Se o polo $a < 0$, o sistema é **instável em malha aberta**: a resposta cresce exponencialmente sem limite.

$$H(s) = \frac{k}{s - |a|}$$

A realimentação pode estabilizar esse tipo de sistema.


---


## 📘 Polo na Origem — Integrador

Quando $a = 0$, o sistema possui um **polo na origem** (integrador):

$$H(s) = \frac{k}{s}$$

- Em malha aberta: a resposta a um degrau é uma **rampa** (cresce indefinidamente)
- Em **malha fechada**: o integrador garante **erro nulo em regime permanente** para entrada em degrau


---


## 📘 Identificação do Sistema a partir da Resposta ao Degrau

Uma vantagem prática: mesmo sem conhecer a estrutura interna do sistema, a partir da curva de resposta ao degrau pode-se estimar:

1. **Constante de tempo $\tau$**: tempo para a saída atingir 63% do valor final
2. **Ganho estático**: $k/a = y(\infty) / A$ (onde $A$ é a amplitude do degrau)
3. **Função de transferência aproximada**: $H(s) \approx \frac{y(\infty)}{\tau s + 1}$


---


## 📘 Exemplos de Sistemas de 1ª Ordem

### ⚡ Sistema RC (elétrico)
$$H(s) = \frac{1/RC}{s + 1/RC}$$

Polo em $s = -1/RC$, constante de tempo $\tau = RC$.

### ⚡ Tanque de nível (hidráulico)
A dinâmica de um tanque com resistência de saída é tipicamente de 1ª ordem.

### ⚡ Motor DC (simplificado)
Desprezando a indutância, a relação velocidade/tensão é de 1ª ordem.


---


## 📘 Código Scilab — Diagrama de Polos/Zeros e Resposta

```scilab
clear; clc;
s=%s;
k=4; a=0.8; b=0; tal=1/a;
if b==0 then num=k; else num=k*(s+b); end;
den=(s+a);
H=syslin('c', num, den);
t=0:0.01:10;
y=csim('step',t,H);
clf();
subplot(121); plzr(H);          % diagrama de pólos e zeros
subplot(122); plot2d(t,y);      % resposta ao degrau
xgrid();
```


---


## 📘 Resumo das Posições do Polo e Comportamento

| Posição do polo | Comportamento | Estabilidade |
| -----------------| --------------| -------------|
| Semiplano esquerdo ($a > 0$) | Exponencial decrescente | Estável |
| Origem ($a = 0$) | Constante (integrador) | Marginalmente estável |
| Semiplano direito ($a < 0$) | Exponencial crescente | Instável |


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
