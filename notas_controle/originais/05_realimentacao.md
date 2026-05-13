# 🎛️ 5. Sistemas com Realimentação

## 📘 Por que usar Realimentação?

A realimentação (malha fechada) permite que o sistema reaja aos seus próprios erros, tornando-o capaz de:
- Rejeitar perturbações externas
- Reduzir o erro em regime permanente
- Ajustar a velocidade de resposta e o amortecimento


---


## 📘 Erro em Regime Permanente

O **erro em regime permanente** $e_{ss}$ é a diferença entre a referência e a saída após o sistema se estabilizar.

Para um sistema em malha fechada com realimentação unitária:

$$e_{ss} = \lim_{t \to \infty} e(t) = \lim_{s \to 0} \frac{sR(s)}{1 + G(s)}$$

onde $R(s)$ é a entrada e $G(s)$ é a função de transferência em malha aberta.


---


## 📘 Classificação por Tipo do Sistema

O **tipo do sistema** é o número de polos em $s = 0$ (integradores) na função de malha aberta $G(s)$.

### ⚡ Constantes de Erro Estático

| Constante | Definição | Entrada associada |
| -----------| -----------| ------------------|
| **Constante de posição** $K_p$ | $\lim_{s\to 0} G(s)$ | Degrau |
| **Constante de velocidade** $K_v$ | $\lim_{s\to 0} sG(s)$ | Rampa |
| **Constante de aceleração** $K_a$ | $\lim_{s\to 0} s^2G(s)$ | Parábola |

> [!tip]
> 💡 Quanto maior a constante de erro, menor o erro em regime permanente.

### ⚡ Tabela de Erros por Tipo

| Entrada | Tipo 0 | Tipo 1 | Tipo 2 |
| ---------| --------| --------| --------|
| Degrau (posição) | $\frac{1}{1+K_p}$ | **0** | **0** |
| Rampa (velocidade) | **∞** | $\frac{1}{K_v}$ | **0** |
| Parábola (aceleração) | **∞** | **∞** | $\frac{1}{K_a}$ |

> Sistemas de **tipo maior** eliminam o erro para entradas de ordem menor.


---


## 📘 Plantas de 1ª Ordem em Malha Fechada

### ⚡ Tipo 0

Para a planta $G(s) = \frac{k}{s+a}$ com realimentação unitária:

A função de malha fechada é:

$$H(s) = \frac{G(s)}{1+G(s)} = \frac{k}{s + a + k}$$

**Efeito do ganho $k$:**
- Maior $k$ → **polo mais à esquerda** → resposta mais rápida
- Maior $k$ → **menor erro estacionário** para entrada degrau: $e_{ss} = \frac{a}{a+k}$
- Porém, ganho infinito seria necessário para erro nulo → não é possível com tipo 0

### ⚡ Tipo 1

Para a planta $G(s) = \frac{k}{s(s+a)}$:

- **Erro nulo** para entrada degrau
- Erro constante $1/K_v$ para entrada rampa


---


## 📘 Plantas de 2ª Ordem em Malha Fechada

### ⚡ Tipo 0 — Subamortecido

Para $G(s) = \frac{k\omega_n^2}{s^2+2\zeta\omega_n s + \omega_n^2}$ com realimentação unitária:

$$H(s) = \frac{k\omega_n^2}{s^2+2\zeta\omega_n s + (1+k)\omega_n^2}$$

**Efeito do ganho $k$:**
- Aumenta a **frequência natural** do sistema em MF: $\omega_{n,MF} = \omega_n\sqrt{1+k}$
- O **coeficiente de amortecimento** em MF se reduz: $\zeta_{MF} = \frac{\zeta}{\sqrt{1+k}}$
- **Não altera** $T_p$ e $\%UP$ de forma independente — ambos mudam juntos

### ⚡ Tipo 1 — Caso subamortecido, criticamente amortecido e sobreamortecido

Para $G(s) = \frac{k}{s(s^2+as+b)}$, o ganho $k$ move os polos de MF e pode levar a instabilidade se $k$ for muito grande.


---


## 📘 Efeito da Realimentação Sobre Perturbações

Em sistemas práticos, perturbações $D(s)$ entram no processo. A realimentação reduz o efeito dessas perturbações na saída:

$$Y_{perturbação}(s) = \frac{G_2(s)}{1 + G_1(s)G_2(s)} D(s)$$

Quanto maior $G_1(s)G_2(s)$, menor o efeito da perturbação.


---


## 📘 Sistemas de Ordem Superior em MF

Para um sistema de ordem $n \geq 3$, a análise em malha fechada é mais complexa. Exemplo com planta de 3ª ordem:

$$G(s) = \frac{k}{s(s+2)(s+3)}$$

- Para $k$ pequeno: sistema estável com boa resposta
- Para $k$ intermediário: sistema oscila mais mas permanece estável
- Para $k \approx 30$: sistema começa a ficar marginalmente estável
- Para $k > 30$: sistema **instável** (polos de MF cruzam para o semiplano direito)

> [!warning]
> ⚠️ Aumentar o ganho indefinidamente sempre acaba desestabilizando sistemas de ordem 3 ou superior!


---


## 📘 Código Scilab — Malha Aberta vs. Malha Fechada (1ª Ordem)

```scilab
clear; clc;
s=%s; t=0:0.01:10;
k=4; a=0.8;

% Malha aberta
num1=k; den1=(s+a);
G=syslin('c', num1, den1);
y1=csim('step',t,G);

% Malha fechada (realimentação unitária)
num2=k; den2=(s+a+k);
H=syslin('c', num2, den2);
y2=csim('step',t,H);

clf();
subplot(221); plzr(G); title('Pólos - Malha Aberta')
subplot(222); plot2d(t,y1); title('Resposta - Malha Aberta')
subplot(223); plzr(H); title('Pólos - Malha Fechada')
subplot(224); plot2d(t,y2); title('Resposta - Malha Fechada')
```


---


## 📘 Código Scilab — Variação do Ganho em MF (1ª Ordem)

```scilab
clear; clc; clf();
s=%s; a=0.8; t=0:0.01:10;
for k = [2 4 8 16 32 64]
    num=k; den=s+a+k;
    H=syslin('c', num, den);
    y=csim('step',t,H);
    plot(t,y)
end
legend(['k=2';'k=4';'k=8';'k=16';'k=32';'k=64'])
xgrid()
```


---


## 📘 Resumo dos Efeitos da Realimentação

| Efeito | Descrição |
| --------| -----------|
| **Redução do erro** | MF reduz ou elimina o erro estacionário |
| **Aceleração** | MF pode tornar o sistema mais rápido |
| **Possível oscilação** | MF com ganho alto pode gerar oscilações |
| **Risco de instabilidade** | Ganho excessivo pode destabilizar |
| **Rejeição de perturbações** | MF atenua o efeito de distúrbios externos |


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
