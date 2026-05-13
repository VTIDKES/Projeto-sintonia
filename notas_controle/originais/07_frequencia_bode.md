# 🎛️ 7. Resposta em Frequência — Diagrama de Bode

## 📘 Conceito Fundamental

No **regime permanente**, quando uma **entrada senoidal** é aplicada a um sistema linear, a saída é também uma senoide de **mesma frequência**, porém com **amplitude** e **fase** diferentes.

$$x(t) = X\sin(\omega t) \Longrightarrow y(t) = Y\sin(\omega t + \phi)$$

O sistema modifica:
- **Magnitude**: razão $M(\omega) = Y/X = |H(j\omega)|$
- **Fase**: diferença $\phi(\omega) = \angle H(j\omega)$

Ambas são **funções da frequência** $\omega$.


---


## 📘 Função de Transferência em Frequência

Para obter a resposta em frequência, substitui-se $s = j\omega$ na função de transferência:

$$H(j\omega) = H(s)\bigg|_{s=j\omega}$$

- **Magnitude**: $|H(j\omega)|$ — quanto o sistema amplifica ou atenua o sinal
- **Fase**: $\angle H(j\omega)$ — quanto o sistema atrasa ou adianta o sinal


---


## 📘 Diagrama de Bode

O Diagrama de Bode representa a resposta em frequência em dois gráficos com **escala logarítmica** no eixo de frequências:

1. **Magnitude** em decibéis [dB]: $M_{dB} = 20\log_{10}|H(j\omega)|$
2. **Fase** em graus [°]

### ⚡ Por que usar escala log e dB?
- Permite visualizar grandes faixas de frequência
- Facilita a adição de contribuições de cada polo e zero
- Assíntotas retilíneas simplificam o esboço manual


---


## 📘 Contribuição de Cada Fator

A função de transferência pode ser fatorada como produto de termos elementares. Cada um contribui aditivamente no diagrama de Bode (em dB e graus):

### ⚡ 1. Ganho constante $k$
- Magnitude: $20\log_{10}|k|$ dB (linha horizontal)
- Fase: $0°$ (se $k > 0$) ou $-180°$ (se $k < 0$)

### ⚡ 2. Polo real: $\frac{1}{s/a + 1}$

| Região | Assíntota de magnitude | Fase |
| --------| ----------------------| ------|
| $\omega \ll a$ | 0 dB | 0° |
| $\omega = a$ (frequência de corte) | -3 dB | -45° |
| $\omega \gg a$ | decai **-20 dB/década** | -90° |

### ⚡ 3. Zero real: $(s/b + 1)$

| Região | Assíntota de magnitude | Fase |
| --------| ----------------------| ------|
| $\omega \ll b$ | 0 dB | 0° |
| $\omega = b$ | +3 dB | +45° |
| $\omega \gg b$ | cresce **+20 dB/década** | +90° |

### ⚡ 4. Polo na origem: $\frac{1}{s}$
- Magnitude: **-20 dB/década** para todas as frequências
- Fase: **-90°** constante

### ⚡ 5. Par de polos complexos: $\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}$

| Região | Assíntota de magnitude | Fase |
| --------| ----------------------| ------|
| $\omega \ll \omega_n$ | 0 dB | 0° |
| $\omega = \omega_n$ | pico ou vale (depende de $\zeta$) | -90° |
| $\omega \gg \omega_n$ | decai **-40 dB/década** | -180° |

> [!tip]
> 💡 Para $\zeta < 0.707$, há um **pico de ressonância** na magnitude próximo a $\omega_n$.


---


## 📘 Erro Máximo das Assíntotas

As assíntotas são aproximações. O erro máximo ocorre na **frequência de quebra**:

- Para polo/zero real: erro máximo de **±3 dB** (e ±45° na fase) na frequência de corte


---


## 📘 Exemplo — Sistema de 1ª Ordem

Para $H(s) = \frac{2}{s+3} = \frac{2/3}{s/3+1}$:

- Ganho DC: $20\log_{10}(2/3) \approx -3.5$ dB
- Frequência de corte: $\omega_c = 3$ rad/s
- Para $\omega > 3$: cai -20 dB/década

```scilab
clear; clc; clf();
s=%s;
num1=2; den1=s+3;
H1=syslin('c', num1, den1);
bode(H1)
gcf().figure_name="Diagrama de Bode";
```


---


## 📘 Exemplo — Sistema de 2ª Ordem (variação de ξ)

```scilab
clear; clc; clf();
s=%s;
k=1; wn=2;
for qsi = [0, 0.3, 0.7, 1.0, 2.0]
    num=k*wn^2; den=s^2+2*qsi*wn*s+wn^2;
    H=syslin('c', num, den);
    bode(H)
end
legend(['ξ=0';'ξ=0,3';'ξ=0,7';'ξ=1';'ξ=2'])
```


---


## 📘 Frequências Características do Sistema de 2ª Ordem

| Parâmetro | Fórmula | Significado |
| -----------| ---------| -------------|
| Frequência natural $\omega_n$ | Dado do sistema | Posição do pico |
| Frequência amortecida $\omega_d$ | $\omega_n\sqrt{1-\zeta^2}$ | Freq. de oscilação livre |
| Frequência de ressonância $\omega_r$ | $\omega_n\sqrt{1-2\zeta^2}$ | Freq. do pico de Bode |
| Frequência de banda passante $\omega_{BW}$ | $\omega_n\sqrt{1-2\zeta^2+\sqrt{4\zeta^4-4\zeta^2+2}}$ | Freq. em que M cai 3 dB |


---


## 📘 Diagrama de Nyquist (Polar)

Alternativa ao Bode: plota a trajetória de $H(j\omega)$ no plano complexo para $\omega \in [-\infty, +\infty]$.

O diagrama polar (curva de Nyquist) é a representação de $H(j\omega)$ em coordenadas polares (módulo e ângulo) conforme $\omega$ varia.

### ⚡ Como interpretar:
- Cada ponto corresponde a uma frequência
- O módulo é $|H(j\omega)|$, o ângulo é $\angle H(j\omega)$
- O gráfico é simétrico em relação ao eixo real (para $\omega$ negativo)


---


## 📘 Filtros

Um **filtro** é um sistema projetado para deixar passar apenas determinadas faixas de frequência.

### ⚡ Tipos de filtros:

| Tipo | Comportamento | Aplicação |
| ------| --------------| -----------|
| **Passa-baixa** | Passa baixas freq., atenua altas | Suavização, anti-aliasing |
| **Passa-alta** | Passa altas freq., atenua baixas | Diferenciação, remoção de DC |
| **Passa-faixa** | Passa uma faixa específica | Sintonização de canal |
| **Rejeita-faixa (notch)** | Atenua uma faixa específica | Rejeição de 60 Hz |

### ⚡ Conceitos de projeto:
- **Banda passante**: faixa onde $|H(j\omega)| \approx 1$ (ou maior que $1/\sqrt{2}$, ou -3 dB)
- **Banda de rejeição**: faixa onde $|H(j\omega)| \approx 0$
- **Frequência de corte**: limite entre as duas bandas

### ⚡ Filtros analógicos clássicos:
- **Butterworth**: resposta máximamente plana na banda passante
- **Chebyshev**: transição mais abrupta, com ripple na passante
- **Bessel**: fase linear (sem distorção de fase)
- **Elíptico**: transição mais abrupta, ripple em ambas as bandas


---


## 📘 Margem de Ganho e Margem de Fase (Estabilidade em Frequência)

Esses parâmetros indicam **quanto o sistema pode ser perturbado** antes de ficar instável:

| Parâmetro | Definição | Valor típico de projeto |
| -----------| -----------| ------------------------|
| **Margem de fase (PM)** | Fase em $\omega_{gc}$ + 180° | PM > 45° (idealmente ~60°) |
| **Margem de ganho (GM)** | $-20\log|H(j\omega_{pc})|$ | GM > 6 dB |

onde:
- $\omega_{gc}$ = frequência de cruzamento de ganho (onde $|H| = 1$)
- $\omega_{pc}$ = frequência de cruzamento de fase (onde $\angle H = -180°$)


---


## 📘 Referências

- SEDRA, A. S. et al. *Microeletrônica*. 8ª ed. Pearson, 2020.
- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
