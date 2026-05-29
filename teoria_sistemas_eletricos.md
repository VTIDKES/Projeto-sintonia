# Teoria de Controle — Sistemas Elétricos

> Documento de referência para análise e projeto de sistemas elétricos de controle.
> Fórmulas no padrão LaTeX — compatível com Obsidian, Jupyter, MkDocs e GitHub com MathJax.

---

## 1. Elementos Passivos Fundamentais

Os sistemas elétricos de controle são compostos por três elementos passivos: resistor $R$, capacitor $C$ e indutor $L$. Cada um possui uma relação constitutiva distinta entre tensão $v(t)$ e corrente $i(t)$.

---

### 1.1 Resistor

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 60" width="220" height="60">
  <line x1="10" y1="30" x2="35" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <polyline points="35,30 42,12 52,48 62,12 72,48 82,12 92,30" fill="none" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="92" y1="30" x2="120" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <text x="135" y="26" font-family="Georgia, serif" font-size="13" fill="#185FA5" font-style="italic">R</text>
  <line x1="10" y1="28" x2="18" y2="28" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
  <text x="12" y="22" font-family="Georgia, serif" font-size="11" fill="#444" font-style="italic">i</text>
  <defs>
    <marker id="arr" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M1,1 L7,4 L1,7" fill="none" stroke="#444" stroke-width="1.2"/>
    </marker>
  </defs>
</svg>

Elemento puramente dissipativo, sem memória dinâmica:

$$v(t) = R \cdot i(t)$$

Impedância no domínio de Laplace:

$$Z_R(s) = R$$

O resistor não introduz polos nem zeros — afeta apenas o **ganho** e o **amortecimento** em circuitos combinados.

---

### 1.2 Capacitor

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 60" width="220" height="60">
  <line x1="10" y1="30" x2="70" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="70" y1="12" x2="70" y2="48" stroke="#185FA5" stroke-width="3" stroke-linecap="round"/>
  <line x1="80" y1="12" x2="80" y2="48" stroke="#185FA5" stroke-width="3" stroke-linecap="round"/>
  <line x1="80" y1="30" x2="120" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <text x="135" y="26" font-family="Georgia, serif" font-size="13" fill="#185FA5" font-style="italic">C</text>
  <line x1="10" y1="28" x2="18" y2="28" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
  <text x="12" y="22" font-family="Georgia, serif" font-size="11" fill="#444" font-style="italic">i</text>
  <defs>
    <marker id="arr2" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M1,1 L7,4 L1,7" fill="none" stroke="#444" stroke-width="1.2"/>
    </marker>
  </defs>
</svg>

Armazena energia no campo elétrico. Sua relação constitutiva envolve a derivada da tensão:

$$i(t) = C \,\frac{dv(t)}{dt} \qquad \Longleftrightarrow \qquad v(t) = \frac{1}{C}\int_0^t i(\tau)\,d\tau$$

Impedância no domínio de Laplace:

$$Z_C(s) = \frac{1}{sC}$$

> **Propriedade física:** a tensão em um capacitor não pode variar instantaneamente, pois isso exigiria corrente infinita.

Em regime permanente DC ($s \to 0$): $Z_C \to \infty$ — o capacitor bloqueia corrente contínua.  
Em alta frequência ($s \to \infty$): $Z_C \to 0$ — curto-circuito para sinais rápidos.

---

### 1.3 Indutor

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 60" width="220" height="60">
  <line x1="10" y1="30" x2="32" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M32,30 Q36,14 40,30 Q44,14 48,30 Q52,14 56,30 Q60,14 64,30 Q68,14 72,30 Q76,14 80,30 Q84,14 88,30" fill="none" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="88" y1="30" x2="120" y2="30" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round"/>
  <text x="135" y="26" font-family="Georgia, serif" font-size="13" fill="#185FA5" font-style="italic">L</text>
  <line x1="10" y1="28" x2="18" y2="28" stroke="#444" stroke-width="1.2" marker-end="url(#arr3)"/>
  <text x="12" y="22" font-family="Georgia, serif" font-size="11" fill="#444" font-style="italic">i</text>
  <defs>
    <marker id="arr3" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M1,1 L7,4 L1,7" fill="none" stroke="#444" stroke-width="1.2"/>
    </marker>
  </defs>
</svg>

Armazena energia no campo magnético:

$$v(t) = L\,\frac{di(t)}{dt} \qquad \Longleftrightarrow \qquad i(t) = \frac{1}{L}\int_0^t v(\tau)\,d\tau$$

Impedância no domínio de Laplace:

$$Z_L(s) = sL$$

> **Propriedade física:** a corrente em um indutor não pode variar instantaneamente, pois isso exigiria tensão infinita.

---

### 1.4 Circuito RC — Diagrama Completo

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 110" width="380" height="110">
  <!-- Fio esquerdo -->
  <line x1="20" y1="30" x2="50" y2="30" stroke="#444" stroke-width="1.5"/>
  <!-- Resistor -->
  <polyline points="50,30 57,14 65,46 73,14 81,46 89,14 97,30" fill="none" stroke="#185FA5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="97" y1="30" x2="130" y2="30" stroke="#444" stroke-width="1.5"/>
  <!-- Capacitor -->
  <line x1="130" y1="12" x2="130" y2="48" stroke="#185FA5" stroke-width="3" stroke-linecap="round"/>
  <line x1="140" y1="12" x2="140" y2="48" stroke="#185FA5" stroke-width="3" stroke-linecap="round"/>
  <line x1="140" y1="30" x2="200" y2="30" stroke="#444" stroke-width="1.5"/>
  <!-- Fio direito + retorno -->
  <line x1="200" y1="15" x2="200" y2="85" stroke="#444" stroke-width="1.5"/>
  <line x1="20" y1="85" x2="200" y2="85" stroke="#444" stroke-width="1.5"/>
  <!-- Fonte de tensão -->
  <circle cx="20" cy="57" r="16" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.5"/>
  <line x1="20" y1="30" x2="20" y2="41" stroke="#444" stroke-width="1.5"/>
  <line x1="20" y1="73" x2="20" y2="85" stroke="#444" stroke-width="1.5"/>
  <text x="20" y="55" text-anchor="middle" font-family="Georgia,serif" font-size="10" fill="#854F0B" font-style="italic">v</text>
  <text x="20" y="65" text-anchor="middle" font-family="Georgia,serif" font-size="10" fill="#854F0B" font-style="italic">in</text>
  <!-- Labels -->
  <text x="73" y="10" text-anchor="middle" font-family="Georgia,serif" font-size="12" fill="#185FA5" font-style="italic">R</text>
  <text x="135" y="8" text-anchor="middle" font-family="Georgia,serif" font-size="12" fill="#185FA5" font-style="italic">C</text>
  <text x="215" y="35" font-family="Georgia,serif" font-size="11" fill="#444">v<tspan font-size="9" dy="3">out</tspan></text>
  <!-- Seta de corrente -->
  <line x1="50" y1="27" x2="62" y2="27" stroke="#444" stroke-width="1.1" marker-end="url(#aci)"/>
  <text x="54" y="22" font-family="Georgia,serif" font-size="10" fill="#444" font-style="italic">i</text>
  <defs>
    <marker id="aci" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="4" markerHeight="4" orient="auto">
      <path d="M1,1 L7,4 L1,7" fill="none" stroke="#444" stroke-width="1.2"/>
    </marker>
  </defs>
</svg>

---

### 1.5 Resumo dos elementos

| Elemento | Relação $v$–$i$ | Impedância $Z(s)$ | Energia armazenada |
|----------|-----------------|-------------------|--------------------|
| Resistor $R$ | $v = Ri$ | $R$ | $0$ (dissipa) |
| Capacitor $C$ | $i = C\,\dot{v}$ | $\dfrac{1}{sC}$ | $\frac{1}{2}Cv^2$ |
| Indutor $L$ | $v = L\,\dot{i}$ | $sL$ | $\frac{1}{2}Li^2$ |

---

## 2. Circuito RC — Sistema de Primeira Ordem

### 2.1 Equacionamento

Circuito série $R$–$C$ com entrada $v_{in}(t)$ e saída $v_{out}(t)$ medida sobre o capacitor.

Aplicando a Lei de Kirchhoff das Tensões (LKT):

$$v_{in}(t) = R\,i(t) + v_{out}(t)$$

Substituindo $i(t) = C\,\dfrac{dv_{out}}{dt}$:

$$RC\,\frac{dv_{out}(t)}{dt} + v_{out}(t) = v_{in}(t)$$

### 2.2 Função de Transferência

Aplicando a Transformada de Laplace com condições iniciais nulas:

$$\boxed{G(s) = \frac{V_{out}(s)}{V_{in}(s)} = \frac{1}{1 + \tau s}}$$

onde $\tau = RC$ é a **constante de tempo** do sistema $[\text{s}]$.

### 2.3 Resposta ao Degrau

Para entrada degrau $V_{in}(s) = \dfrac{A}{s}$:

$$V_{out}(s) = \frac{A}{s(1 + \tau s)}$$

Invertendo pela transformada de Laplace:

$$\boxed{v_{out}(t) = A\left(1 - e^{-t/\tau}\right), \quad t \geq 0}$$

| Instante | $v_{out}/A$ |
|----------|-------------|
| $t = \tau$ | $0{,}632$ |
| $t = 2\tau$ | $0{,}865$ |
| $t = 3\tau$ | $0{,}950$ |
| $t = 5\tau$ | $0{,}993$ |
| $t \to \infty$ | $1{,}000$ |

O tempo de acomodação convencional (critério $\pm 2\%$) é $t_s \approx 4\tau$.

### 2.4 Polo e Diagrama de Bode

Polo em:

$$s = -\frac{1}{\tau} = -\frac{1}{RC}$$

No diagrama de Bode com $\omega_c = 1/\tau$:

| Frequência | Magnitude | Fase |
|------------|-----------|------|
| $\omega \ll \omega_c$ | $\approx 0\,\text{dB}$ | $\approx 0°$ |
| $\omega = \omega_c$ | $-3\,\text{dB}$ | $-45°$ |
| $\omega \gg \omega_c$ | $-20\,\text{dB/déc}$ | $\to -90°$ |

O RC atua como **filtro passa-baixa de primeira ordem**.

---

## 3. Circuito RLC — Sistema de Segunda Ordem

### 3.1 Equação Diferencial

Circuito série $R$–$L$–$C$ com saída sobre o capacitor:

$$L C\,\frac{d^2 v_{out}}{dt^2} + RC\,\frac{dv_{out}}{dt} + v_{out} = v_{in}$$

### 3.2 Função de Transferência

$$\boxed{G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}}$$

com os parâmetros:

$$\omega_n = \frac{1}{\sqrt{LC}} \quad [\text{rad/s}] \qquad \zeta = \frac{R}{2}\sqrt{\frac{C}{L}} \quad [\text{adimensional}]$$

### 3.3 Classificação pelo Amortecimento

| Condição | Regime | Polos $s_{1,2}$ | Resposta ao degrau |
|----------|--------|-----------------|-------------------|
| $\zeta = 0$ | Não amortecido | $\pm j\omega_n$ | Oscilação senoidal permanente |
| $0 < \zeta < 1$ | Subamortecido | $-\zeta\omega_n \pm j\omega_d$ | Oscila e converge |
| $\zeta = 1$ | Criticamente amortecido | $-\omega_n$ (duplo) | Converge sem oscilar, mais rápido |
| $\zeta > 1$ | Superamortecido | dois reais negativos | Converge lentamente sem oscilar |

onde a **frequência natural amortecida** é:

$$\omega_d = \omega_n\sqrt{1 - \zeta^2}$$

### 3.4 Resposta ao Degrau — Caso Subamortecido

$$v_{out}(t) = A\left[1 - e^{-\zeta\omega_n t}\left(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\right)\right]$$

### 3.5 Indicadores de Desempenho

**Sobressinal percentual:**

$$M_p\% = 100\,\exp\!\left(\frac{-\pi\zeta}{\sqrt{1-\zeta^2}}\right)$$

**Tempo de pico:**

$$t_p = \frac{\pi}{\omega_d}$$

**Tempo de acomodação** (critério $2\%$):

$$t_s \approx \frac{4}{\zeta\,\omega_n}$$

**Tempo de subida** (aproximação para $\zeta \approx 0{,}5$):

$$t_r \approx \frac{1{,}8}{\omega_n}$$

### 3.6 Margens de Estabilidade

- **Margem de ganho** $G_M$: quanto o ganho pode crescer antes da instabilidade $[\text{dB}]$. Recomendado: $G_M > 6\,\text{dB}$.
- **Margem de fase** $\phi_M$: quanto atraso de fase pode ser adicionado. Recomendado: $\phi_M > 45°$.

Relação prática para o RLC de segunda ordem:

$$\phi_M \approx 100\,\zeta \quad \text{(para } 0 < \zeta < 0{,}7\text{)}$$

---

## 4. Transformada de Laplace — Referência Rápida

A transformada de Laplace unilateral é definida por:

$$\mathcal{L}\{f(t)\} = F(s) = \int_0^{\infty} f(t)\,e^{-st}\,dt$$

### 4.1 Pares Fundamentais

| $f(t)$, $t \geq 0$ | $F(s)$ |
|--------------------|--------|
| $\delta(t)$ | $1$ |
| $u(t)$ | $\dfrac{1}{s}$ |
| $t$ | $\dfrac{1}{s^2}$ |
| $e^{-at}$ | $\dfrac{1}{s+a}$ |
| $\sin(\omega t)$ | $\dfrac{\omega}{s^2+\omega^2}$ |
| $\cos(\omega t)$ | $\dfrac{s}{s^2+\omega^2}$ |
| $e^{-at}\sin(\omega t)$ | $\dfrac{\omega}{(s+a)^2+\omega^2}$ |
| $e^{-at}\cos(\omega t)$ | $\dfrac{s+a}{(s+a)^2+\omega^2}$ |
| $t^n$ | $\dfrac{n!}{s^{n+1}}$ |

### 4.2 Propriedades Essenciais

**Derivada no tempo:**

$$\mathcal{L}\left\{\frac{d^n f}{dt^n}\right\} = s^n F(s) - s^{n-1}f(0^-) - \cdots - f^{(n-1)}(0^-)$$

**Integral:**

$$\mathcal{L}\left\{\int_0^t f(\tau)\,d\tau\right\} = \frac{F(s)}{s}$$

**Teorema do valor final** (para sistemas estáveis):

$$\lim_{t \to \infty} f(t) = \lim_{s \to 0} s\,F(s)$$

**Teorema do valor inicial:**

$$\lim_{t \to 0^+} f(t) = \lim_{s \to \infty} s\,F(s)$$

---

## 5. Análise de Estabilidade no Plano $s$

### 5.1 Condição de Estabilidade BIBO

Um sistema LTI é **estável** (BIBO) se e somente se todos os polos de $G(s)$ têm parte real estritamente negativa:

$$\text{Re}(p_i) < 0 \quad \forall\, i \qquad \Longleftrightarrow \qquad \text{sistema estável}$$

### 5.2 Localização dos Polos — RLC

$$s_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

- Distância à origem: $|s_{1,2}| = \omega_n$
- Ângulo com o eixo real negativo: $\theta = \arccos(\zeta)$

Para $R, L, C > 0$ o RLC é sempre estável — polos no semiplano esquerdo fechado.

### 5.3 Critério de Routh–Hurwitz

Para o polinômio $a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0$, o sistema é estável se e somente se todos os coeficientes $a_i > 0$ **e** todos os determinantes de Routh são positivos.

Para segunda ordem $s^2 + 2\zeta\omega_n s + \omega_n^2$: estável $\Leftrightarrow$ $\zeta > 0$ e $\omega_n > 0$.

---

## 6. Controladores PID com Amplificador Operacional

### 6.1 Ação Proporcional (P)

Configuração inversora com dois resistores:

$$G_P(s) = -\frac{R_2}{R_1} = -K_p$$

### 6.2 Ação Integral (I)

Substituindo $R_2$ por capacitor $C$:

$$G_I(s) = -\frac{1}{sR_1 C} = -\frac{K_i}{s}, \qquad K_i = \frac{1}{R_1 C}$$

### 6.3 Ação Derivativa (D)

Capacitor na entrada:

$$G_D(s) = -sR_2 C = -K_d\,s, \qquad K_d = R_2 C$$

### 6.4 Controlador PID Completo

$$\boxed{G_{PID}(s) = K_p + \frac{K_i}{s} + K_d\,s = \frac{K_d s^2 + K_p s + K_i}{s}}$$

| Ação | Efeito no transitório | Efeito no regime permanente |
|------|-----------------------|-----------------------------|
| $K_p$ | Reduz tempo de subida | Erro inversamente proporcional a $K_p$ |
| $K_i$ | Pode aumentar sobressinal | Elimina erro estático (tipo 0 → tipo 1) |
| $K_d$ | Reduz sobressinal e oscilação | Não altera erro estático |

---

## 7. Diagrama de Bode — Sistema de Segunda Ordem

Para $G(j\omega) = \dfrac{\omega_n^2}{(j\omega)^2 + 2\zeta\omega_n(j\omega) + \omega_n^2}$:

### 7.1 Magnitude

$$\left|G(j\omega)\right| = \frac{1}{\sqrt{\left[1-\left(\dfrac{\omega}{\omega_n}\right)^2\right]^2 + \left[2\zeta\dfrac{\omega}{\omega_n}\right]^2}}$$

Em dB: $\left|G(j\omega)\right|_{dB} = 20\log_{10}\left|G(j\omega)\right|$

### 7.2 Fase

$$\angle G(j\omega) = -\arctan\!\left(\frac{2\zeta\,\omega/\omega_n}{1 - (\omega/\omega_n)^2}\right)$$

### 7.3 Pico de Ressonância

Para $\zeta < \dfrac{1}{\sqrt{2}} \approx 0{,}707$, existe pico na frequência:

$$\omega_r = \omega_n\sqrt{1 - 2\zeta^2}$$

com valor de pico:

$$M_r = \frac{1}{2\zeta\sqrt{1-\zeta^2}}$$

---

## 8. Referências

- OGATA, K. *Modern Control Engineering*. 5. ed. Prentice Hall, 2010.
- NISE, N. S. *Control Systems Engineering*. 7. ed. Wiley, 2015.
- FRANKLIN, G. F.; POWELL, J. D.; EMAMI-NAEINI, A. *Feedback Control of Dynamic Systems*. 8. ed. Pearson, 2019.
- DORF, R. C.; BISHOP, R. H. *Modern Control Systems*. 13. ed. Pearson, 2017.
