# Teoria de Controle — Sistemas Mecânicos: Massa-Mola-Amortecedor

> Documento de referência para análise e projeto de sistemas mecânicos de translação.
> Fórmulas no padrão LaTeX — compatível com Obsidian, Jupyter, MkDocs e GitHub com MathJax.

---

## 1. Analogia Eletromecânica

A estrutura matemática do sistema massa-mola-amortecedor é **idêntica** à do circuito RLC série. A tabela abaixo estabelece a correspondência direta entre as grandezas dos dois domínios.

| Grandeza elétrica | Símbolo | Grandeza mecânica | Símbolo | Unidade SI |
|-------------------|---------|-------------------|---------|------------|
| Tensão | $v(t)$ | Força | $F(t)$ | N |
| Corrente | $i(t)$ | Velocidade | $\dot{x}(t)$ | m/s |
| Carga elétrica | $q(t)$ | Deslocamento | $x(t)$ | m |
| Resistor | $R$ | Amortecedor | $b$ | N·s/m |
| Indutor | $L$ | Massa | $m$ | kg |
| Capacitor (inverso) | $1/C$ | Mola | $k$ | N/m |

Consequentemente, substituindo $L \to m$, $R \to b$, $1/C \to k$, $v \to F$ e $q \to x$ nas equações do RLC, obtém-se exatamente a equação de movimento mecânica.

---

## 2. Elementos Mecânicos Fundamentais

---

### 2.1 Massa ($m$)

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 90" width="300" height="90">
  <!-- Chão (referencial) -->
  <line x1="20" y1="75" x2="280" y2="75" stroke="#888" stroke-width="2"/>
  <line x1="20" y1="75" x2="30" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="40" y1="75" x2="50" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="60" y1="75" x2="70" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="80" y1="75" x2="90" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="100" y1="75" x2="110" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="120" y1="75" x2="130" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="140" y1="75" x2="150" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="160" y1="75" x2="170" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="180" y1="75" x2="190" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="200" y1="75" x2="210" y2="85" stroke="#888" stroke-width="1.2"/>
  <!-- Massa (bloco) -->
  <rect x="90" y="42" width="80" height="33" rx="4" fill="#E6F1FB" stroke="#185FA5" stroke-width="2"/>
  <text x="130" y="63" text-anchor="middle" font-family="Georgia,serif" font-size="16" fill="#185FA5" font-style="italic">m</text>
  <!-- Seta de força -->
  <line x1="170" y1="58" x2="220" y2="58" stroke="#D85A30" stroke-width="2" marker-end="url(#frc)"/>
  <text x="225" y="54" font-family="Georgia,serif" font-size="13" fill="#D85A30" font-style="italic">F(t)</text>
  <!-- Seta de deslocamento -->
  <line x1="130" y1="35" x2="148" y2="35" stroke="#444" stroke-width="1.3" marker-end="url(#dsl)"/>
  <text x="108" y="31" font-family="Georgia,serif" font-size="11" fill="#444" font-style="italic">x(t)</text>
  <defs>
    <marker id="frc" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#D85A30"/>
    </marker>
    <marker id="dsl" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#444"/>
    </marker>
  </defs>
</svg>

A massa representa a **inércia translacional** do sistema. Pela Segunda Lei de Newton:

$$F(t) = m\,\ddot{x}(t) = m\,\frac{d^2x(t)}{dt^2}$$

No domínio de Laplace (condições iniciais nulas):

$$F(s) = m\,s^2\,X(s) \qquad \Longrightarrow \qquad Z_m(s) = ms^2$$

**Energia cinética armazenada:**

$$E_{cin} = \frac{1}{2}\,m\,\dot{x}^2(t)$$

---

### 2.2 Mola ($k$)

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 90" width="300" height="90">
  <!-- Chão -->
  <line x1="20" y1="75" x2="100" y2="75" stroke="#888" stroke-width="2"/>
  <line x1="20" y1="75" x2="30" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="40" y1="75" x2="50" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="60" y1="75" x2="70" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="80" y1="75" x2="90" y2="85" stroke="#888" stroke-width="1.2"/>
  <!-- Mola (vertical) -->
  <line x1="60" y1="75" x2="60" y2="65" stroke="#0F6E56" stroke-width="1.8"/>
  <polyline points="60,65 52,60 68,52 52,44 68,36 52,28 68,20 60,15" fill="none" stroke="#0F6E56" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="60" y1="15" x2="60" y2="8" stroke="#0F6E56" stroke-width="1.8"/>
  <!-- Bloco superior -->
  <rect x="35" y="1" width="50" height="10" rx="2" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.5"/>
  <!-- Seta de força -->
  <line x1="95" y1="42" x2="130" y2="42" stroke="#D85A30" stroke-width="2" marker-end="url(#frc2)"/>
  <text x="135" y="38" font-family="Georgia,serif" font-size="13" fill="#D85A30" font-style="italic">F</text>
  <!-- Label k -->
  <text x="72" y="44" font-family="Georgia,serif" font-size="14" fill="#0F6E56" font-style="italic">k</text>
  <!-- Seta deslocamento -->
  <line x1="165" y1="8" x2="165" y2="22" stroke="#444" stroke-width="1.3" marker-end="url(#dsl2)"/>
  <text x="170" y="18" font-family="Georgia,serif" font-size="11" fill="#444" font-style="italic">x</text>
  <defs>
    <marker id="frc2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#D85A30"/>
    </marker>
    <marker id="dsl2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#444"/>
    </marker>
  </defs>
</svg>

A mola é o elemento elástico — armazena energia potencial e obedece à **Lei de Hooke**:

$$F(t) = k\,x(t)$$

No domínio de Laplace:

$$F(s) = k\,X(s) \qquad \Longrightarrow \qquad Z_k(s) = k$$

**Energia potencial armazenada:**

$$E_{pot} = \frac{1}{2}\,k\,x^2(t)$$

A constante $k$ é chamada de **rigidez** ou **coeficiente de mola**. Quanto maior $k$, maior a frequência natural $\omega_n$.

---

### 2.3 Amortecedor Viscoso ($b$)

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 90" width="300" height="90">
  <!-- Chão -->
  <line x1="20" y1="75" x2="100" y2="75" stroke="#888" stroke-width="2"/>
  <line x1="20" y1="75" x2="30" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="40" y1="75" x2="50" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="60" y1="75" x2="70" y2="85" stroke="#888" stroke-width="1.2"/>
  <line x1="80" y1="75" x2="90" y2="85" stroke="#888" stroke-width="1.2"/>
  <!-- Amortecedor (corpo) -->
  <line x1="60" y1="75" x2="60" y2="60" stroke="#854F0B" stroke-width="1.8"/>
  <rect x="44" y="38" width="32" height="22" rx="2" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.8"/>
  <!-- Pistão interno -->
  <line x1="60" y1="38" x2="60" y2="20" stroke="#854F0B" stroke-width="3" stroke-linecap="round"/>
  <rect x="50" y="28" width="20" height="6" rx="1" fill="#854F0B"/>
  <!-- Bloco superior -->
  <line x1="60" y1="20" x2="60" y2="8" stroke="#854F0B" stroke-width="1.8"/>
  <rect x="35" y="1" width="50" height="10" rx="2" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.5"/>
  <!-- Seta de força -->
  <line x1="95" y1="42" x2="135" y2="42" stroke="#D85A30" stroke-width="2" marker-end="url(#frc3)"/>
  <text x="140" y="38" font-family="Georgia,serif" font-size="13" fill="#D85A30" font-style="italic">F</text>
  <!-- Label b -->
  <text x="80" y="52" font-family="Georgia,serif" font-size="14" fill="#854F0B" font-style="italic">b</text>
  <!-- Seta velocidade -->
  <line x1="170" y1="8" x2="170" y2="22" stroke="#444" stroke-width="1.3" marker-end="url(#dsl3)"/>
  <text x="175" y="18" font-family="Georgia,serif" font-size="11" fill="#444" font-style="italic">ẋ</text>
  <defs>
    <marker id="frc3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#D85A30"/>
    </marker>
    <marker id="dsl3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#444"/>
    </marker>
  </defs>
</svg>

O amortecedor dissipa energia de forma proporcional à velocidade — análogo ao resistor:

$$F(t) = b\,\dot{x}(t) = b\,\frac{dx(t)}{dt}$$

No domínio de Laplace:

$$F(s) = b\,s\,X(s) \qquad \Longrightarrow \qquad Z_b(s) = bs$$

**Potência dissipada:**

$$P_{diss}(t) = b\,\dot{x}^2(t) \geq 0$$

O amortecedor não armazena energia — apenas a converte em calor.

---

### 2.4 Sistema Completo — Massa-Mola-Amortecedor

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 130" width="400" height="130">
  <!-- Chão -->
  <line x1="20" y1="110" x2="380" y2="110" stroke="#888" stroke-width="2"/>
  <line x1="20"  y1="110" x2="32"  y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="44"  y1="110" x2="56"  y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="68"  y1="110" x2="80"  y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="92"  y1="110" x2="104" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="116" y1="110" x2="128" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="140" y1="110" x2="152" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="164" y1="110" x2="176" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="188" y1="110" x2="200" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="212" y1="110" x2="224" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="236" y1="110" x2="248" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="260" y1="110" x2="272" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="284" y1="110" x2="296" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="308" y1="110" x2="320" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="332" y1="110" x2="344" y2="122" stroke="#888" stroke-width="1.2"/>
  <line x1="356" y1="110" x2="368" y2="122" stroke="#888" stroke-width="1.2"/>
  <!-- Mola (esquerda) -->
  <line x1="70" y1="110" x2="70" y2="98" stroke="#0F6E56" stroke-width="1.8"/>
  <polyline points="70,98 62,93 78,85 62,77 78,69 62,61 78,53 70,48" fill="none" stroke="#0F6E56" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="70" y1="48" x2="70" y2="42" stroke="#0F6E56" stroke-width="1.8"/>
  <text x="82" y="78" font-family="Georgia,serif" font-size="13" fill="#0F6E56" font-style="italic">k</text>
  <!-- Amortecedor (direita) -->
  <line x1="160" y1="110" x2="160" y2="95" stroke="#854F0B" stroke-width="1.8"/>
  <rect x="146" y="73" width="28" height="22" rx="2" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.8"/>
  <line x1="160" y1="73" x2="160" y2="58" stroke="#854F0B" stroke-width="3" stroke-linecap="round"/>
  <rect x="150" y="62" width="20" height="6" rx="1" fill="#854F0B"/>
  <line x1="160" y1="58" x2="160" y2="42" stroke="#854F0B" stroke-width="1.8"/>
  <text x="176" y="88" font-family="Georgia,serif" font-size="13" fill="#854F0B" font-style="italic">b</text>
  <!-- Massa (bloco) -->
  <rect x="50" y="20" width="130" height="25" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="2"/>
  <text x="115" y="37" text-anchor="middle" font-family="Georgia,serif" font-size="16" fill="#185FA5" font-style="italic">m</text>
  <!-- Seta de força -->
  <line x1="180" y1="32" x2="240" y2="32" stroke="#D85A30" stroke-width="2.2" marker-end="url(#frc4)"/>
  <text x="245" y="28" font-family="Georgia,serif" font-size="13" fill="#D85A30" font-style="italic">F(t)</text>
  <!-- Seta deslocamento -->
  <line x1="100" y1="12" x2="128" y2="12" stroke="#444" stroke-width="1.3" marker-end="url(#dsl4)"/>
  <text x="72" y="16" font-family="Georgia,serif" font-size="11" fill="#444" font-style="italic">x(t) →</text>
  <defs>
    <marker id="frc4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#D85A30"/>
    </marker>
    <marker id="dsl4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,1 L7,4 L0,7" fill="#444"/>
    </marker>
  </defs>
</svg>

---

### 2.5 Resumo dos Elementos Mecânicos

| Elemento | Lei constitutiva | Impedância $Z(s)$ | Energia |
|----------|-----------------|-------------------|---------| 
| Massa $m$ | $F = m\ddot{x}$ | $ms^2$ | $\frac{1}{2}m\dot{x}^2$ |
| Mola $k$ | $F = kx$ | $k$ | $\frac{1}{2}kx^2$ |
| Amortecedor $b$ | $F = b\dot{x}$ | $bs$ | $0$ (dissipa) |

---

## 3. Equação de Movimento e Função de Transferência

### 3.1 Sistema Padrão com 1 Grau de Liberdade

Massa $m$ conectada ao referencial por mola $k$ e amortecedor $b$ em paralelo, sujeita a força externa $F(t)$.

Aplicando a Segunda Lei de Newton ($\sum F = m\ddot{x}$):

$$F(t) - k\,x(t) - b\,\dot{x}(t) = m\,\ddot{x}(t)$$

Reorganizando na **forma canônica**:

$$\boxed{m\,\ddot{x}(t) + b\,\dot{x}(t) + k\,x(t) = F(t)}$$

### 3.2 Função de Transferência

Aplicando a Transformada de Laplace com condições iniciais nulas:

$$(ms^2 + bs + k)\,X(s) = F(s)$$

$$\boxed{G(s) = \frac{X(s)}{F(s)} = \frac{1/m}{s^2 + \dfrac{b}{m}\,s + \dfrac{k}{m}}}$$

### 3.3 Forma Padrão de Segunda Ordem

Identificando com $G(s) = \dfrac{\omega_n^2/k}{s^2 + 2\zeta\omega_n s + \omega_n^2}$:

$$\boxed{\omega_n = \sqrt{\frac{k}{m}}} \quad [\text{rad/s}] \qquad \boxed{\zeta = \frac{b}{2\sqrt{km}} = \frac{b}{2m\omega_n}} \quad [\text{adimensional}]$$

$$\omega_d = \omega_n\sqrt{1 - \zeta^2} \quad [\text{rad/s}] \qquad \text{(frequência amortecida)}$$

---

## 4. Parâmetros Físicos e Interpretação

### 4.1 Frequência Natural $\omega_n$

Frequência de oscilação livre do sistema **sem amortecimento** ($b = 0$):

$$\omega_n = \sqrt{\frac{k}{m}} \qquad f_n = \frac{\omega_n}{2\pi} \quad [\text{Hz}] \qquad T_n = \frac{2\pi}{\omega_n} \quad [\text{s}]$$

Representa a taxa de troca de energia entre a mola (potencial) e a massa (cinética). Aumentar $k$ eleva $\omega_n$; aumentar $m$ reduz $\omega_n$.

### 4.2 Amortecimento Crítico

O amortecimento crítico $b_c$ é o menor valor de $b$ que elimina a oscilação:

$$b_c = 2\sqrt{km} = 2m\omega_n$$

O fator de amortecimento é a razão:

$$\zeta = \frac{b}{b_c}$$

### 4.3 Classificação do Comportamento

| Condição | Regime | Polos $s_{1,2}$ | Comportamento físico |
|----------|--------|-----------------|----------------------|
| $\zeta = 0$ | Não amortecido | $\pm j\omega_n$ | Oscilação senoidal permanente |
| $0 < \zeta < 1$ | Subamortecido | $-\zeta\omega_n \pm j\omega_d$ | Oscila e converge ao equilíbrio |
| $\zeta = 1$ | Criticamente amortecido | $-\omega_n$ (duplo) | Retorno mais rápido sem ultrapassar |
| $\zeta > 1$ | Superamortecido | dois reais $< 0$ | Retorno lento e monotônico |

---

## 5. Resposta ao Degrau de Força

Para $F(t) = F_0\,u(t)$, o deslocamento em Laplace é:

$$X(s) = \frac{F_0/m}{s\left(s^2 + \dfrac{b}{m}s + \dfrac{k}{m}\right)}$$

**Valor final** pelo Teorema do Valor Final:

$$x(\infty) = \lim_{s \to 0} s\,X(s) = \frac{F_0}{k}$$

Fisicamente: em regime permanente toda a força é suportada pela mola ($x = F_0/k$); o amortecedor não contribui pois a velocidade é nula.

### 5.1 Caso Subamortecido ($0 < \zeta < 1$)

$$x(t) = \frac{F_0}{k}\left[1 - e^{-\zeta\omega_n t}\left(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\right)\right]$$

### 5.2 Caso Criticamente Amortecido ($\zeta = 1$)

$$x(t) = \frac{F_0}{k}\left[1 - e^{-\omega_n t}\left(1 + \omega_n t\right)\right]$$

### 5.3 Caso Superamortecido ($\zeta > 1$)

Polos reais distintos:

$$s_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

A resposta é uma combinação de duas exponenciais. O polo **menos negativo** $s_1$ domina o comportamento em regime transitório tardio.

---

## 6. Indicadores de Desempenho

### 6.1 Sobressinal Percentual

$$\boxed{M_p\% = 100\,\exp\!\left(\frac{-\pi\zeta}{\sqrt{1-\zeta^2}}\right)} \qquad (0 < \zeta < 1)$$

| $\zeta$ | $M_p\%$ |
|---------|---------|
| $0{,}1$ | $72{,}9\%$ |
| $0{,}2$ | $52{,}7\%$ |
| $0{,}3$ | $37{,}2\%$ |
| $0{,}5$ | $16{,}3\%$ |
| $0{,}7$ | $4{,}6\%$ |
| $0{,}9$ | $0{,}2\%$ |
| $\geq 1{,}0$ | $0\%$ |

**Regra prática:** projetos de engenharia mecânica utilizam tipicamente $0{,}4 \leq \zeta \leq 0{,}7$.

### 6.2 Tempo de Pico $t_p$

Instante em que a resposta atinge o máximo:

$$t_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n\sqrt{1-\zeta^2}}$$

### 6.3 Tempo de Acomodação $t_s$

$$t_s \approx \frac{4}{\zeta\,\omega_n} \quad \text{(critério } \pm 2\text{\%)} \qquad t_s \approx \frac{3}{\zeta\,\omega_n} \quad \text{(critério } \pm 5\text{\%)}$$

### 6.4 Tempo de Subida $t_r$ (10% → 90%)

$$t_r \approx \frac{1 - 0{,}4169\zeta + 2{,}917\zeta^2}{\omega_n}$$

### 6.5 Especificação Inversa

Dadas as especificações $M_p\%$ e $t_s$, calcular os parâmetros:

$$\zeta = \frac{-\ln(M_p/100)}{\sqrt{\pi^2 + \ln^2(M_p/100)}} \qquad \omega_n = \frac{4}{\zeta\,t_s}$$

---

## 7. Resposta Livre (Condições Iniciais $\neq 0$)

Para $x(0) = x_0$, $\dot{x}(0) = v_0$ e $F(t) = 0$:

### 7.1 Caso Subamortecido

$$x(t) = e^{-\zeta\omega_n t}\left[x_0\cos(\omega_d t) + \frac{v_0 + \zeta\omega_n x_0}{\omega_d}\sin(\omega_d t)\right]$$

### 7.2 Caso Não Amortecido ($\zeta = 0$)

$$x(t) = x_0\cos(\omega_n t) + \frac{v_0}{\omega_n}\sin(\omega_n t)$$

Energia total conservada:

$$E = \frac{1}{2}m\dot{x}^2 + \frac{1}{2}kx^2 = \frac{1}{2}kx_0^2 + \frac{1}{2}mv_0^2 = \text{constante}$$

---

## 8. Análise no Plano $s$

### 8.1 Geometria dos Polos

$$s_{1,2} = -\zeta\omega_n \pm j\omega_d$$

- Distância à origem: $|s_{1,2}| = \omega_n$
- Ângulo com o eixo real negativo: $\theta = \arccos(\zeta)$
- Parte real: $\sigma = \zeta\omega_n = 4/t_s$ (define velocidade de decaimento)
- Parte imaginária: $\omega_d = \omega_n\sqrt{1-\zeta^2} = \pi/t_p$ (define frequência de oscilação)

### 8.2 Lugar das Raízes — Variando $b$

Com $m$ e $k$ fixos, ao variar $b$ de $0$ a $\infty$:

$$b = 0 \;\Rightarrow\; s = \pm j\omega_n \qquad b = b_c \;\Rightarrow\; s = -\omega_n \text{ (duplo)} \qquad b \to \infty \;\Rightarrow\; s_1 \to 0,\; s_2 \to -\infty$$

Para $m, b, k > 0$: todos os polos têm $\text{Re}(s) < 0$ — sistema **sempre estável**.

---

## 9. Balanço Energético

### 9.1 Equação de Energia

Multiplicando a equação de movimento por $\dot{x}$:

$$m\ddot{x}\dot{x} + b\dot{x}^2 + kx\dot{x} = F\dot{x}$$

Integrando no tempo:

$$\underbrace{\frac{d}{dt}\left(\frac{1}{2}m\dot{x}^2 + \frac{1}{2}kx^2\right)}_{\dot{E}_{total}} = \underbrace{F\dot{x}}_{P_{entrada}} - \underbrace{b\dot{x}^2}_{P_{dissipada}}$$

### 9.2 Fator de Qualidade $Q$

$$Q = \frac{\omega_n\,m}{b} = \frac{1}{2\zeta}$$

Um $Q$ alto indica pouco amortecimento (sistema ressoa bem); um $Q$ baixo indica amortecimento elevado.

---

## 10. Controle do Sistema Mecânico

### 10.1 Controlador Proporcional (P)

Realimentação proporcional do deslocamento:

$$F(t) = K_p\left[x_{ref}(t) - x(t)\right]$$

Sistema em malha fechada:

$$G_{MF}(s) = \frac{K_p/m}{s^2 + \dfrac{b}{m}s + \dfrac{k + K_p}{m}}$$

Parâmetros equivalentes: $\omega_{n,MF} = \sqrt{\dfrac{k+K_p}{m}}$, $\zeta_{MF} = \dfrac{b}{2\sqrt{m(k+K_p)}}$

Aumentar $K_p$ eleva $\omega_n$ mas reduz $\zeta$ — mais rápido, porém mais oscilatório.

### 10.2 Controlador PD

$$F(t) = K_p\,e(t) + K_d\,\dot{e}(t), \qquad e = x_{ref} - x$$

Amortecimento equivalente: $b_{eq} = b + K_d$

$$\zeta_{MF} = \frac{b + K_d}{2\sqrt{m(k+K_p)}}$$

$K_d$ aumenta o amortecimento efetivo, reduzindo o sobressinal sem alterar (significativamente) $\omega_n$.

### 10.3 Controlador PID

$$F(t) = K_p\,e + K_i\int e\,dt + K_d\,\dot{e}$$

Função de transferência do controlador:

$$C(s) = K_p + \frac{K_i}{s} + K_d s = \frac{K_d s^2 + K_p s + K_i}{s}$$

**Sintonia de Ziegler–Nichols** (método da oscilação):

$$K_p = 0{,}6\,K_{pu} \qquad K_i = \frac{K_p}{0{,}5\,T_u} \qquad K_d = 0{,}125\,K_p\,T_u$$

onde $K_{pu}$ é o ganho de oscilação e $T_u$ o período de oscilação no limite de estabilidade.

---

## 11. Diagrama de Bode — Sistema Mecânico

Para $G(j\omega) = \dfrac{1/k}{\left[1-\left(\dfrac{\omega}{\omega_n}\right)^2\right] + j\left[2\zeta\dfrac{\omega}{\omega_n}\right]}$:

### 11.1 Três Regiões de Frequência

**Região de rigidez** ($\omega \ll \omega_n$):

$$|G(j\omega)| \approx \frac{1}{k}, \qquad \angle G \approx 0°$$

**Região de ressonância** ($\omega \approx \omega_n$):

$$|G(j\omega_n)| = \frac{1}{b\,\omega_n} = \frac{Q}{k}, \qquad \angle G = -90°$$

**Região de inércia** ($\omega \gg \omega_n$):

$$|G(j\omega)| \approx \frac{1}{m\omega^2} \quad (-40\,\text{dB/déc}), \qquad \angle G \to -180°$$

### 11.2 Frequência de Ressonância e Pico

Para $\zeta < \dfrac{1}{\sqrt{2}}$:

$$\omega_r = \omega_n\sqrt{1 - 2\zeta^2} \qquad |G(j\omega_r)|_{max} = \frac{1}{2\zeta k\sqrt{1-\zeta^2}}$$

---

## 12. Sistemas Físicos Equivalentes

| Sistema físico | $m$ (inércia) | $k$ (rigidez) | $b$ (dissipação) |
|----------------|---------------|----------------|------------------|
| Suspensão automotiva | Carroceria | Mola de suspensão | Amortecedor hidráulico |
| Braço robótico | Braço + carga | Rigidez da junta | Atrito viscoso |
| Edifício sob sismo | Laje do andar | Rigidez estrutural | Dissipadores viscoelásticos |
| Membrana de microfone | Membrana | Tensão superficial | Resistência do ar |
| Equivalente elétrico | $L$ | $1/C$ | $R$ |

---

## 13. Sistema com Dois Graus de Liberdade

Para duas massas $m_1$ e $m_2$ acopladas, as equações de movimento formam o sistema:

$$m_1\ddot{x}_1 + (b_1+b_2)\dot{x}_1 - b_2\dot{x}_2 + (k_1+k_2)x_1 - k_2 x_2 = F_1$$

$$m_2\ddot{x}_2 - b_2\dot{x}_1 + b_2\dot{x}_2 - k_2 x_1 + k_2 x_2 = F_2$$

Na **forma matricial**:

$$\mathbf{M}\,\ddot{\mathbf{x}} + \mathbf{B}\,\dot{\mathbf{x}} + \mathbf{K}\,\mathbf{x} = \mathbf{F}$$

As **frequências naturais** são as raízes do problema de autovalores generalizado:

$$\det\!\left(\mathbf{K} - \omega_n^2\,\mathbf{M}\right) = 0$$

---

## 14. Referências

- OGATA, K. *System Dynamics*. 4. ed. Pearson, 2004.
- PALM III, W. J. *System Dynamics*. 3. ed. McGraw-Hill, 2014.
- INMAN, D. J. *Engineering Vibration*. 4. ed. Pearson, 2014.
- RAO, S. S. *Mechanical Vibrations*. 6. ed. Pearson, 2017.
- DE SILVA, C. W. *Vibration: Fundamentals and Practice*. 2. ed. CRC Press, 2006.
