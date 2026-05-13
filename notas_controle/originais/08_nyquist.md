# 🎛️ 8. Critério de Nyquist

## 📘 O que é o Critério de Nyquist?

O **Critério de Nyquist** é um método gráfico de análise de estabilidade que relaciona:
- A **resposta em frequência** da malha aberta $G(j\omega)H(j\omega)$
- A **estabilidade** do sistema em malha fechada

É análogo ao Lugar das Raízes (que usa pólos e zeros), mas opera no domínio da frequência, sendo especialmente útil quando o sistema possui **atraso de transporte** ou é definido por dados experimentais.


---


## 📘 Conceito Fundamental

O diagrama de Nyquist traça a curva de $G(j\omega)H(j\omega)$ no plano complexo para $\omega$ variando de $-\infty$ a $+\infty$ (ou de $0$ a $+\infty$, já que a curva é simétrica para sistemas reais).

### ⚡ Critério de Nyquist (forma simplificada):

Para um sistema com $P$ pólos de malha aberta no semiplano direito (SPD):

$$Z = N + P$$

onde:
- $Z$ = número de zeros de $(1+GH)$ no SPD = **pólos de malha fechada no SPD**
- $N$ = número de voltas (encirclement) do ponto $(-1, 0)$ no sentido horário
- $P$ = número de pólos de $GH$ no SPD

**Para estabilidade:** $Z = 0$ (nenhum polo de MF no SPD), o que exige $N = -P$.


---


## 📘 Interpretação Gráfica

### ⚡ Ponto crítico $(-1, j0)$

O ponto $-1$ no eixo real do diagrama de Nyquist é o **ponto crítico**. A posição da curva em relação a esse ponto determina a estabilidade:

| Situação | Estabilidade |
| ----------| -------------|
| Curva **não encircla** $(-1,0)$ e $P=0$ | Sistema **estável** |
| Curva **encircla** $(-1,0)$ uma vez em sentido horário e $P=0$ | 2 polos de MF no SPD — **instável** |
| Curva **passa pelo** $(-1,0)$ | Sistema **marginalmente estável** |

### ⚡ Efeito do ganho

À medida que o ganho $k$ é variado:
- O diagrama de Nyquist **expande** (ganho maior) ou **encolhe** (ganho menor) como um balão
- O ponto crítico $(-1,0)$ fica dentro ou fora da curva conforme $k$ muda


---


## 📘 Relação com Margens de Estabilidade

Do diagrama de Nyquist derivam-se as margens de estabilidade:

### ⚡ Margem de fase (PM)
Ângulo entre a curva de Nyquist e o eixo negativo real quando $|G(j\omega)H(j\omega)| = 1$:

$$PM = 180° + \angle G(j\omega_{gc})H(j\omega_{gc})$$

### ⚡ Margem de ganho (GM)
Recíproco do módulo de $GH$ quando a curva cruza o eixo real negativo:

$$GM = \frac{1}{|G(j\omega_{pc})H(j\omega_{pc})|}$$

> [!tip]
> 💡 O diagrama de Nyquist e o de Bode contêm a mesma informação, mas em formatos diferentes. O Bode separa magnitude e fase; o Nyquist os combina em um único diagrama polar.


---


## 📘 Desvio em torno de Pólos no Eixo Imaginário

Se a função $GH$ possui pólos sobre o eixo imaginário (incluindo a origem), a curva de Nyquist não pode ser traçada diretamente nesses pontos (iria ao infinito).

**Solução:** fazer um **desvio semicircular** de raio $\varepsilon \to 0$ em torno do polo, criando um contorno que contorna o polo pelo SPD. Isso gera um arco de grande raio no diagrama de Nyquist.

### ⚡ Para polo na origem $\frac{1}{s}$:
- O desvio gera um arco de raio $\to \infty$ com rotação de **+90°** (sentido anti-horário) no diagrama de Nyquist


---


## 📘 Exemplos

### ⚡ Exemplo 1 — Planta estável sem polo no SPD ($P = 0$)

$$G(s) = \frac{k}{(s+1)(s+2)(s+3)}$$

Para $k$ pequeno: curva não encircla $(-1,0)$ → **estável** ($Z = 0$).

Para $k$ muito grande: curva encircla $(-1,0)$ → **instável** ($Z = 2$).

O valor crítico de $k$ é encontrado quando a curva passa exatamente pelo ponto $(-1,0)$, correspondendo à **margem de ganho**.


---


### ⚡ Exemplo 2 — Sistema com polo na origem

$$G(s) = \frac{k}{s(s+a)(s+b)}$$

O polo na origem exige um desvio semicircular. O diagrama de Nyquist terá:
- Um arco de raio infinito para $\omega \to 0$ (do desvio)
- A curva convencional para $\omega > 0$

Para estabilidade, deve-se verificar que a curva não encircla $(-1,0)$ para o ganho escolhido.


---


### ⚡ Exemplo 3 — Sistema instável em malha aberta ($P \neq 0$)

Se $G(s)$ tem $P = 2$ polos no SPD, para que o sistema em MF seja estável, o diagrama de Nyquist deve encirclar $(-1,0)$ exatamente **2 vezes no sentido anti-horário** ($N = -2$), de modo que $Z = N + P = -2 + 2 = 0$.


---


## 📘 Procedimento para Análise de Nyquist

1. **Identificar** os pólos e zeros de $GH$ e determinar $P$
2. **Verificar** se há pólos no eixo imaginário → planejar desvios
3. **Traçar** o diagrama de Nyquist para $\omega \in [0, +\infty)$
4. **Completar** com o espelho (parte negativa) — simétrico ao eixo real
5. **Contar** os encirclements $N$ ao redor de $(-1,0)$
6. **Calcular** $Z = N + P$ e verificar estabilidade


---


## 📘 Comparação: Bode × Nyquist × Routh-Hurwitz

| Critério | Tipo | Vantagem | Limitação |
| ----------| ------| ----------| -----------|
| **Routh-Hurwitz** | Algébrico | Exato, analítico | Só para sistemas racionais, sem atraso |
| **Lugar das Raízes** | Gráfico (polo-zero) | Visualização intuitiva | Requer modelo analítico |
| **Bode** | Frequência (magnitude/fase) | Fácil de esboçar, margens claras | Menos preciso para instabilidade condicional |
| **Nyquist** | Frequência (polar) | Rigoroso para qualquer sistema | Mais trabalhoso de traçar |


---


## 📘 Estabilidade Condicional

Em alguns sistemas, o aumento do ganho pode:
1. Primeiro **estabilizar** o sistema (instável para $k$ pequeno)
2. Depois **desestabilizar** novamente (instável para $k$ grande)

Esses sistemas têm **duas margens de ganho** e são chamados de **condicionalmente estáveis**. O critério de Nyquist os identifica corretamente; o simples critério de Bode pode falhar nesses casos.


---


## 📘 Código Scilab — Diagrama de Nyquist

```scilab
clear; clc; clf();
s=%s;
num=10; den=s^2+5*s+25;
G=syslin('c', num, den);

% Resposta em frequência (polar plot)
[frq, respf] = repfreq(G, 0.0001, 500, 0.0001);
[db, phi] = dbphi(respf);
plot2d(phi, db);
xgrid;
xlabel("Fase em °")
ylabel("Magnitude em dB")
gcf().figure_name = "Magnitude por Fase (Nyquist-like)";
```


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
