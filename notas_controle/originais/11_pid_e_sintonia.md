# 🎛️ 11. Controladores PID e Sintonia

## 📘 A Estrutura PID

O PID é o controlador mais usado na indústria. Ele forma o sinal de controle a partir de três leituras do mesmo erro $e(t) = r(t) - y(t)$:

$$u(t) = K_p\,e(t) + K_i\int_0^t e(\tau)\,d\tau + K_d\,\frac{de(t)}{dt}$$

No domínio de Laplace:

$$G_{PID}(s) = K_p + \frac{K_i}{s} + K_d s = \frac{K_d s^2 + K_p s + K_i}{s}$$

Cada termo olha o erro em um tempo diferente: o proporcional olha o **presente**, o integral acumula o **passado** e o derivativo antecipa o **futuro**.


---


## 📘 As Três Formas de Escrever o Mesmo Controlador

### ⚡ Forma paralela (a mostrada acima)

$$G(s) = K_p + \frac{K_i}{s} + K_d s$$

### ⚡ Forma ideal (ou padrão, ISA)

$$G(s) = K_p\left(1 + \frac{1}{T_i s} + T_d s\right)$$

com $T_i = K_p/K_i$ (tempo integral) e $T_d = K_d/K_p$ (tempo derivativo).

### ⚡ Forma série (ou interativa, típica de instrumentos antigos)

$$G(s) = K_p'\left(1 + \frac{1}{T_i' s}\right)(1 + T_d' s)$$

> [!warning]
> ⚠️ Antes de aplicar qualquer tabela de sintonia, confira qual forma o seu equipamento usa. Os mesmos números em formas diferentes produzem controladores diferentes.


---


## 📘 Efeito de Cada Ação

| Ação | Tempo de subida | Sobressinal | Tempo de acomodação | Erro em regime | Estabilidade |
| ------| ----------------| ------------| --------------------| ---------------| -------------|
| Aumentar $K_p$ | Diminui | Aumenta | Muda pouco | Diminui | Piora |
| Aumentar $K_i$ | Diminui | Aumenta | Aumenta | **Elimina** | Piora |
| Aumentar $K_d$ | Muda pouco | Diminui | Diminui | Sem efeito | Melhora |

### ⚡ Por que o integrador zera o erro

O termo $K_i/s$ acrescenta um polo na origem, elevando o **tipo** do sistema. Um sistema tipo 0 vira tipo 1, e o erro ao degrau passa de $1/(1+K_p)$ para zero. Enquanto houver erro, a integral continua crescendo e empurrando o atuador. Só para de crescer quando o erro chega a zero.

### ⚡ Por que o derivativo amortece

O termo $K_d s$ reage à **velocidade** do erro, não ao seu valor. Quando a saída se aproxima rápido demais da referência, o derivativo já começa a frear antes de a referência ser ultrapassada. É um amortecimento antecipado.


---


## 📘 Os Dois Problemas Práticos do PID

### ⚡ 1. Saturação do integrador (*windup*)

Se o atuador satura, o erro persiste, mas a integral continua acumulando um valor que o atuador não consegue entregar. Quando o erro finalmente inverte, essa carga acumulada demora a se desfazer e a saída dá um sobressinal enorme.

**Soluções:**

- **Parada condicional**: interrompe a integração enquanto o sinal estiver saturado
- **Retrocálculo (*back-calculation*)**: realimenta a diferença entre o sinal calculado e o entregue, descarregando a integral
- **Limitação direta**: satura o próprio termo integral

### ⚡ 2. Ruído no derivativo

A derivada amplifica ruído de medição: um sinal pequeno e rápido vira uma correção grande. Duas correções combinadas resolvem:

**Derivativo filtrado**, que limita o ganho em alta frequência a $N$ (tipicamente entre 8 e 20):

$$G_D(s) = \frac{K_d s}{1 + \dfrac{K_d}{K_p N}s}$$

**Derivada sobre a medição**, não sobre o erro. Como $\frac{d}{dt}(r - y) = -\frac{dy}{dt}$ para referência constante, usar $-dy/dt$ evita o pico violento no instante em que a referência muda (o chamado *derivative kick*), sem alterar a rejeição de perturbação.


---


## 📘 Sintonia por Ziegler-Nichols

Duas receitas clássicas de 1942. São rápidas e servem de **ponto de partida**, não de resposta final.

### ⚡ Método 1 — Curva de reação (malha aberta)

Aplique um degrau na planta em malha aberta e ajuste uma reta na inclinação máxima da resposta. Extraia:

- $L$ = atraso aparente (onde a reta cruza o eixo do tempo)
- $T$ = constante de tempo aparente
- $K$ = ganho estático

| Controlador | $K_p$ | $T_i$ | $T_d$ |
| ------------| ------| ------| ------|
| P | $T/L$ | — | — |
| PI | $0{,}9\,T/L$ | $L/0{,}3$ | — |
| PID | $1{,}2\,T/L$ | $2L$ | $0{,}5L$ |

### ⚡ Método 2 — Ganho crítico (malha fechada)

Com apenas ação proporcional, aumente $K_p$ até a saída oscilar com amplitude constante. Anote o ganho crítico $K_u$ e o período de oscilação $T_u$.

| Controlador | $K_p$ | $T_i$ | $T_d$ |
| ------------| ------| ------| ------|
| P | $0{,}5\,K_u$ | — | — |
| PI | $0{,}45\,K_u$ | $T_u/1{,}2$ | — |
| PID | $0{,}6\,K_u$ | $T_u/2$ | $T_u/8$ |

> [!warning]
> ⚠️ O método do ganho crítico leva a planta deliberadamente à beira da instabilidade. Não use em processos onde oscilar é caro ou perigoso.

A sintonia de Ziegler-Nichols mira o decaimento de um quarto de amplitude, o que costuma dar sobressinal alto, na faixa de 25% a 50%. É agressiva para a maioria dos processos.


---


## 📘 Outras Regras de Sintonia

### ⚡ CHR, com foco em resposta sem sobressinal

| Controlador | $K_p$ | $T_i$ | $T_d$ |
| ------------| ------| ------| ------|
| PI | $0{,}35\,T/(KL)$ | $1{,}2\,T$ | — |
| PID | $0{,}6\,T/(KL)$ | $T$ | $0{,}5L$ |

### ⚡ IMC, sintonia por modelo interno

Para uma planta de primeira ordem com atraso, $G(s) = \dfrac{K e^{-Ls}}{Ts+1}$:

$$K_p = \frac{T + L/2}{K(\lambda + L/2)}, \qquad T_i = T + \frac{L}{2}, \qquad T_d = \frac{TL}{2T + L}$$

O parâmetro $\lambda$ é a constante de tempo desejada em malha fechada. Ele é o botão único de agressividade: $\lambda$ pequeno dá resposta rápida e nervosa, $\lambda$ grande dá resposta lenta e robusta. Uma escolha comum é $\lambda \approx T$.

> [!tip]
> 💡 O IMC é o método preferido em processos industriais justamente por ter um só parâmetro de ajuste com significado físico claro.


---


## 📘 Sintonia Manual, na Ordem Certa

Quando não há modelo disponível:

1. Zere $K_i$ e $K_d$. Aumente $K_p$ até a resposta ficar rápida, aceitando um pouco de oscilação.
2. Reduza $K_p$ em cerca de 30% para dar margem.
3. Aumente $K_i$ até o erro em regime desaparecer em tempo aceitável. Se aparecer oscilação lenta, diminua.
4. Só então aumente $K_d$, se houver sobressinal a combater. Se o atuador começar a chiar, o derivativo está alto ou falta filtro.
5. Teste a rejeição de perturbação, não só a resposta à referência. São requisitos diferentes.


---


## 📘 Quando o PID Não Basta

| Situação | Por quê | Alternativa |
| ---------| --------| ------------|
| Atraso de transporte grande ($L > T$) | O PID reage tarde demais | Preditor de Smith |
| Perturbação medível e dominante | Corrigir depois do erro é tarde | Controle antecipativo (*feedforward*) |
| Faixas de operação muito distintas | Um só conjunto de ganhos não serve | Escalonamento de ganhos |
| Múltiplas entradas e saídas acopladas | Malhas brigam entre si | Desacoplamento ou controle multivariável |
| Restrições explícitas no atuador | O PID não sabe da restrição | Controle preditivo (MPC) |


---


## 📘 Código Scilab

```scilab
clear; clc; clf();
s = %s;
G = syslin('c', 1, s^2 + 1.2*s + 4);   // planta: wn = 2, zeta = 0,3

Kp = 8; Ki = 10; Kd = 3;
C = syslin('c', Kd*s^2 + Kp*s + Ki, s);

MF = G*C / (1 + G*C);                  // malha fechada
t = 0:0.01:12;
y = csim('step', t, MF);
plot(t, y); xgrid();
```

Comparando as três ações:

```scilab
clear; clc; clf();
s = %s; t = 0:0.01:12;
G = syslin('c', 1, s^2 + 1.2*s + 4);

ganhos = [8 0 0; 8 10 0; 8 10 3];      // P, PI, PID
for i = 1:3
    Kp = ganhos(i,1); Ki = ganhos(i,2); Kd = ganhos(i,3);
    C = syslin('c', Kd*s^2 + Kp*s + Ki, s);
    MF = G*C / (1 + G*C);
    plot(t, csim('step', t, MF));
end
legend(['P';'PI';'PID']); xgrid();
```


---


## 📘 Referências

- ÅSTRÖM, K. J.; HÄGGLUND, T. *Advanced PID Control*. ISA, 2006.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
