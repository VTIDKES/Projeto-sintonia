# 🎛️ 9. Espaço de Estados

## 📘 Descrição Interna vs. Externa

| Tipo | Descrição |
| ------| -----------|
| **Descrição externa** | Relação entrada-saída (função de transferência). Pode ser obtida por medições externas. |
| **Descrição interna** | Revela o comportamento completo de todos os sinais internos do sistema. |

Uma descrição externa pode ser obtida a partir de uma interna, mas o **inverso não é sempre válido** — especialmente quando o sistema é **não controlável** ou **não observável**.

> [!warning]
> ⚠️ Sistemas não controláveis ou não observáveis devem ser evitados no projeto.


---


## 📘 Por que usar Espaço de Estados?

A representação em espaço de estados é mais geral e poderosa que a função de transferência:

- Descreve sistemas **lineares e não lineares**
- Descreve sistemas **invariantes e variantes no tempo**
- Descreve sistemas **SISO e MIMO**
- Facilita simulação computacional de sistemas complexos
- Permite análise de controlabilidade e observabilidade
- Um mesmo sistema possui **diversas realizações** em espaço de estados


---


## 📘 Equações de Estado

O modelo em espaço de estados de um sistema contínuo é descrito por:

$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}u(t) \quad \text{(equação de estado)}$$

$$y(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}u(t) \quad \text{(equação de saída)}$$

onde:
- $\mathbf{x}(t)$ = vetor de **variáveis de estado** ($n \times 1$)
- $u(t)$ = entrada ($m \times 1$)
- $y(t)$ = saída ($p \times 1$)
- $\mathbf{A}$ = **matriz do sistema** ($n \times n$)
- $\mathbf{B}$ = **matriz de entrada** ($n \times m$)
- $\mathbf{C}$ = **matriz de saída** ($p \times n$)
- $\mathbf{D}$ = **matriz de transmissão direta** ($p \times m$) — geralmente $\mathbf{0}$


---


## 📘 Relação com a Função de Transferência

A função de transferência pode ser obtida a partir das matrizes de estado:

$$H(s) = \mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$$

A **matriz resolvente** $(s\mathbf{I} - \mathbf{A})^{-1}$ é chamada de **matriz de transição no domínio s**.


---


## 📘 Realizações — Da Função de Transferência ao Espaço de Estados

### ⚡ Forma Direta (Controlador Canônica)

Para $H(s) = \frac{b_{n-1}s^{n-1} + \cdots + b_1 s + b_0}{s^n + a_{n-1}s^{n-1} + \cdots + a_1 s + a_0}$:

$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ -a_0 & -a_1 & -a_2 & \cdots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix}0\\0\\\vdots\\1\end{bmatrix}$$

$$\mathbf{C} = \begin{bmatrix}b_0 & b_1 & \cdots & b_{n-1}\end{bmatrix}, \quad \mathbf{D} = [0]$$

### ⚡ Realização em Cascata

Fatora-se $H(s)$ como produto de termos de 1ª e 2ª ordem:

$$H(s) = H_1(s) \cdot H_2(s) \cdots H_k(s)$$

Cada bloco tem suas próprias matrizes A, B, C, D e são conectados em série. O vetor de estados é a concatenação dos estados de cada bloco.

### ⚡ Realização em Paralelo

Expande-se $H(s)$ em frações parciais:

$$H(s) = H_1(s) + H_2(s) + \cdots + H_k(s)$$

A matriz $\mathbf{A}$ é **diagonal por blocos**, o que facilita a análise.


---


## 📘 Solução no Domínio do Tempo

### ⚡ Via Transformada de Laplace

Aplicando a Laplace à equação de estado:

$$s\mathbf{X}(s) - \mathbf{x}(0) = \mathbf{A}\mathbf{X}(s) + \mathbf{B}U(s)$$

$$(s\mathbf{I} - \mathbf{A})\mathbf{X}(s) = \mathbf{x}(0) + \mathbf{B}U(s)$$

$$\mathbf{X}(s) = (s\mathbf{I}-\mathbf{A})^{-1}\mathbf{x}(0) + (s\mathbf{I}-\mathbf{A})^{-1}\mathbf{B}U(s)$$

A equação de saída fica:

$$Y(s) = \underbrace{\mathbf{C}(s\mathbf{I}-\mathbf{A})^{-1}\mathbf{x}(0)}_{\text{resposta de estado nulo}} + \underbrace{[\mathbf{C}(s\mathbf{I}-\mathbf{A})^{-1}\mathbf{B}+\mathbf{D}]U(s)}_{\text{resposta de entrada nula}}$$

### ⚡ Via Exponencial de Matriz (solução direta)

A solução no domínio do tempo é:

$$\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}(0) + \int_0^t e^{\mathbf{A}(t-\tau)}\mathbf{B}u(\tau)\,d\tau$$

onde $e^{\mathbf{A}t}$ é a **matriz de transição de estados** $\boldsymbol{\Phi}(t) = \mathcal{L}^{-1}\{(s\mathbf{I}-\mathbf{A})^{-1}\}$.


---


## 📘 Controlabilidade

Um sistema é **completamente controlável** se é possível transferir qualquer estado inicial $\mathbf{x}(0)$ para qualquer estado desejado em tempo finito, utilizando uma entrada $u(t)$ adequada.

### ⚡ Matriz de Controlabilidade:

$$\mathcal{C} = \begin{bmatrix}\mathbf{B} & \mathbf{AB} & \mathbf{A}^2\mathbf{B} & \cdots & \mathbf{A}^{n-1}\mathbf{B}\end{bmatrix}$$

O sistema é controlável se e somente se $\text{rank}(\mathcal{C}) = n$.

```scilab
Cc = cont_mat(A, B)   % Scilab: matriz de controlabilidade
rank(Cc)              % deve ser n para sistema controlável
```


---


## 📘 Observabilidade

Um sistema é **completamente observável** se é possível determinar qualquer estado inicial $\mathbf{x}(0)$ a partir das observações da saída $y(t)$ em um intervalo de tempo finito.

### ⚡ Matriz de Observabilidade:

$$\mathcal{O} = \begin{bmatrix}\mathbf{C} \\ \mathbf{CA} \\ \mathbf{CA}^2 \\ \vdots \\ \mathbf{CA}^{n-1}\end{bmatrix}$$

O sistema é observável se e somente se $\text{rank}(\mathcal{O}) = n$.

```scilab
O = obsv_mat(A, C)   % Scilab: matriz de observabilidade
rank(O)              % deve ser n para sistema observável
```


---


## 📘 Formas Canônicas

### ⚡ Forma Canônica do Controlador
Forma especial onde a controlabilidade é garantida. A matriz $\mathbf{A}$ tem forma companheira (como na forma direta).

### ⚡ Forma Canônica do Observador
Forma especial onde a observabilidade é garantida. Transposta da forma canônica do controlador.


---


## 📘 Código Scilab — Conversão FT ↔ Espaço de Estados

```scilab
clear; clc; s=%s;
num=s*(s+2); den=(s+1)*(s^2+2*s+5);
H=syslin('c', num, den);

% Função de transferência → espaço de estados
sistema = tf2ss(H);
A = sistema(2)
B = sistema(3)
C = sistema(4)
D = sistema(5)
```

```scilab
% Espaço de estados → função de transferência
A=[3 1 0; -3 5 -5; 0 1 -1];
B=[2; 7; 8];
C=[1 -2 -9];
D=[0];
sistema = syslin('c', A, B, C, D);
G = ss2tf(sistema)   % converte para FT
```


---


## 📘 Código Scilab — Controlabilidade e Observabilidade

```scilab
clear; clc;
A=[3 1 0 4 -2; -3 5 -5 2 -1; 0 1 -1 2 8; -7 6 -3 -4 0; -6 0 4 -3 1];
B=[2; 7; 8; 5; 4];
C=[1 -2 -9 7 6];
D=[0];

Cc = cont_mat(A, B)     % matriz de controlabilidade
O  = obsv_mat(A, C)     % matriz de observabilidade

disp(rank(Cc))          % 5 = controlável
disp(rank(O))           % 5 = observável
```


---


## 📘 Vantagens do Espaço de Estados

| Aspecto | Função de Transferência | Espaço de Estados |
| ---------| ------------------------| ------------------|
| **Sistemas MIMO** | Difícil | Natural |
| **Condições iniciais** | Limitado | Totalmente incorporado |
| **Projeto de controladores** | Métodos clássicos | Realimentação de estados (LQR, etc.) |
| **Análise interna** | Não disponível | Completa |
| **Simulação** | Menos eficiente | Altamente eficiente |
| **Não-linearidades** | Difícil | Mais acessível |


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
