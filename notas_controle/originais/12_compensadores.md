# 🎛️ 12. Compensadores de Avanço e Atraso

## 📘 Por que Compensar

Ajustar apenas o ganho move os polos de malha fechada **ao longo do lugar das raízes existente**. Se o desempenho desejado não está sobre essa curva, nenhum valor de $k$ resolve.

O compensador acrescenta polos e zeros à malha aberta e, com isso, **redesenha o lugar das raízes** para que a especificação passe a ser alcançável.

| Problema | Compensador | O que ele faz |
| ---------| ------------| --------------|
| Resposta lenta ou pouco amortecida | **Avanço** | Adianta a fase, puxa os ramos para a esquerda |
| Erro em regime alto | **Atraso** | Aumenta o ganho em baixa frequência |
| Os dois ao mesmo tempo | **Avanço-atraso** | Combina os dois efeitos |


---


## 📘 Compensador de Avanço de Fase

$$G_c(s) = K_c\,\frac{s + z_c}{s + p_c}, \qquad |z_c| < |p_c|$$

O zero fica **mais perto da origem** que o polo. Na forma normalizada:

$$G_c(s) = K_c\,\alpha\,\frac{Ts + 1}{\alpha Ts + 1}, \qquad \alpha = \frac{z_c}{p_c} < 1$$

### ⚡ Fase máxima

O avanço de fase é máximo na média geométrica entre o zero e o polo:

$$\omega_m = \sqrt{z_c\,p_c} = \frac{1}{T\sqrt{\alpha}}$$

$$\phi_m = \arcsin\!\left(\frac{1-\alpha}{1+\alpha}\right) \qquad \Longleftrightarrow \qquad \alpha = \frac{1 - \sin\phi_m}{1 + \sin\phi_m}$$

Nessa frequência o ganho vale $1/\sqrt{\alpha}$, ou seja, $-10\log_{10}\alpha$ em dB.

| $\alpha$ | Avanço máximo $\phi_m$ | Ganho em $\omega_m$ |
| ---------| -----------------------| --------------------|
| $0{,}5$ | $19{,}5°$ | $3{,}0$ dB |
| $0{,}2$ | $41{,}8°$ | $7{,}0$ dB |
| $0{,}1$ | $54{,}9°$ | $10{,}0$ dB |
| $0{,}05$ | $64{,}8°$ | $13{,}0$ dB |

> [!warning]
> ⚠️ Não tente arrancar mais de 60° de um único estágio. Abaixo de $\alpha = 0{,}05$ o ganho em alta frequência fica alto demais e o sistema passa a amplificar ruído. Precisando de mais, use dois estágios em cascata.


---


## 📘 Projeto de Avanço pelo Diagrama de Bode

O objetivo é atingir uma margem de fase alvo, tipicamente entre 45° e 60°.

1. Ajuste o ganho $K_c$ para atender ao erro em regime exigido.
2. Levante o Bode do sistema já com esse ganho e meça a margem de fase atual.
3. Calcule o avanço necessário, somando uma folga de 5° a 12° porque o compensador empurra a frequência de cruzamento para a direita:

$$\phi_m = \text{MF}_{\text{desejada}} - \text{MF}_{\text{atual}} + \text{folga}$$

4. Obtenha $\alpha$ a partir de $\phi_m$.
5. Encontre a nova frequência de cruzamento: é onde a magnitude do sistema não compensado vale $10\log_{10}\alpha$ em dB (valor negativo). Faça $\omega_m$ igual a ela.
6. Calcule $T = 1/(\omega_m\sqrt{\alpha})$, de onde saem $z_c = 1/T$ e $p_c = 1/(\alpha T)$.
7. Verifique o resultado no Bode. Se a margem ficou aquém, aumente a folga do passo 3 e repita.


---


## 📘 Projeto de Avanço pelo Lugar das Raízes

Aqui se trabalha com a **condição de ângulo**.

1. Traduza a especificação em um par de polos dominantes desejados $s_d = -\zeta\omega_n \pm j\omega_n\sqrt{1-\zeta^2}$.
2. Calcule quanto de ângulo está faltando para que $s_d$ pertença ao lugar:

$$\phi_{\text{falta}} = 180° - \angle G(s_d)H(s_d)$$

3. Posicione o zero e o polo do compensador de modo que a diferença entre os ângulos que eles enxergam de $s_d$ seja exatamente $\phi_{\text{falta}}$.
4. Aplique a condição de módulo em $s_d$ para achar $K_c$.

Uma escolha prática comum é colocar o zero do compensador **sobre a projeção de $s_d$ no eixo real**, ou cancelando um polo lento da planta. O polo vai então para a esquerda, na posição que fecha o ângulo que falta.


---


## 📘 Compensador de Atraso de Fase

$$G_c(s) = K_c\,\frac{s + z_c}{s + p_c}, \qquad |z_c| > |p_c|$$

Agora o **polo** é que está mais perto da origem. Na forma normalizada:

$$G_c(s) = K_c\,\beta\,\frac{Ts+1}{\beta Ts + 1}, \qquad \beta = \frac{z_c}{p_c} > 1$$

O propósito não é a fase, e sim o ganho: o atraso multiplica por $\beta$ o ganho em baixa frequência, reduzindo o erro em regime pelo mesmo fator, e deixa a região da frequência de cruzamento praticamente intocada.

### ⚡ Regra de posicionamento

Coloque zero e polo bem abaixo da frequência de cruzamento, uma década antes:

$$z_c = \frac{1}{T} \approx \frac{\omega_{gc}}{10}$$

Assim, no entorno de $\omega_{gc}$, o atraso de fase introduzido é pequeno, tipicamente menos de 6°, e a margem de fase quase não sofre.

> [!tip]
> 💡 O compensador de atraso é o primo analógico da ação integral. A diferença é que o polo fica **perto** da origem em vez de exatamente sobre ela, o que reduz o erro sem zerá-lo e sem o risco de windup.


---


## 📘 Compensador de Avanço-Atraso

$$G_c(s) = K_c\,\underbrace{\frac{s + z_1}{s + p_1}}_{\text{avanço}}\;\underbrace{\frac{s + z_2}{s + p_2}}_{\text{atraso}}$$

Projete em duas etapas independentes, sempre nessa ordem:

1. **Primeiro o avanço**, para acertar a resposta transitória e a margem de fase.
2. **Depois o atraso**, para corrigir o erro em regime sem estragar o que já foi ajustado, posicionando-o uma década abaixo da nova frequência de cruzamento.


---


## 📘 Comparação com o PID

| Compensador | Equivalente aproximado | Diferença principal |
| ------------| -----------------------| --------------------|
| Avanço | PD | O avanço tem polo real, o PD tem ganho infinito em alta frequência |
| Atraso | PI | O PI tem polo exatamente na origem, o atraso apenas perto dela |
| Avanço-atraso | PID | Mesma ideia, realização diferente |

Na prática, o PD puro nunca é implementado sozinho: o polo do compensador de avanço é exatamente o filtro que torna a ação derivativa realizável.


---


## 📘 Resumo dos Efeitos

| Aspecto | Avanço | Atraso |
| --------| -------| -------|
| Margem de fase | Aumenta | Diminui pouco |
| Frequência de cruzamento | Aumenta | Diminui um pouco |
| Banda passante | Aumenta | Diminui |
| Velocidade de resposta | Mais rápida | Levemente mais lenta |
| Erro em regime | Melhora pouco | **Melhora muito** |
| Sensibilidade a ruído | **Piora** | Melhora |


---


## 📘 Código Scilab

```scilab
clear; clc; clf();
s = %s;
G = syslin('c', 1, s*(s+2));           // planta com integrador

// compensador de avanco: alpha = 0,1 e wm = 4 rad/s
alfa = 0.1; wm = 4;
T  = 1/(wm*sqrt(alfa));
zc = 1/T; pc = 1/(alfa*T);
Gc = syslin('c', (s+zc), (s+pc));

bode([G; G*Gc], 0.1, 100);
legend(['sem compensacao'; 'com avanco']);
```

Verificando as margens:

```scilab
[gm, fr] = g_margin(G*Gc);   // margem de ganho e frequencia
[pm, fp] = p_margin(G*Gc);   // margem de fase e frequencia
mprintf('MG = %.1f dB  |  MF = %.1f graus\n', gm, pm);
```


---


## 📘 Referências

- FRANKLIN, G. F.; POWELL, J. D.; EMAMI-NAEINI, A. *Feedback Control of Dynamic Systems*. 8ª ed. Pearson, 2019.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
