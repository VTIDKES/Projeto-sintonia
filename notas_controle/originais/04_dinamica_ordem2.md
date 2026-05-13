# 🎛️ 4. Dinâmica — Sistemas de 2ª Ordem

## 📘 Função de Transferência Padrão

A forma canônica de um sistema de 2ª ordem é:

$$H(s) = \frac{k\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

onde:
- $k$ = ganho estático
- $\omega_n$ = **frequência natural não amortecida** (rad/s)
- $\zeta$ (zeta ou ξ) = **coeficiente de amortecimento** (adimensional)


---


## 📘 Tipos de Resposta — Posição dos Polos

Os polos do denominador são:

$$s_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

### ⚡ Tabela de comportamento:

| Condição | Tipo dos polos | Comportamento | Nome |
| ----------| ---------------| ---------------| ------|
| $\zeta > 1$ | Reais distintos | Sem oscilação, decaimento lento | **Sobreamortecido** |
| $\zeta = 1$ | Reais iguais | Sem oscilação, decaimento mais rápido | **Criticamente amortecido** |
| $0 < \zeta < 1$ | Complexos conjugados | Oscilação amortecida | **Subamortecido** |
| $\zeta = 0$ | Imaginários puros | Oscilação constante (não para) | **Não amortecido** |
| $\zeta < 0$ | Semiplano direito | Oscilação crescente | **Instável** |


---


## 📘 Diagrama de Polos no Plano Complexo

Para polos complexos $s = -\sigma \pm j\omega_d$:

- **Parte real**: $\sigma = \zeta\omega_n$ → determina a taxa de decaimento
- **Parte imaginária**: $\omega_d = \omega_n\sqrt{1-\zeta^2}$ → **frequência natural amortecida**
- **Módulo**: $|s| = \omega_n$
- **Ângulo**: $\cos\theta = \zeta$

> [!tip]
> 💡 Quanto mais à esquerda os polos, mais rápido é o decaimento. Quanto maior a parte imaginária, mais rápida é a oscilação.


---


## 📘 Especificações de Desempenho (Resposta ao Degrau — Subamortecido)

Para $0 < \zeta < 1$ com entrada degrau unitário:

### ⚡ Tempo de pico $T_p$
Instante em que ocorre o primeiro pico (máximo):

$$T_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n\sqrt{1-\zeta^2}}$$

### ⚡ Ultrapassagem percentual $\%UP$ (overshoot)
Quanto a resposta ultrapassa o valor final:

$$\%UP = 100\, e^{-\pi\zeta/\sqrt{1-\zeta^2}}$$

> Depende **apenas** de $\zeta$. Quanto maior $\zeta$, menor o overshoot.

### ⚡ Tempo de acomodação $T_s$
Tempo para a resposta entrar e permanecer na faixa de ±2% do valor final:

$$T_s \approx \frac{4}{\zeta\omega_n} = \frac{4}{\sigma}$$

### ⚡ Tempo de subida $T_r$
Tempo para a resposta ir de 10% a 90% do valor final (depende de $\zeta$ e $\omega_n$).


---


## 📘 Efeito dos Parâmetros

### ⚡ Variação de $\zeta$ (amortecimento)

| $\zeta$ | Overshoot | Oscilação | Acomodação |
| ---------| -----------| -----------| ------------|
| Pequeno | Grande | Muito oscilatório | Rápida entrada, demora para estabilizar |
| $\approx 0.7$ | ~5% | Moderado | Bom compromisso (projeto típico) |
| $= 1$ | Nenhum | Nenhuma | Crítico |
| Grande | Nenhum | Nenhuma | Lento para chegar ao valor final |

### ⚡ Variação de $\omega_n$ (frequência natural)
- Maior $\omega_n$ → resposta **mais rápida** em todas as métricas
- Não altera a forma da curva (apenas a escala temporal)

### ⚡ Variação do ganho $k$
- Altera o **valor final**: $y(\infty) = k$
- Não altera $T_p$, $T_s$, $\%UP$ (para sistema padrão sem zeros adicionais)


---


## 📘 Polos e Zeros Adicionais

As fórmulas de $T_p$, $T_s$ e $\%UP$ são válidas **somente** para sistemas com exatamente dois polos complexos e nenhum zero.

### ⚡ Polo adicional
- Um polo adicional real $s = -p$ "atrasa" a resposta
- Se $|p|$ for **muito maior** que $|\sigma|$: polo é dominado pelos polos complexos → pode ser ignorado
- Se $|p|$ for **próximo** de $|\sigma|$: o polo afeta significativamente a resposta → não pode ser ignorado

### ⚡ Zero adicional
- Zeros afetam a **amplitude** (resíduo) de cada componente, mas **não a natureza** da resposta
- Zeros próximos dos polos dominantes têm maior efeito
- Zeros no **semiplano direito** causam fase não mínima: a resposta inicia na direção oposta


---


## 📘 Polo na Origem — Sistema de 2ª Ordem com Integrador

Para:

$$H(s) = \frac{k}{s(s + a)} \quad \text{ou} \quad H(s) = \frac{k}{s^2 + k}$$

- Em malha aberta: resposta ao degrau é uma **rampa** ou **oscilação permanente**
- Em malha fechada com realimentação unitária: pode resultar em sistema estável com erro nulo para entrada degrau


---


## 📘 Código Scilab — Variação do Coeficiente de Amortecimento

```scilab
clear; clc; clf();
s=%s; k=1; wn=2; t=0:0.01:10;
for qsi = [0, 0.1, 0.4, 0.7, 1.0, 1.5]
    num=k*(wn^2); den=(s^2+2*qsi*wn*s+wn^2);
    H=syslin('c', num, den);
    y=csim('step',t,H);
    plot(t,y)
end
legend(['ξ=0';'ξ=0.1';'ξ=0.4';'ξ=0.7';'ξ=1';'ξ=1.5'])
```


---


## 📘 Código Scilab — Resposta e Métricas de Desempenho

```scilab
clear; clc;
s=%s;
k=1; qsi=0.7; wn=2; b=0; a=0;
num=k*(wn^2); den=(s^2+2*qsi*wn*s+wn^2);
H=syslin('c', num, den);
y_inf=k;
t=0:0.001:10; y=csim('step',t,H);

% Calcular métricas
Tp = t(find(y==max(y),1));        % tempo de pico
UP = (max(y)-y_inf)/y_inf*100;   % ultrapassagem %

clf();
subplot(121); plzr(H);
subplot(122); plot2d(t,y); xgrid();
mprintf('Tp=%.3f s | %%UP=%.1f%%', Tp, UP);
```


---


## 📘 Exemplos Físicos de Sistemas de 2ª Ordem

### ⚡ Sistema massa-mola-amortecedor (mecânico)

$$H(s) = \frac{1/m}{s^2 + (b/m)s + (k/m)}$$

onde: $m$ = massa, $b$ = amortecimento, $k$ = rigidez da mola.

Aqui: $\omega_n = \sqrt{k/m}$ e $\zeta = b/(2\sqrt{km})$

### ⚡ Circuito RLC (elétrico)

$$H(s) = \frac{1/LC}{s^2 + (R/L)s + 1/LC}$$

onde: $\omega_n = 1/\sqrt{LC}$ e $\zeta = R/(2)\sqrt{C/L}$


---


## 📘 Resumo das Relações Importantes

$$\omega_d = \omega_n\sqrt{1-\zeta^2} \quad (\text{frequência amortecida})$$

$$\sigma = \zeta\omega_n \quad (\text{taxa de decaimento})$$

$$\%UP = 100\,e^{-\pi\zeta/\sqrt{1-\zeta^2}}$$

$$T_p = \frac{\pi}{\omega_d}, \quad T_s \approx \frac{4}{\sigma}$$


---


## 📘 Referências

- LATHI, B. P.; GREEN, R. *Sinais e sistemas lineares*. 3ª ed. Oxford, 2018.
- DORF, R. C.; BISHOP, R. H. *Sistemas de controle modernos*. 13ª ed. LTC, 2017.
- OGATA, K. *Engenharia de controle moderno*. 5ª ed. Pearson, 2014.
- NISE, N. S. *Engenharia de sistemas de controle*. 7ª ed. Wiley, 2018.
