# Guia de Estudos do Sintonia

Este guia resume os principais conceitos usados no app Sintonia. Ele serve como apoio rapido para quem esta montando funcoes de transferencia, diagramas de blocos, respostas no tempo, Bode, Nyquist e espaco de estados.

## Como Usar Este Guia

Use este material como uma ponte entre a teoria e o app:

- Se voce esta comecando, leia `Sinais e Sistemas`, depois `Laplace` e `Funcao de Transferencia`.
- Se voce esta simulando uma planta simples, consulte `Sistemas de 1 Ordem` e `Sistemas de 2 Ordem`.
- Se esta fechando uma malha, leia `Realimentacao`, `Erro Estacionario` e `Estabilidade`.
- Se esta analisando frequencia, use `Bode`, `Margens` e `Nyquist`.
- Se esta usando matrizes A, B, C e D, va direto para `Espaco de Estados`.

## Sinais e Sistemas

Um sinal e uma grandeza que carrega informacao. Em controle, normalmente a entrada e chamada de `u(t)` ou `r(t)`, e a saida de `y(t)`.

Um sistema transforma entrada em saida. No app, essa transformacao aparece como:

$$G(s) = \frac{Y(s)}{U(s)}$$

Tipos importantes:

| Conceito | Ideia principal | Exemplo no app |
| --- | --- | --- |
| Entrada | O que excita o sistema | Degrau, rampa, senoide |
| Saida | O que se observa | Resposta no tempo |
| Sistema SISO | Uma entrada e uma saida | Planta simples G(s) |
| Sistema MIMO | Multiplas entradas/saidas | Mais comum em espaco de estados |

Sinais mais usados em testes:

- Degrau: testa rapidez, sobressinal e acomodacao.
- Rampa: testa erro estacionario para entradas que crescem com o tempo.
- Impulso: revela a resposta natural do sistema.
- Senoidal: base da analise em frequencia.

## Transformada de Laplace

A Transformada de Laplace troca equacoes diferenciais por algebra em `s`. Isso facilita modelar sistemas dinamicos.

$$\mathcal{L}\{f(t)\}=F(s)$$

Transformadas uteis:

| Tempo | Laplace |
| --- | --- |
| Impulso `delta(t)` | `1` |
| Degrau `u(t)` | `1/s` |
| Rampa `t` | `1/s^2` |
| Exponencial `e^{-at}` | `1/(s+a)` |
| Senoide `sin(wt)` | `w/(s^2+w^2)` |

Regra pratica: no dominio de Laplace, derivar vira multiplicar por `s` e integrar vira dividir por `s`.

## Funcao de Transferencia

A funcao de transferencia representa a relacao entre saida e entrada com condicoes iniciais nulas:

$$G(s)=\frac{Y(s)}{U(s)}=\frac{b_m s^m+\cdots+b_1s+b_0}{a_n s^n+\cdots+a_1s+a_0}$$

No app, voce pode escrever, por exemplo:

- Numerador: `4`
- Denominador: `s^2+2s+4`

Interpretacao:

- Zeros: raizes do numerador.
- Polos: raizes do denominador.
- Ordem do sistema: maior potencia do denominador.
- Ganho DC: valor aproximado da saida final para degrau unitario, quando o sistema e estavel.

## Diagramas de Blocos

Diagramas de blocos ajudam a montar sistemas complexos por partes.

Operacoes principais:

| Conexao | Resultado |
| --- | --- |
| Serie | `G_total = G1 * G2` |
| Paralelo | `G_total = G1 + G2` |
| Realimentacao negativa | `T = G/(1 + G*H)` |
| Realimentacao positiva | `T = G/(1 - G*H)` |

No modo visual do app:

- A porta azul e saida.
- A porta verde e entrada.
- O somador define sinais `+` e `-`.
- O bloco `H(s)` normalmente representa sensor ou realimentacao.

## Sistemas de 1 Ordem

Forma comum:

$$G(s)=\frac{K}{\tau s+1}$$

Onde:

- `K` e o ganho.
- `tau` e a constante de tempo.

Leitura rapida:

| Medida | Aproximacao |
| --- | --- |
| 63,2% do valor final | `t = tau` |
| Tempo de subida | `Tr aproximadamente 2,2 tau` |
| Tempo de acomodacao 2% | `Ts aproximadamente 4 tau` |

Se `tau` diminui, a resposta fica mais rapida. Se `K` aumenta, o valor final aumenta.

## Sistemas de 2 Ordem

Forma padrao:

$$G(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}$$

Onde:

- `omega_n` e a frequencia natural.
- `zeta` e o fator de amortecimento.

Comportamento por amortecimento:

| zeta | Comportamento |
| --- | --- |
| `0 < zeta < 1` | Subamortecido, tem oscilacao e sobressinal |
| `zeta = 1` | Criticamente amortecido |
| `zeta > 1` | Sobreamortecido, sem oscilacao e mais lento |
| `zeta = 0` | Oscilatorio |

Metricas importantes:

$$M_p = e^{-\frac{\zeta\pi}{\sqrt{1-\zeta^2}}}\cdot100\%$$

$$T_s \approx \frac{4}{\zeta\omega_n}$$

## Realimentacao e Erro Estacionario

Na realimentacao negativa, a saida medida volta para comparar com a referencia. O erro e:

$$E(s)=R(s)-Y(s)$$

Funcao de transferencia de malha fechada:

$$T(s)=\frac{G(s)}{1+G(s)H(s)}$$

Constantes de erro:

| Constante | Formula | Entrada associada |
| --- | --- | --- |
| `Kp` | `lim s->0 G(s)` | Degrau |
| `Kv` | `lim s->0 sG(s)` | Rampa |
| `Ka` | `lim s->0 s^2G(s)` | Parabola |

Tipos de sistema:

- Tipo 0: nenhum polo na origem.
- Tipo 1: um polo na origem.
- Tipo 2: dois polos na origem.

Quanto maior o tipo, melhor tende a ser o rastreamento de entradas mais exigentes, mas a estabilidade pode ficar mais delicada.

## Estabilidade

Um sistema continuo e estavel quando todos os polos relevantes estao no semiplano esquerdo:

$$Re(p_i)<0$$

Leitura pelo plano complexo:

| Posicao dos polos | Interpretacao |
| --- | --- |
| Esquerda | Estavel |
| Direita | Instavel |
| Eixo imaginario simples | Marginal |
| Origem | Integrador |

Regra pratica no app:

- Polos muito perto do eixo imaginario indicam resposta lenta ou oscilatoria.
- Polos no semiplano direito indicam instabilidade.
- Aumentar ganho pode melhorar rapidez, mas tambem pode reduzir margem de estabilidade.

## Bode e Resposta em Frequencia

No Bode, avaliamos:

$$G(j\omega)$$

O grafico de magnitude mostra ganho por frequencia. O grafico de fase mostra atraso ou avanco.

Contribuicoes tipicas:

| Elemento | Magnitude | Fase |
| --- | --- | --- |
| Ganho `K` | Sobe/desce tudo | Nao muda se `K > 0` |
| Polo real | -20 dB/dec apos canto | Ate -90 graus |
| Zero real | +20 dB/dec apos canto | Ate +90 graus |
| Integrador `1/s` | -20 dB/dec sempre | -90 graus |

Margens:

- Margem de ganho: quanto o ganho pode aumentar antes da instabilidade.
- Margem de fase: quanto atraso de fase ainda cabe antes da instabilidade.

Como regra pratica, margem de fase maior costuma indicar resposta mais robusta e menos oscilatoria.

## Nyquist

Nyquist analisa a estabilidade de malha fechada usando a curva de `G(jw)H(jw)` no plano complexo.

Ponto critico:

$$-1 + j0$$

Forma simplificada:

$$Z = N + P$$

Onde:

- `P` e o numero de polos de malha aberta no semiplano direito.
- `N` e o numero de envolvimentos do ponto `-1`.
- `Z` e o numero de polos instaveis da malha fechada.

Para uma planta de malha aberta estavel (`P=0`), a curva nao deve envolver o ponto `-1`.

## Lugar Geometrico das Raizes

O LGR mostra como os polos de malha fechada se movem quando o ganho `K` varia.

Equacao caracteristica:

$$1 + KG(s)H(s)=0$$

Use o LGR para responder:

- Que ganho deixa o sistema mais rapido?
- Em que ganho o sistema fica instavel?
- Onde ficam os polos dominantes?
- O sistema fica mais oscilatorio quando K aumenta?

## Espaco de Estados

Representacao interna:

$$\dot{x}=Ax+Bu$$

$$y=Cx+Du$$

Significado:

| Matriz | Papel |
| --- | --- |
| `A` | Dinamica interna |
| `B` | Como a entrada atua nos estados |
| `C` | Como os estados viram saida |
| `D` | Caminho direto entrada-saida |

Relacao com funcao de transferencia:

$$G(s)=C(sI-A)^{-1}B+D$$

Quando usar:

- Sistemas com varios estados fisicos.
- Modelos MIMO.
- Controle moderno.
- Casos em que a funcao de transferencia fica pouco intuitiva.

## Checklist de Analise no App

1. Escreva a planta `G(s)` ou as matrizes `A, B, C, D`.
2. Veja polos e zeros.
3. Confira se os polos indicam estabilidade.
4. Rode resposta ao degrau para avaliar rapidez e sobressinal.
5. Se houver realimentacao, confira erro estacionario.
6. Use Bode para ver margens de estabilidade.
7. Use Nyquist quando quiser confirmar robustez em frequencia.
8. Ajuste ganho `K` com cuidado: rapidez demais pode trazer oscilacao.

## Exemplos Rapidos

### Planta de 1 ordem

Numerador:

```text
1
```

Denominador:

```text
s+1
```

O valor final para degrau unitario tende a 1, e a resposta nao deve oscilar.

### Planta de 2 ordem subamortecida

Numerador:

```text
4
```

Denominador:

```text
s^2+1.2s+4
```

Espere uma resposta com sobressinal, porque os polos tendem a ser complexos conjugados.

### Realimentacao negativa simples

Use:

```text
G(s) = 10/(s^2+3s+1)
H(s) = 1
```

A malha fechada fica:

$$T(s)=\frac{10}{s^2+3s+11}$$

## Referencias

- LATHI, B. P.; GREEN, R. Sinais e sistemas lineares.
- DORF, R. C.; BISHOP, R. H. Sistemas de controle modernos.
- OGATA, K. Engenharia de controle moderno.
- NISE, N. S. Engenharia de sistemas de controle.
