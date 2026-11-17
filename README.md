# Projeto-sintonia
PT-BR
Olá, meu nome é Vitor.
Este projeto nasceu da necessidade de apresentar, de forma visual, didática e acessível, uma plataforma de ensino para a disciplina de Sistemas de Controle.

O Projeto Sintonia tem como objetivo utilizar o Streamlit em conjunto com o Google Colab como ferramentas de apoio ao aprendizado, tornando os conteúdos mais interativos e compreensíveis. Além disso, busca colocar o aluno como protagonista do processo de aprendizagem, alinhando-se às metodologias ativas, como PBL (Problem-Based Learning), PjBL (Project-Based Learning) e Peer Instruction.

Aqui, queremos criar uma cultura sólida e um ecossistema totalmente colaborativo, no qual estudantes e interessados possam aprender, compartilhar e evoluir juntos.

## Objetivo de aprendizagem e público-alvo

- **Público-alvo:** estudantes de graduação em Engenharia de Controle e Automação e Engenharia Elétrica.
- **Disciplina de referência:** Sistemas de Controle / Controle Clássico.
- **Objetivo de aprendizagem:** ao usar o Sintonia, o estudante deve ser capaz de
  modelar sistemas LTI SISO, **analisar** a resposta no tempo e na frequência e
  **avaliar** a estabilidade (polos, margens de ganho e de fase, lugar geométrico
  das raízes), justificando escolhas de topologia, compensador e ganho.

| Tela | Competência desenvolvida |
|------|--------------------------|
| Modo Diagrama de Blocos | Síntese e simplificação topológica de malhas de controle; análise da resposta no tempo e na frequência |
| Teoria | Fundamentação: estabilidade, Bode, Nyquist, LGR, PID e compensadores |

O editor calcula polos, zeros e respostas automaticamente; a decisão sobre
topologia, compensador e ganho permanece com o estudante.

## Interface (diretrizes RED-Eng)

- **Barra lateral:** apenas navegação global e alternador **Básico / Avançado**.
- **Básico / Avançado:** o Básico mostra os blocos essenciais (entrada, saída,
  planta, ganho, sensor, somador, ramificação) e as análises de resposta no
  tempo, desempenho e polos/zeros. O Avançado libera PID, PD, integrador,
  derivador, atuador, espaço de estados, Bode, Nyquist e LGR. O diagrama é
  mantido ao alternar o nível ou trocar de tela.
- **Breadcrumbs** (`Início → Sistemas de Controle (CTRL) → Tela`) no topo de cada tela.
- **Como utilizar:** roteiro Configurar → Executar → Interpretar.
- **Cenários:** o botão **Cenários** da barra do editor abre uma biblioteca de
  casos prontos (1ª ordem, 2ª ordem subamortecida, criticamente amortecida e
  superamortecida, atraso com aproximação de Padé, malha fechada com ganho K,
  malha fechada com PID e um caso instável). O PID só aparece no Modo
  Avançado. Carregar um cenário substitui o diagrama (dá para desfazer) e
  mostra a descrição e o que observar. Os cenários ficam em `cenarios.py`.
- **Desfazer / Refazer / Resetar:** botões e atalhos Ctrl+Z, Ctrl+Y e
  Ctrl+Shift+Z no editor; **Resetar** na Entrada Manual restaura os campos iniciais.
- **Função de transferência como fonte única:** o par `{n, d}` da FT equivalente
  é a única fonte de verdade de todos os gráficos e métricas (resposta no tempo,
  desempenho, polos e zeros, Bode, Nyquist, LGR e diagnóstico). Novas análises
  devem derivar dele.
- **Diagnóstico interpretativo:** cartão Estável / Marginal / Instável com margens
  e orientação pedagógica; erros no formato Situação → Causa provável → Ação recomendada.

## Guia de estudos

O projeto tambem inclui um guia de consulta em `notas_controle/guia_sintonia.md`.
Esse material resume os principais topicos de Sistemas de Controle usados no app:
sinais e sistemas, Transformada de Laplace, funcoes de transferencia,
diagramas de blocos, realimentacao, estabilidade, Bode, Nyquist, LGR e
espaco de estados, sistemas eletricos e sistemas mecanicos.

No Streamlit, o guia aparece como uma janela movel **Guia**, para que o
usuario selecione um assunto, arraste a janela pela barra superior e consulte
a teoria enquanto monta ou analisa um sistema. O mesmo conteudo fica disponivel
em tela cheia na aba **Teoria** da navegacao.

As notas originais usadas como base tambem ficam em `notas_controle/originais/`.

## Changelog

- **v1.3 (16/11/2026)** — lacunas restantes da análise RED-Eng: `config.toml`
  movido para `.streamlit/config.toml` e corrigido (TOML válido); título da
  página; paleta da tela inicial e da janela Guia acompanhando o tema claro ou
  escuro; ação primária única em azul (`#1f77b4`); cálculo robusto com erros
  didáticos: na Entrada Manual (denominador vazio ou nulo, polinômio inválido,
  NaN em matrizes) e também nos blocos do diagrama (numerador e denominador de
  Planta, Sensor e Atuador, ganhos e matrizes; o bloco com problema é marcado) e
  entrada de somador sem ligação; correção do cálculo de espaço de estados com
  n >= 3; selo Estável / Marginal / Instável único e sempre visível no
  cabeçalho de resultados (polo repetido sobre o eixo imaginário conta como
  Instável); MG/MF só são suprimidas quando há laço de realimentação no
  diagrama (detecção por ciclo no grafo) ou na Entrada Manual em Malha Fechada,
  e um diagrama em paralelo continua sendo malha aberta; unidades nos eixos;
  Desfazer e Refazer; Resetar na Entrada Manual; biblioteca de Cenários
  guiados; indicador "Calculando... etapa X de 3"; viewport sem bloqueio de
  zoom, portas de entrada quadradas e de saída redondas; realce do último bloco
  alterado também ao adicionar bloco e criar conexão; removido o selo de depuração.
- **v1.2 (15/11/2026)** — o app passa a ter apenas a tela inicial, o Modo Diagrama de Blocos e a
  Teoria (Modo Classico e Modo Simulacao com Elementos removidos). O alternador
  Basico/Avancado agora atua no editor de blocos, e o diagrama e mantido ao
  alternar o nivel ou trocar de tela.
- **v1.1 (15/11/2026)** — adequacao a analise RED-Eng: sidebar so de navegacao,
  breadcrumbs, modo Basico/Avancado, diagnostico interpretativo de
  estabilidade, cartoes de erro didaticos, aba Teoria reativada, editor de
  blocos acompanhando o tema do app e navegavel por teclado, limpeza de arquivo
  legado, dependencia morta e documentacao duplicada. (Os cenarios do Modo
  Classico, o recalculo automatico e o rotulo Beta foram descartados junto com
  o Modo Classico e o Modo Simulacao com Elementos, removidos na v1.2; os
  cenarios voltaram na v1.3 dentro do editor de blocos.)

EN

Hello, my name is Vitor.
This project was born from the need to present, in a visual, didactic, and accessible way, a learning platform for the Control Systems discipline.

The Sintonia Project aims to use Streamlit together with Google Colab as tools to support learning, making the concepts more interactive and easier to understand. In addition, the project seeks to place the student as the protagonist of their own learning process, aligning with active learning methodologies such as PBL (Problem-Based Learning), PjBL (Project-Based Learning), and Peer Instruction.

Here, we intend to build a solid culture and a fully collaborative ecosystem, where students and enthusiasts can learn, share, and grow together.
