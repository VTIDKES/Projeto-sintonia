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

- **v1.2 (15/11/2026)** — o app passa a ter apenas a tela inicial, o Modo Diagrama de Blocos e a
  Teoria (Modo Classico e Modo Simulacao com Elementos removidos). O alternador
  Basico/Avancado agora atua no editor de blocos, e o diagrama e mantido ao
  alternar o nivel ou trocar de tela.
- **v1.1 (15/11/2026)** — adequacao a analise RED-Eng: sidebar so de navegacao,
  breadcrumbs, modo Basico/Avancado, cenarios guiados, recalculo automatico no
  Modo Classico, diagnostico interpretativo de estabilidade, cartoes de erro
  didaticos, aba Teoria reativada, editor de blocos acompanhando o tema do app
  e navegavel por teclado, rotulo Beta no Modo Simulacao com Elementos, limpeza
  de arquivo legado, dependencia morta e documentacao duplicada.

EN

Hello, my name is Vitor.
This project was born from the need to present, in a visual, didactic, and accessible way, a learning platform for the Control Systems discipline.

The Sintonia Project aims to use Streamlit together with Google Colab as tools to support learning, making the concepts more interactive and easier to understand. In addition, the project seeks to place the student as the protagonist of their own learning process, aligning with active learning methodologies such as PBL (Problem-Based Learning), PjBL (Project-Based Learning), and Peer Instruction.

Here, we intend to build a solid culture and a fully collaborative ecosystem, where students and enthusiasts can learn, share, and grow together.
