# -*- coding: utf-8 -*-
"""Componentes de interface compartilhados pelas telas do Sintonia (CTRL).

Seguem as diretrizes de identidade unificada do ecossistema SINTONIA
(análise RED-Eng): sidebar só de navegação, breadcrumbs, onboarding em
3 passos e rodapé de versão.
"""

import streamlit as st

APP_VERSAO = "v1.3 — última atualização: 16/11/2026"

# (chave em st.session_state.modo_selecionado, rótulo)
ROTAS = [
    (None, "🏠 Início"),
    ("canvas", "Modo Diagrama de Blocos"),
    ("teoria", "📖 Teoria"),
]


def sidebar_navegacao(modo):
    """Sidebar com papel único: navegação global + alternador Básico/Avançado."""
    with st.sidebar:
        st.header("Navegação")
        for chave, rotulo in ROTAS:
            if st.button(rotulo, key=f"nav_{chave}", use_container_width=True,
                         type="primary" if chave == modo else "secondary"):
                st.session_state.modo_selecionado = chave
                st.rerun()
        st.toggle(
            "⚙ Modo Avançado", key="avancado",
            help="Básico: entrada, saída, planta G(s), ganho K, sensor H(s), "
                 "somador e ramificação; resposta no tempo, desempenho e "
                 "polos/zeros. Avançado: libera PID, PD, integrador, "
                 "derivador, atuador, espaço de estados, Bode, Nyquist e LGR.")
        st.markdown("---")


def breadcrumbs(modo):
    caminho = ["Início", "Sistemas de Controle (CTRL)"]
    if modo is not None:
        caminho.append(dict(ROTAS)[modo].replace("📖 ", ""))
    caminho[-1] = f"**{caminho[-1]}**"
    st.caption(" → ".join(caminho))


def rodape():
    st.markdown("---")
    st.caption(f"Projeto Sintonia · Sistemas de Controle (CTRL) · {APP_VERSAO}")


def onboarding(configurar, executar, interpretar):
    """Contrato fixo de 3 passos: Configurar → Executar → Interpretar."""
    with st.expander("📖 Como utilizar"):
        st.markdown(
            f"1. **Configurar** — {configurar}\n"
            f"2. **Executar** — {executar}\n"
            f"3. **Interpretar** — {interpretar}")
