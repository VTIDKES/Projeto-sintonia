# -*- coding: utf-8 -*-
"""Projeto Sintonia — Sistemas de Controle (CTRL).

Ponto de entrada do app Streamlit: navegação entre o Modo Diagrama de
Blocos e a Teoria.
"""

import streamlit as st

from modo_diagrama_blocos import modo_canvas
from modo_guia_estudos import modo_guia_estudos
from ui_comum import sidebar_navegacao, breadcrumbs, rodape

TELAS = {
    'canvas': modo_canvas,
    'teoria': modo_guia_estudos,
}


def main():
    st.set_page_config(
        page_title="Modelagem de Sistemas de Controle",
        page_icon="⚙️",
        layout="wide",
    )
    if st.session_state.get('modo_selecionado') not in TELAS:
        st.session_state.modo_selecionado = 'canvas'

    modo = st.session_state.modo_selecionado
    sidebar_navegacao(modo)
    breadcrumbs(modo)
    TELAS[modo]()
    rodape()


if __name__ == "__main__":
    main()
