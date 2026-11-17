# -*- coding: utf-8 -*-
"""Projeto Sintonia — Sistemas de Controle (CTRL).

Ponto de entrada do app Streamlit: tela inicial e navegação entre o Modo
Diagrama de Blocos e a Teoria.
"""

import streamlit as st

from modo_diagrama_blocos import modo_canvas
from modo_guia_estudos import modo_guia_estudos, render_guia_janela
from ui_comum import sidebar_navegacao, breadcrumbs, rodape


def tela_inicial():
    st.markdown("""
    <style>
    .mode-card {
        background: rgba(128,128,128,.10);
        color: inherit;
        border: 2px solid rgba(128,128,128,.35);
        border-radius: 16px;
        overflow: hidden;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        min-height: 350px;
        user-select: none;
    }
    .mode-card-body {
        padding: 28px 24px 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
    }
    .mode-card-diagram  { border-top: 4px solid #1f77b4; }
    .mode-card-diagram:hover {
        border-color: #1f77b4;
        transform: translateY(-3px);
        box-shadow: 0 14px 40px rgba(31,119,180,0.25);
        background: rgba(31,119,180,.10);
    }
    .mode-icon {
        width: 100%;
        max-width: 230px;
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .mode-title { font-size: 20px; font-weight: 700; margin-bottom: 10px; color: inherit; }
    .mode-desc  { font-size: 14px; color: inherit; opacity: .75; line-height: 1.7; flex: 1; }
    .welcome-title {
        text-align: center; font-size: 32px; font-weight: 700;
        margin-bottom: 8px;
    }
    .welcome-sub {
        text-align: center; font-size: 16px; opacity: .7;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="welcome-title">Sistema de Modelagem e Análise de Controle</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Clique no card para começar</div>',
                unsafe_allow_html=True)

    render_guia_janela("Guia de Estudos")

    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("""
        <div class="mode-card mode-card-diagram">
            <div class="mode-card-body">
                <div class="mode-icon">
                    <svg viewBox="0 0 240 80" xmlns="http://www.w3.org/2000/svg" width="100%">
                      <defs>
                        <marker id="arw-diag" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                          <polygon points="0 0, 8 3, 0 6" fill="#1f77b4"/>
                        </marker>
                      </defs>
                      <text x="3" y="45" fill="currentColor" font-size="12" font-family="monospace" font-weight="bold">R(s)</text>
                      <line x1="35" y1="40" x2="58" y2="40" stroke="#1f77b4" stroke-width="2" marker-end="url(#arw-diag)"/>
                      <rect x="60" y="18" width="120" height="44" fill="none" stroke="#1f77b4" stroke-width="1.5" rx="5"/>
                      <text x="120" y="35" fill="currentColor" font-size="13" font-family="monospace" text-anchor="middle">4</text>
                      <line x1="73" y1="40" x2="167" y2="40" stroke="currentColor" stroke-width="1" opacity="0.35"/>
                      <text x="120" y="55" fill="currentColor" font-size="13" font-family="monospace" text-anchor="middle">s + 4</text>
                      <line x1="180" y1="40" x2="203" y2="40" stroke="#1f77b4" stroke-width="2" marker-end="url(#arw-diag)"/>
                      <text x="206" y="45" fill="currentColor" font-size="12" font-family="monospace" font-weight="bold">Y(s)</text>
                    </svg>
                </div>
                <div class="mode-title">Modo Diagrama de Blocos</div>
                <div class="mode-desc">
                    Editor visual de diagrama de blocos. Arraste, conecte
                    e calcule a FT equivalente graficamente.
                    Entrada manual e espaço de estados integrados.
                    <br><b>Competência:</b> síntese e simplificação de malhas.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Entrar no Modo Diagrama de Blocos", key="btn_canvas", type="primary", use_container_width=True):
            st.session_state.modo_selecionado = 'canvas'
            st.rerun()


TELAS = {
    None: tela_inicial,
    'canvas': modo_canvas,
    'teoria': modo_guia_estudos,
}


def main():
    st.set_page_config(
        page_title="Sintonia · Sistemas de Controle (CTRL)",
        page_icon="⚙️",
        layout="wide",
    )
    modo = st.session_state.get('modo_selecionado')
    sidebar_navegacao(modo)
    breadcrumbs(modo)
    TELAS[modo]()
    rodape()


if __name__ == "__main__":
    main()
