# -*- coding: utf-8 -*-
"""Modo de consulta didatica para teoria de sistemas de controle."""

from pathlib import Path

import streamlit as st


GUIDE_PATH = Path(__file__).parent / "notas_controle" / "guia_sintonia.md"


TOPICS = {
    "Visao geral": "Como Usar Este Guia",
    "Sinais e sistemas": "Sinais e Sistemas",
    "Laplace": "Transformada de Laplace",
    "Funcao de transferencia": "Funcao de Transferencia",
    "Diagrama de blocos": "Diagramas de Blocos",
    "1 ordem": "Sistemas de 1 Ordem",
    "2 ordem": "Sistemas de 2 Ordem",
    "Realimentacao": "Realimentacao e Erro Estacionario",
    "Estabilidade": "Estabilidade",
    "Bode": "Bode e Resposta em Frequencia",
    "Nyquist": "Nyquist",
    "LGR": "Lugar Geometrico das Raizes",
    "Espaco de estados": "Espaco de Estados",
    "Checklist": "Checklist de Analise no App",
    "Exemplos": "Exemplos Rapidos",
}


def _load_guide_markdown():
    return GUIDE_PATH.read_text(encoding="utf-8")


def _extract_section(markdown, heading):
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start < 0:
        return markdown

    next_heading = markdown.find("\n## ", start + len(marker))
    if next_heading < 0:
        return markdown[start:].strip()
    return markdown[start:next_heading].strip()


def _render_topic_cards():
    st.markdown(
        """
        <style>
        .guide-hero {
            background: linear-gradient(135deg, #12172a, #1b243a);
            border: 1px solid #303858;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 18px;
        }
        .guide-hero h1 {
            color: #e0e4f0;
            font-size: 30px;
            margin: 0 0 8px;
        }
        .guide-hero p {
            color: #a8b0d0;
            margin: 0;
            line-height: 1.6;
        }
        .guide-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 22px;
        }
        .guide-chip {
            border: 1px solid #333654;
            background: #1a1d2e;
            border-radius: 999px;
            color: #cbd5e1;
            font-size: 12px;
            padding: 6px 10px;
        }
        </style>
        <div class="guide-hero">
            <h1>Guia de Estudos</h1>
            <p>Notas de consulta para usar junto com o Sintonia: conceitos,
            formulas, interpretacao dos graficos e exemplos prontos para testar.</p>
        </div>
        <div class="guide-chip-row">
            <span class="guide-chip">Laplace</span>
            <span class="guide-chip">Polos e zeros</span>
            <span class="guide-chip">Bode</span>
            <span class="guide-chip">Nyquist</span>
            <span class="guide-chip">Realimentacao</span>
            <span class="guide-chip">Espaco de estados</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guia_popup(label="Consultar Guia de Estudos"):
    """Render a compact guide popover with topic selection."""
    with st.popover(label, use_container_width=True):
        markdown = _load_guide_markdown()
        topic = st.selectbox(
            "Assunto",
            list(TOPICS.keys()),
            key=f"guia_popup_topic_{label}",
        )
        st.markdown(_extract_section(markdown, TOPICS[topic]))

        with st.expander("Guia completo"):
            st.markdown(markdown)


def modo_guia_estudos():
    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Topicos")
        topic = st.radio(
            "Escolha uma nota",
            list(TOPICS.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("Use o guia como apoio enquanto monta ou interpreta um sistema no app.")

    _render_topic_cards()

    markdown = _load_guide_markdown()
    section = _extract_section(markdown, TOPICS[topic])
    st.markdown(section)

    with st.expander("Ver guia completo"):
        st.markdown(markdown)
