# -*- coding: utf-8 -*-
"""Modo Diagrama de Blocos do app Sintonia."""

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from cenarios import CENARIOS
from modo_guia_estudos import render_guia_janela
from ui_comum import onboarding


def _load_visual_editor_html():
    html_path = Path(__file__).parent / "visual_blocks_frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


def modo_canvas():
    avancado = st.session_state.get('avancado', False)

    st.title("Modo Diagrama de Blocos")
    st.caption(
        "Competência desenvolvida: **síntese e simplificação topológica** de "
        "malhas de controle — você decide a topologia; o editor calcula a "
        "função de transferência equivalente.")
    st.caption(
        "O editor calcula automaticamente a FT equivalente, os polos, os zeros "
        "e as respostas no tempo e na frequência; a topologia, o compensador e "
        "o ganho são decisão do estudante.")
    onboarding(
        "adicione blocos (ou abra **Cenários** para um exemplo pronto), "
        "arraste para posicionar e conecte porta azul redonda (saída) → "
        "porta verde quadrada (entrada).",
        "clique em **CALCULAR**; se você alterar o diagrama depois, o aviso "
        "“recálculo pendente” aparece ao lado do botão.",
        "leia o cartão de **Diagnóstico** (estabilidade, margens e efeito da "
        "realimentação) antes dos gráficos.")
    st.caption(
        "Nível atual: **Avançado** — todos os blocos e análises liberados."
        if avancado else
        "Nível atual: **Básico** — blocos essenciais, resposta no tempo, "
        "desempenho e polos/zeros. Ative **⚙ Modo Avançado** na barra lateral "
        "para PID, PD, integrador, derivador, atuador, espaço de estados, "
        "Bode, Nyquist e LGR. O diagrama é mantido ao alternar.")

    with st.sidebar:
        with st.expander("Como usar", expanded=True):
            st.markdown("""
            **Diagrama de Blocos:**
            1. Clique **+ Adicionar Bloco** ou use os botões rápidos
            2. Arraste para posicionar
            3. Conecte: porta azul redonda (saída) → porta verde quadrada (entrada)
            4. Edite parâmetros no painel lateral
            5. Clique **CALCULAR** para ver resultados

            **Entrada Manual:**
            1. Clique em "Entrada Manual"
            2. Escolha: T(s) Direta, Malha Fechada ou Malha Aberta
               (Espaço de Estados no Modo Avançado)
            3. Preencha os campos
            4. Clique **CALCULAR T(s)**

            **Teclado:** Tab percorre os blocos, setas movem o bloco
            selecionado, Delete remove, Esc cancela a conexão.
            """)

    render_guia_janela("Guia")

    # Escapa "</" para que o JSON não feche a tag script do editor.
    cenarios_json = json.dumps(CENARIOS, ensure_ascii=False).replace("</", "<\\/")
    html_content = _load_visual_editor_html().replace(
        '__AVANCADO__', 'true' if avancado else 'false').replace(
        '__CENARIOS__', cenarios_json)
    components.html(html_content, height=2800, scrolling=True)
