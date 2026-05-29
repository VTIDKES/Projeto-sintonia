# -*- coding: utf-8 -*-
"""Modo de desenho e simulacao de circuitos e elementos graficos."""

from pathlib import Path
import json

import control as ctrl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import streamlit as st
import streamlit.components.v1 as components

from modo_guia_estudos import render_guia_janela


BASE_DIR = Path(__file__).parent
FRONTEND_PATH = BASE_DIR / "circuitos_frontend" / "index.html"

PRESETS = {
    "RC - 1 ordem": [
        {"type": "fonte_v", "value": 5, "x": 150, "y": 300, "rotation": 0},
        {"type": "resistor", "value": 1000, "x": 300, "y": 300, "rotation": 0},
        {"type": "capacitor", "value": 10, "x": 450, "y": 300, "rotation": 0},
    ],
    "RLC - 2 ordem": [
        {"type": "fonte_v", "value": 5, "x": 130, "y": 300, "rotation": 0},
        {"type": "resistor", "value": 100, "x": 270, "y": 300, "rotation": 0},
        {"type": "indutor", "value": 100, "x": 410, "y": 300, "rotation": 0},
        {"type": "capacitor", "value": 10, "x": 550, "y": 300, "rotation": 0},
    ],
    "Massa-mola-amortecedor": [
        {"type": "forca", "value": 1, "x": 130, "y": 300, "rotation": 0},
        {"type": "massa", "value": 1, "x": 280, "y": 300, "rotation": 0},
        {"type": "mola", "value": 10, "x": 430, "y": 300, "rotation": 0},
        {"type": "amortecedor", "value": 2, "x": 580, "y": 300, "rotation": 0},
    ],
}

VALID_TYPES = {
    "resistor", "capacitor", "indutor", "fonte_v",
    "forca", "massa", "mola", "amortecedor",
}


def _load_circuit_editor_html():
    return FRONTEND_PATH.read_text(encoding="utf-8")


def _parse_elements(json_text):
    if isinstance(json_text, list):
        raw = json_text
    else:
        raw = json.loads(json_text or "[]")
    if not isinstance(raw, list):
        raise ValueError("A definicao precisa ser uma lista JSON de elementos.")

    elements = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tipo = item.get("type")
        if tipo not in VALID_TYPES:
            continue
        value = float(item.get("value", 0))
        elements.append({
            "type": tipo,
            "value": value,
            "x": int(float(item.get("x", 0))) if "x" in item else None,
            "y": int(float(item.get("y", 0))) if "y" in item else None,
            "rotation": int(float(item.get("rotation", 0))),
        })
    return elements


def _safe_elements_for_canvas(json_text):
    try:
        return _parse_elements(json_text)
    except Exception:
        return []


def _first_value(grouped, key, default=None):
    values = grouped.get(key, [])
    if not values:
        return default
    return float(values[0])


def _group_values(elements):
    grouped = {}
    for item in elements:
        grouped.setdefault(item["type"], []).append(float(item["value"]))
    return grouped


def calcular_sistema(elements):
    grouped = _group_values(elements)

    resistor = _first_value(grouped, "resistor")
    capacitor_uf = _first_value(grouped, "capacitor")
    indutor_mh = _first_value(grouped, "indutor")
    fonte_v = _first_value(grouped, "fonte_v", 1.0)

    if resistor and capacitor_uf and not indutor_mh:
        cap = capacitor_uf * 1e-6
        tau = resistor * cap
        if tau <= 0:
            raise ValueError("R e C precisam ser maiores que zero.")
        tf_obj = ctrl.tf([fonte_v], [tau, 1])
        params = {
            "Sistema": "RC - 1 ordem",
            "tau": f"{tau * 1e3:.4g} ms",
            "wc": f"{1 / tau:.4g} rad/s",
            "fc": f"{1 / (2 * np.pi * tau):.4g} Hz",
            "Polo": f"s = -{1 / tau:.4g}",
            "Vs": f"{fonte_v:.4g} V",
        }
        return tf_obj, params

    if resistor and capacitor_uf and indutor_mh:
        cap = capacitor_uf * 1e-6
        ind = indutor_mh * 1e-3
        if cap <= 0 or ind <= 0:
            raise ValueError("L e C precisam ser maiores que zero.")
        wn = 1 / np.sqrt(ind * cap)
        zeta = resistor / 2 * np.sqrt(cap / ind)
        tf_obj = ctrl.tf([fonte_v * wn ** 2], [1, 2 * zeta * wn, wn ** 2])
        tipo = "Subamortecido" if zeta < 1 else (
            "Criticamente amortecido" if abs(zeta - 1) < 0.01 else "Superamortecido"
        )
        params = {
            "Sistema": "RLC - 2 ordem",
            "wn": f"{wn:.4g} rad/s",
            "zeta": f"{zeta:.4g}",
            "Tipo": tipo,
            "Vs": f"{fonte_v:.4g} V",
        }
        return tf_obj, params

    massa = _first_value(grouped, "massa")
    mola = _first_value(grouped, "mola")
    amortecedor = _first_value(grouped, "amortecedor")
    forca = _first_value(grouped, "forca", 1.0)

    if massa and mola and amortecedor is not None:
        if massa <= 0 or mola <= 0:
            raise ValueError("m e k precisam ser maiores que zero.")
        wn = np.sqrt(mola / massa)
        zeta = amortecedor / (2 * np.sqrt(mola * massa))
        tf_obj = ctrl.tf([forca / massa], [1, amortecedor / massa, mola / massa])
        tipo = "Subamortecido" if zeta < 1 else (
            "Criticamente amortecido" if abs(zeta - 1) < 0.01 else "Superamortecido"
        )
        params = {
            "Sistema": "Massa-mola-amortecedor",
            "wn": f"{wn:.4g} rad/s",
            "zeta": f"{zeta:.4g}",
            "Tipo": tipo,
            "y_final": f"{forca / mola:.4g} m",
        }
        return tf_obj, params

    return None, {
        "Sistema": (
            "Monte RC, RLC ou Massa-mola-amortecedor para simular."
        )
    }


def _estimate_time(tf_obj):
    poles = ctrl.poles(tf_obj)
    stable_real = [abs(p.real) for p in poles if p.real < -1e-9]
    if stable_real:
        return float(np.clip(25 / min(stable_real), 0.02, 50))
    return 5.0


def gerar_plots(tf_obj):
    poles = list(ctrl.poles(tf_obj))
    zeros = list(ctrl.zeros(tf_obj))

    t_end = _estimate_time(tf_obj)
    t = np.linspace(0, t_end, 1200)
    t_out, y_out = ctrl.step_response(tf_obj, T=t)
    y_final = float(y_out[-1])

    den = np.asarray(tf_obj.den[0][0], dtype=float)
    num = np.asarray(tf_obj.num[0][0], dtype=float)
    try:
        ref = max(abs(p) for p in poles) if poles else 1.0
        w = np.logspace(np.log10(max(ref / 200, 1e-3)), np.log10(max(ref * 200, 1)), 600)
        _, mag, phase = signal.bode(signal.TransferFunction(num, den), w=w)
    except Exception:
        w = np.logspace(-2, 4, 600)
        _, mag, phase = signal.bode(signal.TransferFunction(num, den), w=w)

    fig_step = go.Figure()
    fig_step.add_trace(go.Scatter(
        x=t_out,
        y=y_out,
        mode="lines",
        line=dict(color="#185fa5", width=2.5),
        hovertemplate="%{x:.4g} s -> %{y:.4g}<extra></extra>",
    ))
    fig_step.update_layout(
        title="Resposta ao degrau",
        xaxis_title="Tempo (s)",
        yaxis_title="Saida",
        height=300,
        margin=dict(l=50, r=20, t=45, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf8",
        showlegend=False,
    )
    fig_step.update_xaxes(gridcolor="#e8e6de")
    fig_step.update_yaxes(gridcolor="#e8e6de")

    fig_bode = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Magnitude (dB)", "Fase (graus)"),
        vertical_spacing=0.1,
    )
    fig_bode.add_trace(go.Scatter(x=w, y=mag, mode="lines", line=dict(color="#0f6e56", width=2)), row=1, col=1)
    fig_bode.add_trace(go.Scatter(x=w, y=phase, mode="lines", line=dict(color="#854f0b", width=2)), row=2, col=1)
    fig_bode.update_xaxes(type="log", gridcolor="#e8e6de", title_text="w (rad/s)", row=2, col=1)
    fig_bode.update_xaxes(type="log", gridcolor="#e8e6de", row=1, col=1)
    fig_bode.update_yaxes(gridcolor="#e8e6de")
    fig_bode.update_layout(
        height=420,
        margin=dict(l=50, r=20, t=55, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf8",
        showlegend=False,
    )

    fig_s = go.Figure()
    fig_s.add_vrect(x0=-1e6, x1=0, fillcolor="#e1f5ee", opacity=0.18, line_width=0)
    fig_s.add_vrect(x0=0, x1=1e6, fillcolor="#fcebeb", opacity=0.18, line_width=0)
    fig_s.add_vline(x=0, line=dict(color="#c8c6be", width=1))
    fig_s.add_hline(y=0, line=dict(color="#c8c6be", width=1))
    if poles:
        fig_s.add_trace(go.Scatter(
            x=[p.real for p in poles],
            y=[p.imag for p in poles],
            mode="markers",
            marker=dict(symbol="x", size=14, color="#d85a30", line_width=2.5),
            name="Polos",
        ))
    if zeros:
        fig_s.add_trace(go.Scatter(
            x=[z.real for z in zeros],
            y=[z.imag for z in zeros],
            mode="markers",
            marker=dict(symbol="circle-open", size=12, color="#0f6e56", line_width=2.5),
            name="Zeros",
        ))
    fig_s.update_layout(
        title="Plano s",
        xaxis_title="Re(s)",
        yaxis_title="Im(s)",
        height=300,
        margin=dict(l=50, r=20, t=45, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf8",
    )
    fig_s.update_xaxes(gridcolor="#e8e6de")
    fig_s.update_yaxes(gridcolor="#e8e6de")

    return fig_step, fig_bode, fig_s, y_final, poles, zeros


def _render_metric_grid(params):
    st.markdown(
        """
        <style>
        .circuit-metric {
            background: #f8f8f6;
            border: 1px solid #c8c6be;
            border-radius: 8px;
            padding: 10px 12px;
            text-align: center;
            min-height: 72px;
        }
        .circuit-metric-value {
            color: #185fa5;
            font-size: 18px;
            font-weight: 750;
            line-height: 1.25;
        }
        .circuit-metric-label {
            color: #5f5e5a;
            font-size: 11px;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    visible = [(k, v) for k, v in params.items() if k != "Sistema"]
    if not visible:
        return
    cols = st.columns(len(visible))
    for col, (label, value) in zip(cols, visible):
        with col:
            st.markdown(
                f"""
                <div class="circuit-metric">
                  <div class="circuit-metric-value">{value}</div>
                  <div class="circuit-metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def modo_circuitos():
    if "circuitos_json" not in st.session_state:
        st.session_state.circuitos_json = json.dumps(PRESETS["RC - 1 ordem"], indent=2)

    with st.sidebar:
        st.header("Navegação")
        if st.button("← Voltar à Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Presets")
        preset = st.selectbox("Sistema", list(PRESETS.keys()))
        if st.button("Carregar preset", use_container_width=True):
            st.session_state.circuitos_json = json.dumps(PRESETS[preset], indent=2)
            st.rerun()

        st.markdown("---")
        st.header("Definição manual")
        manual_json = st.text_area(
            "JSON dos elementos",
            key="circuitos_json",
            height=210,
        )
        st.caption("Tipos: resistor, capacitor, indutor, fonte_v, massa, mola, amortecedor, forca.")

    render_guia_janela("Guia")

    st.title("Modo Circuitos e Elementos Gráficos")
    st.caption("Editor visual para montar circuitos eletricos e analogias mecanicas.")

    initial_elements = _safe_elements_for_canvas(st.session_state.circuitos_json)
    html_content = _load_circuit_editor_html().replace(
        "__INITIAL_ELEMENTS__",
        json.dumps(initial_elements, ensure_ascii=False),
    )
    components.html(html_content, height=632, scrolling=False)

    col_run, col_note = st.columns([1, 3])
    with col_run:
        simular = st.button("Simular", type="primary", use_container_width=True)
    with col_note:
        st.info("Para simular alteracoes feitas no canvas, copie o JSON do painel do editor e cole na definicao manual.")

    if not simular:
        return

    try:
        elements = _parse_elements(manual_json)
        tf_obj, params = calcular_sistema(elements)
        if tf_obj is None:
            st.warning(params["Sistema"])
            return

        st.subheader(params["Sistema"])
        _render_metric_grid(params)
        fig_step, fig_bode, fig_s, y_final, poles, zeros = gerar_plots(tf_obj)

        tab_step, tab_bode, tab_s = st.tabs([
            "Resposta ao degrau",
            "Bode",
            "Plano s",
        ])
        with tab_step:
            st.plotly_chart(fig_step, use_container_width=True)
            st.metric("Valor final aproximado", f"{y_final:.4g}")
        with tab_bode:
            st.plotly_chart(fig_bode, use_container_width=True)
        with tab_s:
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown("**Polos**")
            for pole in poles:
                status = "estavel" if pole.real < 0 else "instavel"
                st.code(f"s = {pole.real:.4g} + j{pole.imag:.4g}  ({status})")
            if zeros:
                st.markdown("**Zeros**")
                for zero in zeros:
                    st.code(f"s = {zero.real:.4g} + j{zero.imag:.4g}")
    except Exception as exc:
        st.error(f"Erro na simulacao: {exc}")
