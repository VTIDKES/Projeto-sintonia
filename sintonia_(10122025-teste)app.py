# app.py
# Streamlit (Python) version of your React "ControlSystemDesigner"
# - Sidebar: add/delete blocks
# - Diagram tab: shows blocks + connections (interactive pan/zoom via Plotly)
# - Analysis tab: Bode, Poles/Zeros, Step, Nyquist (using SciPy + NumPy)
#
# Run:
#   pip install streamlit numpy scipy plotly
#   streamlit run app.py

import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal


# =========================
# Utils / Parsing
# =========================

def _now_id() -> int:
    return int(time.time() * 1000)


def parse_num_coeffs(num_text: str) -> List[float]:
    """
    JS: block.data.num.split(',').map(parseFloat || 1)
    Here: allow "1" or "1,0,0"
    """
    s = (num_text or "").strip()
    if not s:
        return [1.0]
    parts = [p.strip() for p in s.split(",")]
    coeffs = []
    for p in parts:
        if p == "":
            continue
        try:
            coeffs.append(float(p))
        except ValueError:
            coeffs.append(1.0)
    return coeffs if coeffs else [1.0]


def parse_den_string(den_str: str) -> List[float]:
    """
    Very similar to your JS "parse simplificado":
    - If contains s^2 or s² -> try extract a*s^2 + b*s + c
    - Else -> try extract a*s + b  (or "s+1" means a=1, b=1)
    """
    den_str = (den_str or "").strip().replace(" ", "")
    if not den_str:
        return [1.0, 1.0]

    # Normalize some unicode
    den_norm = den_str.replace("²", "^2")

    if "s^2" in den_norm or "s^2" in den_norm:
        # Try patterns like: "s^2+2s+1", "2s^2+3s+4", "s^2+s+1"
        m = re.match(r"(?:(-?\d*\.?\d*)?)s\^2([+-]\d*\.?\d*)?s?([+-]\d*\.?\d*)?$", den_norm)
        # fallback: broader match for "a*s^2 + b*s + c"
        if not m:
            m = re.match(r"(-?\d*\.?\d*)?s\^2([+-]\d*\.?\d*)s([+-]\d*\.?\d*)", den_norm)

        if m:
            a_txt, b_txt, c_txt = (m.group(1), m.group(2), m.group(3))
            a = float(a_txt) if a_txt not in (None, "", "+", "-") else (-1.0 if a_txt == "-" else 1.0)
            b = float(b_txt) if b_txt not in (None, "", "+", "-") else (0.0 if b_txt in (None, "") else (-1.0 if b_txt == "-" else 1.0))
            c = float(c_txt) if c_txt not in (None, "", "+", "-") else (0.0 if c_txt in (None, "") else (-1.0 if c_txt == "-" else 1.0))
            return [a, b, c]
        return [1.0, 2.0, 1.0]

    # First-order: "s+1", "2s+3", "s-4"
    m = re.match(r"(?:(-?\d*\.?\d*)?)s([+-]\d*\.?\d*)?$", den_norm)
    if m:
        a_txt, b_txt = m.group(1), m.group(2)
        a = float(a_txt) if a_txt not in (None, "", "+", "-") else (-1.0 if a_txt == "-" else 1.0)
        b = float(b_txt) if b_txt not in (None, "", "+", "-") else (0.0 if b_txt in (None, "") else (-1.0 if b_txt == "-" else 1.0))
        # JS default for first order was [1,1] if fail
        if b == 0.0 and ("+" not in den_norm and "-" not in den_norm[1:]):
            b = 1.0
        return [a, b]

    # fallback: try a + b (no s) -> treat as constant + 1*s? (not ideal)
    return [1.0, 1.0]


# =========================
# Control Analysis (Python)
# =========================

class ControlAnalysis:
    @staticmethod
    def tf_from_coeffs(num: List[float], den: List[float]) -> signal.TransferFunction:
        num = np.array(num, dtype=float)
        den = np.array(den, dtype=float)
        return signal.TransferFunction(num, den)

    @staticmethod
    def bode(num: List[float], den: List[float], w_min_exp: float = -2, w_max_exp: float = 2, points: int = 200):
        w = np.logspace(w_min_exp, w_max_exp, points)
        sys = ControlAnalysis.tf_from_coeffs(num, den)
        w, mag, phase = signal.bode(sys, w=w)  # mag in dB, phase in deg
        return w, mag, phase

    @staticmethod
    def poles_zeros(num: List[float], den: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        zeros = np.roots(num) if len(num) > 1 else np.array([])
        poles = np.roots(den) if len(den) > 1 else np.array([])
        return zeros, poles

    @staticmethod
    def step(num: List[float], den: List[float], t_max: float = 10.0, points: int = 400):
        sys = ControlAnalysis.tf_from_coeffs(num, den)
        t = np.linspace(0, t_max, points)
        tout, y = signal.step(sys, T=t)
        return tout, y

    @staticmethod
    def nyquist(num: List[float], den: List[float], w_min_exp: float = -2, w_max_exp: float = 2, points: int = 400):
        w = np.logspace(w_min_exp, w_max_exp, points)
        sys = ControlAnalysis.tf_from_coeffs(num, den)
        _, H = signal.freqresp(sys, w=w)
        return H.real, H.imag


# =========================
# Block Model
# =========================

BLOCK_TYPES = [
    {
        "id": "transfer",
        "name": "Função de Transferência",
        "icon": "□",
        "color": "#3b82f6",
        "defaultData": {"num": "1", "den": "s+1"},
        "inputs": 1,
        "outputs": 1,
    },
    {
        "id": "sum",
        "name": "Somador",
        "icon": "⊕",
        "color": "#10b981",
        "defaultData": {"signs": ["+", "+"]},
        "inputs": 2,
        "outputs": 1,
    },
    {
        "id": "gain",
        "name": "Ganho",
        "icon": "K",
        "color": "#f59e0b",
        "defaultData": {"value": "1"},
        "inputs": 1,
        "outputs": 1,
    },
    {
        "id": "input",
        "name": "Entrada",
        "icon": "→",
        "color": "#8b5cf6",
        "defaultData": {"label": "R(s)"},
        "inputs": 0,
        "outputs": 1,
    },
    {
        "id": "output",
        "name": "Saída",
        "icon": "⊣",
        "color": "#ec4899",
        "defaultData": {"label": "Y(s)"},
        "inputs": 1,
        "outputs": 0,
    },
    {
        "id": "branch",
        "name": "Ponto de Ramificação",
        "icon": "•",
        "color": "#6366f1",
        "defaultData": {},
        "inputs": 1,
        "outputs": 2,
    },
]


def type_by_id(tid: str) -> Dict:
    for t in BLOCK_TYPES:
        if t["id"] == tid:
            return t
    return BLOCK_TYPES[0]


def ensure_state():
    if "blocks" not in st.session_state:
        st.session_state.blocks = []
    if "connections" not in st.session_state:
        st.session_state.connections = []
    if "selected_block_id" not in st.session_state:
        st.session_state.selected_block_id = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "diagram"
    if "analysis" not in st.session_state:
        st.session_state.analysis = None


def add_block(t: Dict):
    st.session_state.blocks.append(
        {
            "id": _now_id(),
            "type": t["id"],
            "x": 200.0 + 30.0 * (len(st.session_state.blocks) % 8),
            "y": 200.0 + 30.0 * (len(st.session_state.blocks) % 8),
            "data": dict(t["defaultData"]),
            "color": t["color"],
            "inputs": t["inputs"],
            "outputs": t["outputs"],
        }
    )


def delete_block(block_id: int):
    st.session_state.blocks = [b for b in st.session_state.blocks if b["id"] != block_id]
    st.session_state.connections = [
        c for c in st.session_state.connections if c["from"] != block_id and c["to"] != block_id
    ]
    if st.session_state.selected_block_id == block_id:
        st.session_state.selected_block_id = None


def get_block(block_id: int) -> Optional[Dict]:
    for b in st.session_state.blocks:
        if b["id"] == block_id:
            return b
    return None


def set_block(block_id: int, new_block: Dict):
    st.session_state.blocks = [new_block if b["id"] == block_id else b for b in st.session_state.blocks]


def add_connection(from_id: int, to_id: int):
    if from_id == to_id:
        return
    # avoid duplicates
    for c in st.session_state.connections:
        if c["from"] == from_id and c["to"] == to_id:
            return
    st.session_state.connections.append({"id": _now_id(), "from": from_id, "to": to_id})


# =========================
# Diagram (Plotly)
# =========================

def diagram_figure(blocks: List[Dict], connections: List[Dict]) -> go.Figure:
    fig = go.Figure()

    # Draw connections as lines (arrows approximated)
    id_to_block = {b["id"]: b for b in blocks}

    for c in connections:
        a = id_to_block.get(c["from"])
        b = id_to_block.get(c["to"])
        if not a or not b:
            continue
        x1, y1 = a["x"] + 160, a["y"] + 40
        x2, y2 = b["x"], b["y"] + 40
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw blocks as rectangles (shapes) and labels (annotations)
    shapes = []
    annotations = []
    for b in blocks:
        x, y = b["x"], b["y"]
        w, h = 160, 80
        shapes.append(
            dict(
                type="rect",
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                line=dict(width=3, color=b["color"]),
                fillcolor="rgba(30, 41, 59, 0.85)",
            )
        )
        t = type_by_id(b["type"])
        title = t["name"]
        content = ""
        if b["type"] == "transfer":
            content = f"{b['data'].get('num','1')}/{b['data'].get('den','s+1')}"
        elif b["type"] == "gain":
            content = str(b["data"].get("value", "1"))
        elif b["type"] in ("input", "output"):
            content = str(b["data"].get("label", ""))
        elif b["type"] == "sum":
            content = "Σ"
        elif b["type"] == "branch":
            content = "•"

        annotations.append(
            dict(
                x=x + w / 2,
                y=y + h - 16,
                text=title,
                showarrow=False,
                font=dict(size=12),
            )
        )
        annotations.append(
            dict(
                x=x + w / 2,
                y=y + h / 2 - 10,
                text=f"<b>{content}</b>",
                showarrow=False,
                font=dict(size=16),
            )
        )

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", autorange="reversed"),
        dragmode="pan",
        height=640,
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


# =========================
# Analysis Plots (Plotly)
# =========================

def bode_fig(w, mag, phase) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=mag, mode="lines", name="Magnitude (dB)"))
    fig.update_xaxes(type="log", title="ω (rad/s)")
    fig.update_yaxes(title="Magnitude (dB)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Bode - Magnitude")
    return fig


def phase_fig(w, phase) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=phase, mode="lines", name="Fase (°)"))
    fig.update_xaxes(type="log", title="ω (rad/s)")
    fig.update_yaxes(title="Fase (°)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Bode - Fase")
    return fig


def step_fig(t, y) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Resposta"))
    fig.update_xaxes(title="Tempo (s)")
    fig.update_yaxes(title="Amplitude")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Resposta ao Degrau")
    return fig


def pole_zero_fig(zeros, poles) -> go.Figure:
    fig = go.Figure()
    if zeros.size:
        fig.add_trace(go.Scatter(x=zeros.real, y=zeros.imag, mode="markers", name="Zeros", marker_symbol="circle-open"))
    if poles.size:
        fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name="Polos", marker_symbol="x"))
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    fig.update_xaxes(title="Re")
    fig.update_yaxes(title="Im", scaleanchor="x")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Polos e Zeros")
    return fig


def nyquist_fig(re, im) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=re, y=im, mode="lines", name="Nyquist"))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode="markers", name="Ponto crítico (-1,0)"))
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    fig.update_xaxes(title="Re")
    fig.update_yaxes(title="Im", scaleanchor="x")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Diagrama de Nyquist")
    return fig


# =========================
# App UI
# =========================

st.set_page_config(page_title="Sistema de Controle - Designer Interativo (Python)", layout="wide")
ensure_state()

st.markdown(
    """
    <style>
      .stApp { background: #0f172a; color: #e2e8f0; }
      h1,h2,h3,h4 { color: #e2e8f0; }
      .small-muted { color: #94a3b8; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_left, col_main, col_right = st.columns([0.9, 2.1, 1.0], gap="large")

# Sidebar-like (left)
with col_left:
    st.title("Sistema de Controle")
    st.caption("Designer Interativo (Streamlit)")

    st.subheader("Elementos")
    for t in BLOCK_TYPES:
        if st.button(f"{t['icon']}  {t['name']}", use_container_width=True):
            add_block(t)

    st.divider()

    st.subheader("Ações")
    if st.button("▶️  Analisar", type="primary", use_container_width=True):
        # mimic your JS: analyze first transfer block
        transfers = [b for b in st.session_state.blocks if b["type"] == "transfer"]
        if not transfers:
            st.warning("Adicione pelo menos um bloco de função de transferência!")
        else:
            b = transfers[0]
            try:
                num = parse_num_coeffs(b["data"].get("num", "1"))
                den = parse_den_string(b["data"].get("den", "s+1"))

                w, mag, ph = ControlAnalysis.bode(num, den)
                z, p = ControlAnalysis.poles_zeros(num, den)
                t, y = ControlAnalysis.step(num, den)
                nr, ni = ControlAnalysis.nyquist(num, den)

                st.session_state.analysis = {
                    "num": num,
                    "den": den,
                    "w": w,
                    "mag": mag,
                    "phase": ph,
                    "zeros": z,
                    "poles": p,
                    "t": t,
                    "y": y,
                    "nyq_re": nr,
                    "nyq_im": ni,
                }
                st.session_state.active_tab = "analysis"
            except Exception as e:
                st.error(f"Erro ao analisar sistema: {e}")

    # zoom controls (diagram is plotly pan/zoom already)
    st.markdown('<div class="small-muted">Dica: no diagrama (Plotly), use scroll para zoom e arraste para pan.</div>', unsafe_allow_html=True)

# Main (center)
with col_main:
    tab1, tab2 = st.tabs(["Diagrama de Blocos", "Análise do Sistema"])
    # Force active tab behavior (Streamlit tabs don't accept programmatic select cleanly)
    # So we just render both; user sees tabs anyway.

    with tab1:
        st.subheader("Diagrama")
        fig = diagram_figure(st.session_state.blocks, st.session_state.connections)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Conexões (modo prático):** escolha origem e destino e clique em *Conectar*.")
        blocks = st.session_state.blocks
        if blocks:
            ids = [b["id"] for b in blocks]
            labels = {b["id"]: f"{type_by_id(b['type'])['name']} — id {b['id']}" for b in blocks}

            c1, c2, c3 = st.columns([1, 1, 0.6])
            with c1:
                from_id = st.selectbox("De (saída)", ids, format_func=lambda i: labels[i], key="conn_from")
            with c2:
                to_id = st.selectbox("Para (entrada)", ids, format_func=lambda i: labels[i], key="conn_to")
            with c3:
                if st.button("Conectar", use_container_width=True):
                    add_connection(from_id, to_id)

            if st.session_state.connections:
                st.write("Conexões atuais:")
                st.dataframe(st.session_state.connections, use_container_width=True, hide_index=True)
        else:
            st.info("Adicione blocos na coluna da esquerda.")

    with tab2:
        st.subheader("Resultados")
        a = st.session_state.analysis
        if not a:
            st.info('Configure ao menos 1 bloco de "Função de Transferência" e clique em **Analisar**.')
        else:
            st.markdown(f"**Função de Transferência analisada:**  \nNumerador = `{a['num']}`  \nDenominador = `{a['den']}`")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(bode_fig(a["w"], a["mag"], a["phase"]), use_container_width=True)
            with c2:
                st.plotly_chart(phase_fig(a["w"], a["phase"]), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(step_fig(a["t"], a["y"]), use_container_width=True)
            with c4:
                st.plotly_chart(pole_zero_fig(a["zeros"], a["poles"]), use_container_width=True)

            st.plotly_chart(nyquist_fig(a["nyq_re"], a["nyq_im"]), use_container_width=True)

# Properties panel (right)
with col_right:
    st.subheader("Propriedades")

    blocks = st.session_state.blocks
    if not blocks:
        st.caption("Selecione um bloco para editar.")
    else:
        options = [b["id"] for b in blocks]
        labels = {b["id"]: f"{type_by_id(b['type'])['name']} — id {b['id']}" for b in blocks}

        sel = st.selectbox(
            "Bloco selecionado",
            options,
            index=0 if st.session_state.selected_block_id is None else max(0, options.index(st.session_state.selected_block_id)) if st.session_state.selected_block_id in options else 0,
            format_func=lambda i: labels[i],
        )
        st.session_state.selected_block_id = sel
        b = get_block(sel)
        if not b:
            st.stop()

        st.write(f"Tipo: **{type_by_id(b['type'])['name']}**")
        st.number_input("Posição X", value=float(b["x"]), step=10.0, key="bx")
        st.number_input("Posição Y", value=float(b["y"]), step=10.0, key="by")
        b["x"] = float(st.session_state.bx)
        b["y"] = float(st.session_state.by)

        # Block-specific fields
        if b["type"] == "transfer":
            num_txt = st.text_input("Numerador (coef. separados por vírgula)", value=b["data"].get("num", "1"))
            den_txt = st.text_input("Denominador (ex: s+1 ou s^2+2s+1)", value=b["data"].get("den", "s+1"))
            b["data"]["num"] = num_txt
            b["data"]["den"] = den_txt

        if b["type"] == "gain":
            val = st.text_input("Valor do ganho", value=b["data"].get("value", "1"))
            b["data"]["value"] = val

        if b["type"] in ("input", "output"):
            lab = st.text_input("Rótulo", value=b["data"].get("label", ""))
            b["data"]["label"] = lab

        color = st.color_picker("Cor", value=b.get("color", "#3b82f6"))
        b["color"] = color

        set_block(b["id"], b)

        st.divider()
        if st.button("🗑️ Deletar bloco", use_container_width=True):
            delete_block(b["id"])
            st.rerun()

        st.divider()
        with st.expander("Exportar / Importar (JSON)"):
            st.caption("Copia/cola para salvar seu diagrama (blocos + conexões).")
            export_obj = {"blocks": st.session_state.blocks, "connections": st.session_state.connections}
            st.code(export_obj, language="python")

            import_txt = st.text_area("Cole aqui um dict Python (mesma estrutura acima) e clique em Importar")
            if st.button("Importar", use_container_width=True):
                try:
                    obj = eval(import_txt, {"__builtins__": {}})
                    st.session_state.blocks = obj.get("blocks", [])
                    st.session_state.connections = obj.get("connections", [])
                    st.success("Importado!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha ao importar: {e}")
