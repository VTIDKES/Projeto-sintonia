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
FRONTEND_CANDIDATES = (
    FRONTEND_PATH,
    BASE_DIR / "circuitos_frontend" / "Index.html",
    BASE_DIR / "Circuitos_frontend" / "index.html",
    BASE_DIR / "Circuitos_Frontend" / "index.html",
)

FALLBACK_EDITOR_HTML = r"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:Segoe UI,Arial,sans-serif}
body{background:#f8f8f6;color:#2c2c2a}
#app{display:grid;grid-template-columns:150px minmax(0,1fr)230px;height:600px;border:1px solid #c8c6be;border-radius:10px;overflow:hidden}
#palette,#props{background:#f1efe8;padding:10px;overflow:auto}
#palette{border-right:1px solid #c8c6be}
#props{border-left:1px solid #c8c6be}
h4{font-size:12px;margin:10px 0 8px;color:#5f5e5a;text-transform:uppercase;letter-spacing:.04em}
.item{width:100%;border:1px solid #c8c6be;border-left:4px solid #185fa5;border-radius:7px;background:#fff;margin:0 0 7px;padding:8px 6px;cursor:pointer;font-size:12px;font-weight:650;color:#444441}
.item.mec{border-left-color:#0f6e56}.item.src{border-left-color:#854f0b}
#stage{position:relative;background:#fafaf8;overflow:hidden}
#cv{width:100%;height:100%;display:block}
#hint{position:absolute;left:10px;bottom:8px;color:#888780;font-size:11px;pointer-events:none}
#toolbar{position:absolute;right:10px;top:10px;display:flex;gap:6px;z-index:2}
button{border:1px solid #c8c6be;border-radius:6px;background:#fff;color:#444441;padding:6px 9px;font-size:12px;cursor:pointer}
button:hover{border-color:#185fa5;color:#185fa5;background:#e6f1fb}
label{display:block;margin-top:8px;color:#5f5e5a;font-size:11px;font-weight:650}
input,textarea{width:100%;border:1px solid #c8c6be;border-radius:6px;background:#fff;color:#2c2c2a;padding:6px 7px;font-size:12px}
textarea{height:250px;font-family:Consolas,monospace;font-size:10px;line-height:1.35;resize:none}
.empty{margin-top:16px;color:#888780;font-size:12px;line-height:1.45;text-align:center}
.selected-title{font-size:13px;font-weight:750;margin:8px 0;color:#2c2c2a}
@media(max-width:780px){#app{grid-template-columns:120px minmax(0,1fr);height:720px}#props{grid-column:1/-1;border-left:0;border-top:1px solid #c8c6be;max-height:240px}}
</style>
</head>
<body>
<div id="app">
  <aside id="palette">
    <h4>Eletricos</h4>
    <button class="item" data-type="resistor">Resistor</button>
    <button class="item" data-type="capacitor">Capacitor</button>
    <button class="item" data-type="indutor">Indutor</button>
    <h4>Fontes</h4>
    <button class="item src" data-type="fonte_v">Fonte V</button>
    <button class="item src" data-type="forca">Forca</button>
    <h4>Mecanicos</h4>
    <button class="item mec" data-type="massa">Massa</button>
    <button class="item mec" data-type="mola">Mola</button>
    <button class="item mec" data-type="amortecedor">Amortecedor</button>
  </aside>
  <main id="stage">
    <div id="toolbar">
      <button id="arrange">Organizar</button>
      <button id="clear">Limpar</button>
      <button id="copy">Copiar JSON</button>
    </div>
    <canvas id="cv"></canvas>
    <div id="hint">Clique para adicionar, arraste para mover e copie o JSON para simular.</div>
  </main>
  <aside id="props">
    <h4>Propriedades</h4>
    <div id="prop"></div>
    <h4>JSON</h4>
    <textarea id="json"></textarea>
  </aside>
</div>
<script>
const initialElements=__INITIAL_ELEMENTS__;
const types={
  resistor:{label:"Resistor",symbol:"R",unit:"ohm",value:1000,color:"#185fa5",bg:"#e6f1fb"},
  capacitor:{label:"Capacitor",symbol:"C",unit:"uF",value:10,color:"#185fa5",bg:"#e6f1fb"},
  indutor:{label:"Indutor",symbol:"L",unit:"mH",value:100,color:"#185fa5",bg:"#e6f1fb"},
  fonte_v:{label:"Fonte V",symbol:"Vs",unit:"V",value:5,color:"#854f0b",bg:"#faeeda"},
  forca:{label:"Forca",symbol:"F",unit:"N",value:1,color:"#854f0b",bg:"#faeeda"},
  massa:{label:"Massa",symbol:"m",unit:"kg",value:1,color:"#0f6e56",bg:"#e1f5ee"},
  mola:{label:"Mola",symbol:"k",unit:"N/m",value:10,color:"#0f6e56",bg:"#e1f5ee"},
  amortecedor:{label:"Amortecedor",symbol:"b",unit:"N.s/m",value:2,color:"#0f6e56",bg:"#e1f5ee"}
};
const canvas=document.getElementById("cv"),ctx=canvas.getContext("2d"),stage=document.getElementById("stage"),prop=document.getElementById("prop"),jsonBox=document.getElementById("json");
let elements=(Array.isArray(initialElements)?initialElements:[]).filter(e=>types[e.type]).map((e,i)=>({type:e.type,value:+e.value||types[e.type].value,x:+e.x||160+i*130,y:+e.y||300}));
let selected=null,drag=null,dx=0,dy=0;
function resize(){const r=stage.getBoundingClientRect();canvas.width=r.width;canvas.height=r.height;draw()} new ResizeObserver(resize).observe(stage); resize();
function serial(){return elements.map(e=>({type:e.type,value:+e.value,x:Math.round(e.x),y:Math.round(e.y),rotation:0}))}
function sync(){jsonBox.value=JSON.stringify(serial(),null,2)}
function hit(x,y){for(let i=elements.length-1;i>=0;i--){const e=elements[i];if(x>=e.x-48&&x<=e.x+48&&y>=e.y-32&&y<=e.y+32)return e}return null}
function symbol(e){const t=types[e.type];ctx.save();ctx.translate(e.x,e.y);ctx.fillStyle=t.bg;ctx.strokeStyle=t.color;ctx.lineWidth=e===selected?3:1.5;round(-48,-32,96,64,8);ctx.fill();ctx.stroke();ctx.fillStyle=t.color;ctx.font="bold 18px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(t.symbol,0,-5);ctx.fillStyle="#5f5e5a";ctx.font="11px Arial";ctx.fillText(`${Number(e.value).toPrecision(4)} ${t.unit}`,0,18);ctx.fillStyle=t.color;ctx.beginPath();ctx.arc(-48,0,4,0,Math.PI*2);ctx.arc(48,0,4,0,Math.PI*2);ctx.fill();ctx.restore()}
function round(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath()}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="rgba(0,0,0,.05)";for(let x=0;x<canvas.width;x+=24){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}for(let y=0;y<canvas.height;y+=24){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}let sorted=[...elements].sort((a,b)=>a.x-b.x);ctx.strokeStyle="#444441";ctx.lineWidth=1.5;for(let i=0;i<sorted.length-1;i++){ctx.beginPath();ctx.moveTo(sorted[i].x+48,sorted[i].y);ctx.lineTo(sorted[i+1].x-48,sorted[i+1].y);ctx.stroke()}elements.forEach(symbol);sync()}
function renderProp(){if(!selected){prop.innerHTML='<div class="empty">Selecione um elemento.</div>';return}const t=types[selected.type];prop.innerHTML=`<div class="selected-title">${t.label}</div><label>Valor</label><input id="val" type="number" step="any" value="${selected.value}"><label>X</label><input id="x" type="number" value="${Math.round(selected.x)}"><label>Y</label><input id="y" type="number" value="${Math.round(selected.y)}"><button id="del" style="margin-top:10px;width:100%">Remover</button>`;document.getElementById("val").oninput=e=>{selected.value=+e.target.value||selected.value;draw()};document.getElementById("x").oninput=e=>{selected.x=+e.target.value||selected.x;draw()};document.getElementById("y").oninput=e=>{selected.y=+e.target.value||selected.y;draw()};document.getElementById("del").onclick=()=>{elements=elements.filter(e=>e!==selected);selected=null;renderProp();draw()}}
document.querySelectorAll(".item").forEach(b=>b.onclick=()=>{const n=elements.length;const e={type:b.dataset.type,value:types[b.dataset.type].value,x:150+n*100,y:canvas.height/2};elements.push(e);selected=e;renderProp();draw()});
canvas.onmousedown=e=>{const r=canvas.getBoundingClientRect();selected=hit(e.clientX-r.left,e.clientY-r.top);if(selected){drag=selected;dx=e.clientX-r.left-selected.x;dy=e.clientY-r.top-selected.y}renderProp();draw()};
canvas.onmousemove=e=>{if(!drag)return;const r=canvas.getBoundingClientRect();drag.x=Math.round((e.clientX-r.left-dx)/10)*10;drag.y=Math.round((e.clientY-r.top-dy)/10)*10;draw()};
window.onmouseup=()=>drag=null;
document.getElementById("arrange").onclick=()=>{elements.forEach((e,i)=>{e.x=130+i*125;e.y=canvas.height/2});draw()};
document.getElementById("clear").onclick=()=>{elements=[];selected=null;renderProp();draw()};
document.getElementById("copy").onclick=()=>navigator.clipboard&&navigator.clipboard.writeText(jsonBox.value);
renderProp();draw();
</script>
</body>
</html>
"""

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
    for path in FRONTEND_CANDIDATES:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return FALLBACK_EDITOR_HTML


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

    st.title("Modo Simulação com Elementos")
    st.caption("Editor visual para manipular elementos graficos eletricos e mecanicos.")

    initial_elements = _safe_elements_for_canvas(st.session_state.circuitos_json)
    html_template = _load_circuit_editor_html()
    html_content = html_template.replace(
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
