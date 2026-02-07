# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle (Streamlit Cloud)
✅ Inclui Editor Visual nível Simulink (canvas + blocos + conexões) em UMA ÚNICA ARQUIVO.

Como funciona em 1 arquivo:
- O app escreve (na primeira execução) um frontend HTML em /tmp/visual_blocks_frontend/index.html
- Declara um Streamlit Custom Component apontando para esse path
- O editor retorna um JSON com nodes/edges para o Python
- O Python converte automaticamente um DIAGRAMA EM SÉRIE (cadeia única) em TransferFunction
  (G(s) e K e integrador/derivador) e usa isso na simulação em Malha Aberta.
- Sua lógica original de blocos (sidebar: Planta/Controlador/Sensor/Outro) continua intacta.

⚠️ Importante (limitação atual do conversor do diagrama):
- Ele compõe APENAS cadeias simples (um caminho principal entrada → ... → saída).
- Se houver ramificações/feedback/somador complexo, ele avisa e não tenta “adivinhar”.

Requisitos:
- streamlit
- pandas, numpy, sympy, control, scipy, plotly (como no seu app)
(NÃO precisa npm, NÃO precisa streamlit-flow-component, NÃO precisa streamlit-sortables.)
"""

import os
import json
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import control as ctrl
from control import TransferFunction, margin, step_response, forced_response, root_locus
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit.components.v1 as components

# =====================================================
# (A) EDITOR VISUAL - componente embutido em 1 arquivo
# =====================================================

_EDITOR_FRONTEND_DIR = Path("/tmp/visual_blocks_frontend")
_EDITOR_INDEX = _EDITOR_FRONTEND_DIR / "index.html"

_EDITOR_HTML = r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Visual Blocks</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root{
        --bg:#0f2b3a;
        --panel:#16384a;
        --panel2:#0b2230;
        --card:#f5f7fb;
        --accent:#22c55e;
        --warn:#f59e0b;
        --danger:#ef4444;
        --muted:#94a3b8;
      }
      html,body{height:100%; margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;}
      .wrap{height:100%; display:flex; flex-direction:column;}
      .top{
        display:grid; grid-template-columns: 1.2fr 1fr 1fr;
        gap:10px; padding:10px; background:#ffffff;
      }
      .pill{
        border-radius:6px; padding:10px 12px; background:#eef6ff; color:#0f172a;
        display:flex; align-items:center; gap:8px; font-size:13px;
        border:1px solid #e5e7eb;
      }
      .pill.ok{background:#ecfdf5;}
      .pill .dot{width:8px; height:8px; border-radius:999px; background:#3b82f6;}
      .pill.ok .dot{background:var(--accent);}
      .bar{
        display:flex; gap:8px; padding:10px; background:#1f4b63; align-items:center; flex-wrap:wrap;
      }
      .btn{
        border:0; border-radius:8px; padding:7px 10px; font-size:12px; cursor:pointer;
        color:white; background:#2563eb;
        display:flex; align-items:center; gap:6px;
        user-select:none;
      }
      .btn.orange{background:#f59e0b; color:#0f172a;}
      .btn.green{background:#22c55e; color:#0f172a;}
      .btn.purple{background:#a855f7;}
      .btn.red{background:#ef4444;}
      .btn.gray{background:#334155;}
      .content{flex:1; display:flex; min-height:0;}
      .canvasWrap{flex:1; position:relative; background:var(--bg); overflow:hidden;}
      .canvas{
        position:absolute; inset:0;
        background-image:
          linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 24px 24px;
      }
      svg.wires{position:absolute; inset:0; pointer-events:none;}
      .node{
        position:absolute;
        width:160px; min-height:58px;
        border-radius:10px;
        background:#ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        border:1px solid rgba(15,23,42,0.15);
        overflow:hidden;
      }
      .node .head{
        display:flex; align-items:center; justify-content:space-between;
        padding:8px 10px; background:#f1f5f9; font-size:12px; font-weight:650;
      }
      .node .body{padding:8px 10px; font-size:12px; color:#0f172a;}
      .tag{font-size:11px; color:#334155; background:#e2e8f0; padding:2px 6px; border-radius:999px;}
      .ports{position:absolute; inset:0; pointer-events:none;}
      .port{
        width:10px; height:10px; border-radius:999px;
        background:#0ea5e9; position:absolute; pointer-events:auto; cursor:crosshair;
        border:2px solid rgba(255,255,255,0.9);
        box-shadow:0 4px 10px rgba(0,0,0,0.25);
      }
      .port.out{background:#22c55e;}
      .port.in{background:#0ea5e9;}
      .node.sel{outline:3px solid rgba(34,197,94,0.75);}
      .side{
        width:280px; background:#ffffff; border-left:1px solid #e5e7eb;
        padding:10px; display:flex; flex-direction:column; gap:10px;
      }
      .card{border:1px solid #e5e7eb; border-radius:10px; padding:10px; background:#f8fafc;}
      .card h4{margin:0 0 8px 0; font-size:13px;}
      .kv{display:grid; grid-template-columns:1fr auto; gap:6px; font-size:12px;}
      .hint{font-size:12px; color:#334155; line-height:1.3;}
      .input{width:100%; padding:7px 8px; border-radius:8px; border:1px solid #cbd5e1; font-size:12px;}
      .small{font-size:11px; color:#475569;}
      .hr{height:1px; background:#e5e7eb; margin:8px 0;}
      .kbd{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; padding:2px 6px; border:1px solid #cbd5e1; border-bottom-width:2px; border-radius:6px; background:white;}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div class="pill"><span class="dot"></span><strong>Modo Editor Visual Ativo</strong>&nbsp;<span class="small">Construa sistemas de controle visualmente!</span></div>
        <div class="pill ok"><span class="dot"></span><strong>Blocos:</strong>&nbsp;<span id="countNodes">0</span></div>
        <div class="pill ok"><span class="dot"></span><strong>Conexões:</strong>&nbsp;<span id="countEdges">0</span></div>
      </div>

      <div class="bar">
        <span class="small" style="color:#e2e8f0; font-weight:700; margin-right:6px;">BLOCOS:</span>
        <button class="btn" data-add="tf">🧮 Função Transferência</button>
        <button class="btn orange" data-add="gain">🔧 Ganho</button>
        <button class="btn purple" data-add="int">∫ Integrador</button>
        <button class="btn gray" data-add="der">d/dt Derivador</button>
        <button class="btn orange" data-add="sum" title="Somador (ainda não é convertido no backend)">➕ Somador</button>
        <button class="btn orange" data-add="sat" title="Saturação (tratada como 1 no backend)">⬆️/⬇️ Saturação</button>

        <span style="flex:1"></span>
        <span class="small" style="color:#e2e8f0; font-weight:700; margin-right:6px;">AÇÕES:</span>
        <button class="btn red" id="btnDelete">🗑️ Deletar</button>
        <button class="btn gray" id="btnDuplicate">📄 Duplicar</button>
        <button class="btn red" id="btnClear">🧹 Limpar Tudo</button>
        <button class="btn green" id="btnAuto">✨ Auto-organizar</button>
      </div>

      <div class="content">
        <div class="canvasWrap" id="wrap">
          <div class="canvas" id="canvas"></div>
          <svg class="wires" id="wires"></svg>
        </div>

        <div class="side">
          <div class="card">
            <h4>📌 Informações do Sistema</h4>
            <div class="kv"><div>Blocos:</div><div id="sideNodes">0</div></div>
            <div class="kv"><div>Conexões:</div><div id="sideEdges">0</div></div>
            <div class="kv"><div>Selecionado:</div><div id="sideSel">Nenhum</div></div>
          </div>

          <div class="card">
            <h4>⚙️ Parâmetros do bloco</h4>
            <div id="paramArea" class="hint">Selecione um bloco para editar.</div>
          </div>

          <div class="card">
            <h4>💡 Dicas</h4>
            <div class="hint">
              • Arraste blocos para posicionar<br/>
              • Clique num bloco para selecionar<br/>
              • Clique no <span class="kbd">ponto verde</span> (saída) e depois no <span class="kbd">ponto azul</span> (entrada) para conectar<br/>
              • <span class="kbd">Del</span> remove selecionado<br/>
              • <span class="kbd">Ctrl+D</span> duplica selecionado
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Streamlit component lib -->
    <script src="https://unpkg.com/streamlit-component-lib/dist/streamlit-component-lib.js"></script>
    <script>
      const wrap = document.getElementById("wrap");
      const canvas = document.getElementById("canvas");
      const wires = document.getElementById("wires");
      const countNodes = document.getElementById("countNodes");
      const countEdges = document.getElementById("countEdges");
      const sideNodes = document.getElementById("sideNodes");
      const sideEdges = document.getElementById("sideEdges");
      const sideSel = document.getElementById("sideSel");
      const paramArea = document.getElementById("paramArea");

      let model = {nodes: [], edges: []};
      let selId = null;
      let drag = null; // {id,dx,dy}
      let connectFrom = null; // {nodeId, port:"out"}

      const randId = (p)=> p + "_" + Math.random().toString(16).slice(2) + Date.now().toString(16);

      function updateCounters(){
        countNodes.textContent = model.nodes.length;
        countEdges.textContent = model.edges.length;
        sideNodes.textContent = model.nodes.length;
        sideEdges.textContent = model.edges.length;
        sideSel.textContent = selId ? (model.nodes.find(n=>n.id===selId)?.label || selId) : "Nenhum";
      }

      function setSel(id){
        selId = id;
        document.querySelectorAll(".node").forEach(el=>{
          el.classList.toggle("sel", el.dataset.id===id);
        });
        renderParams();
        updateCounters();
      }

      function nodeTitle(type){
        if(type==="tf") return "G(s)";
        if(type==="sum") return "Σ";
        if(type==="gain") return "K";
        if(type==="int") return "1/s";
        if(type==="der") return "s";
        if(type==="sat") return "sat";
        return type;
      }

      function defaultParams(type){
        if(type==="tf") return {num:"1", den:"s+1"};
        if(type==="gain") return {k:"1"};
        if(type==="sum") return {signs:"+ +"};
        if(type==="int") return {};
        if(type==="der") return {};
        if(type==="sat") return {min:"-1", max:"1"};
        return {};
      }

      function addNode(type){
        const r = wrap.getBoundingClientRect();
        const n = {
          id: randId("n"),
          type,
          label: nodeTitle(type),
          x: Math.max(20, r.width*0.12 + Math.random()*60),
          y: Math.max(30, r.height*0.2 + Math.random()*60),
          params: defaultParams(type),
        };
        model.nodes.push(n);
        render();
        setSel(n.id);
        pushToStreamlit();
      }

      function removeSelected(){
        if(!selId) return;
        model.edges = model.edges.filter(e => e.src!==selId && e.dst!==selId);
        model.nodes = model.nodes.filter(n=>n.id!==selId);
        selId = null;
        render();
        pushToStreamlit();
      }

      function duplicateSelected(){
        if(!selId) return;
        const n0 = model.nodes.find(n=>n.id===selId);
        if(!n0) return;
        const n = {...n0, id: randId("n"), x:n0.x+30, y:n0.y+30, params: {...n0.params}};
        model.nodes.push(n);
        render();
        setSel(n.id);
        pushToStreamlit();
      }

      function clearAll(){
        model = {nodes: [], edges: []};
        selId = null;
        connectFrom = null;
        render();
        pushToStreamlit();
      }

      function autoArrange(){
        const marginX = 40, marginY = 80;
        let x = marginX, y = marginY;
        model.nodes.forEach((n,i)=>{
          n.x = x + i*200;
          n.y = y;
        });
        render();
        pushToStreamlit();
      }

      function edgeExists(src,dst){
        return model.edges.some(e => e.src===src && e.dst===dst);
      }

      function connect(srcId, dstId){
        if(srcId===dstId) return;
        if(edgeExists(srcId,dstId)) return;
        model.edges.push({id: randId("e"), src: srcId, srcPort:"out", dst: dstId, dstPort:"in"});
        connectFrom = null;
        render();
        pushToStreamlit();
      }

      function getPortPos(nodeId, kind){
        const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
        if(!nodeEl) return {x:0,y:0};
        const portEl = nodeEl.querySelector(`.port.${kind}`);
        const r = portEl.getBoundingClientRect();
        const rw = wrap.getBoundingClientRect();
        return {x: r.left - rw.left + r.width/2, y: r.top - rw.top + r.height/2};
      }

      function renderWires(){
        while(wires.firstChild) wires.removeChild(wires.firstChild);
        const rw = wrap.getBoundingClientRect();
        wires.setAttribute("width", rw.width);
        wires.setAttribute("height", rw.height);

        model.edges.forEach(e=>{
          const a = getPortPos(e.src, "out");
          const b = getPortPos(e.dst, "in");
          const dx = Math.max(40, (b.x - a.x) * 0.5);
          const c1 = {x: a.x + dx, y: a.y};
          const c2 = {x: b.x - dx, y: b.y};

          const path = document.createElementNS("http://www.w3.org/2000/svg","path");
          const d = `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`;
          path.setAttribute("d", d);
          path.setAttribute("fill","none");
          path.setAttribute("stroke","rgba(34,197,94,0.95)");
          path.setAttribute("stroke-width","3");
          path.setAttribute("stroke-linecap","round");
          wires.appendChild(path);

          const dot = document.createElementNS("http://www.w3.org/2000/svg","circle");
          dot.setAttribute("cx", b.x);
          dot.setAttribute("cy", b.y);
          dot.setAttribute("r", 4.5);
          dot.setAttribute("fill","rgba(34,197,94,1)");
          wires.appendChild(dot);
        });
      }

      function renderParams(){
        if(!selId){
          paramArea.innerHTML = `<div class="hint">Selecione um bloco para editar.</div>`;
          return;
        }
        const n = model.nodes.find(n=>n.id===selId);
        if(!n){
          paramArea.innerHTML = `<div class="hint">Selecione um bloco para editar.</div>`;
          return;
        }

        function inputRow(label, key, value){
          return `
            <div style="margin-bottom:8px;">
              <div class="small" style="margin-bottom:4px;">${label}</div>
              <input class="input" data-param="${key}" value="${String(value ?? "")}"/>
            </div>
          `;
        }

        let html = `<div class="small">Tipo: <strong>${n.type}</strong></div><div class="hr"></div>`;
        if(n.type==="tf"){
          html += inputRow("Numerador (em s)", "num", n.params.num);
          html += inputRow("Denominador (em s)", "den", n.params.den);
        }else if(n.type==="gain"){
          html += inputRow("Ganho K", "k", n.params.k);
        }else if(n.type==="sum"){
          html += inputRow("Sinais (ex: + -)", "signs", n.params.signs);
        }else if(n.type==="sat"){
          html += inputRow("Min", "min", n.params.min);
          html += inputRow("Max", "max", n.params.max);
        }else{
          html += `<div class="hint">Sem parâmetros para este bloco.</div>`;
        }

        paramArea.innerHTML = html;

        paramArea.querySelectorAll("input[data-param]").forEach(inp=>{
          inp.addEventListener("input", (ev)=>{
            const k = ev.target.getAttribute("data-param");
            n.params[k] = ev.target.value;
            pushToStreamlitDebounced();
          });
        });
      }

      function render(){
        canvas.querySelectorAll(".node").forEach(el=>el.remove());

        model.nodes.forEach(n=>{
          const el = document.createElement("div");
          el.className = "node";
          el.dataset.id = n.id;
          el.style.left = `${n.x}px`;
          el.style.top  = `${n.y}px`;

          const bodyText = (n.type==="tf")
            ? `${n.params.num || "1"} / ${n.params.den || "1"}`
            : (n.type==="gain" ? `K=${n.params.k||"1"}` : "");

          el.innerHTML = `
            <div class="head">
              <span>${n.label}</span>
              <span class="tag">${n.type}</span>
            </div>
            <div class="body">${bodyText}</div>
            <div class="ports">
              <div class="port in"  style="left:-6px; top:50%; transform:translateY(-50%);" title="Entrada"></div>
              <div class="port out" style="right:-6px; top:50%; transform:translateY(-50%);" title="Saída"></div>
            </div>
          `;

          canvas.appendChild(el);

          el.addEventListener("mousedown",(ev)=>{
            if(ev.target.classList.contains("port")) return;
            setSel(n.id);
            drag = {id:n.id, dx: ev.offsetX, dy: ev.offsetY};
          });

          const portIn = el.querySelector(".port.in");
          const portOut = el.querySelector(".port.out");

          portOut.addEventListener("mousedown",(ev)=>{
            ev.stopPropagation();
            setSel(n.id);
            connectFrom = {nodeId: n.id, port:"out"};
          });

          portIn.addEventListener("mousedown",(ev)=>{
            ev.stopPropagation();
            setSel(n.id);
            if(connectFrom && connectFrom.port==="out"){
              connect(connectFrom.nodeId, n.id);
            }
          });
        });

        document.querySelectorAll(".node").forEach(el=>{
          el.classList.toggle("sel", el.dataset.id===selId);
        });

        updateCounters();
        renderWires();
        renderParams();
      }

      window.addEventListener("mousemove",(ev)=>{
        if(!drag) return;
        const n = model.nodes.find(n=>n.id===drag.id);
        if(!n) return;
        const rw = wrap.getBoundingClientRect();
        n.x = Math.max(10, Math.min(rw.width-180, ev.clientX - rw.left - drag.dx));
        n.y = Math.max(10, Math.min(rw.height-90, ev.clientY - rw.top - drag.dy));
        const el = document.querySelector(`.node[data-id="${n.id}"]`);
        if(el){
          el.style.left = `${n.x}px`;
          el.style.top  = `${n.y}px`;
        }
        renderWires();
      });

      window.addEventListener("mouseup", ()=>{
        if(drag){
          drag = null;
          pushToStreamlit();
        }
      });

      window.addEventListener("keydown",(ev)=>{
        if(ev.key==="Delete") removeSelected();
        if(ev.ctrlKey && (ev.key==="d" || ev.key==="D")){
          ev.preventDefault();
          duplicateSelected();
        }
      });

      document.querySelectorAll("button[data-add]").forEach(btn=>{
        btn.addEventListener("click", ()=> addNode(btn.getAttribute("data-add")));
      });
      document.getElementById("btnDelete").addEventListener("click", removeSelected);
      document.getElementById("btnDuplicate").addEventListener("click", duplicateSelected);
      document.getElementById("btnClear").addEventListener("click", clearAll);
      document.getElementById("btnAuto").addEventListener("click", autoArrange);

      let sendTimer = null;
      function pushToStreamlit(){
        if(window.Streamlit){
          window.Streamlit.setComponentValue(JSON.stringify(model));
          window.Streamlit.setFrameHeight(document.body.scrollHeight);
        }
      }
      function pushToStreamlitDebounced(){
        clearTimeout(sendTimer);
        sendTimer = setTimeout(pushToStreamlit, 250);
      }

      function onRender(event){
        const data = event.detail.args || {};
        try{
          const incoming = JSON.parse(data.model || "{}");
          if(incoming && incoming.nodes && incoming.edges){
            model = incoming;
          }
        }catch(err){}
        render();
        pushToStreamlit();
      }

      window.Streamlit.events.addEventListener(window.Streamlit.RENDER_EVENT, onRender);
      window.Streamlit.setComponentReady();
      window.Streamlit.setFrameHeight(document.body.scrollHeight);

      new ResizeObserver(()=>{ renderWires(); window.Streamlit.setFrameHeight(document.body.scrollHeight); }).observe(wrap);
    </script>
  </body>
</html>
"""

def _ensure_editor_frontend():
    """Garante que o HTML do editor exista no filesystem (Streamlit Cloud permite escrever em /tmp)."""
    try:
        _EDITOR_FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
        if not _EDITOR_INDEX.exists():
            _EDITOR_INDEX.write_text(_EDITOR_HTML, encoding="utf-8")
    except Exception as e:
        st.error(f"Falha ao preparar o editor visual: {e}")
        st.stop()

_ensure_editor_frontend()

# declara o componente
_visual_blocks = components.declare_component("visual_blocks", path=str(_EDITOR_FRONTEND_DIR))

def visual_blocks_editor(model=None, height=620, key="visual_blocks"):
    if model is None:
        model = {"nodes": [], "edges": []}
    v = _visual_blocks(model=json.dumps(model), height=height, key=key, default=json.dumps(model))
    try:
        return json.loads(v) if isinstance(v, str) else (v or model)
    except Exception:
        return model

# =====================================================
# (B) SEU APP ORIGINAL (com poucas mudanças)
# =====================================================

ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabólica']

def formatar_numero(valor):
    if np.isinf(valor):
        return '∞'
    if np.isnan(valor):
        return '-'
    return f"{valor:.3f}"

def converter_para_tf(numerador_str, denominador_str):
    s = sp.Symbol('s')
    num = parse_expr(str(numerador_str).replace('^', '**'), local_dict={'s': s})
    den = parse_expr(str(denominador_str).replace('^', '**'), local_dict={'s': s})

    num, den = sp.fraction(sp.together(num / den))
    num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
    den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]

    if den_coeffs and den_coeffs[0] != 1:
        fator = den_coeffs[0]
        num_coeffs = [c / fator for c in num_coeffs]
        den_coeffs = [c / fator for c in den_coeffs]

    return TransferFunction(num_coeffs, den_coeffs), (num, den)

def tipo_do_sistema(G):
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    return sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))

def constantes_de_erro(G):
    s = ctrl.tf('s')
    G_min = ctrl.minreal(G, verbose=False)
    tipo = tipo_do_sistema(G_min)

    Kp = Kv = Ka = np.inf
    try:
        if tipo == 0:
            Kp = ctrl.dcgain(G_min)
        elif tipo == 1:
            Kv = ctrl.dcgain(s * G_min)
        else:
            Ka = ctrl.dcgain(s**2 * G_min)
    except Exception:
        pass

    if tipo == 0:
        Kv = Ka = np.inf
    elif tipo == 1:
        Kp = 0
        Ka = np.inf
    else:
        Kp = Kv = 0

    return tipo, Kp, Kv, Ka

def calcular_malha_fechada(planta, controlador=None, sensor=None):
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])
    return ctrl.feedback(controlador * planta, sensor)

def calcular_desempenho(tf):
    den = tf.den[0][0]
    ordem = len(den) - 1
    polos = ctrl.poles(tf)
    gm, pm, wg, wp = margin(tf)
    gm_db = 20 * np.log10(gm) if gm not in [np.inf] and gm > 0 else np.inf

    resultado = {
        'Margem de ganho': f"{formatar_numero(gm)} ({'∞' if gm == np.inf else f'{formatar_numero(gm_db)} dB'})",
        'Margem de fase': f"{formatar_numero(pm)}°",
        'Freq. cruz. fase': f"{formatar_numero(wg)} rad/s",
        'Freq. cruz. ganho': f"{formatar_numero(wp)} rad/s"
    }

    if ordem == 1:
        tau = -1 / polos[0].real
        resultado.update({
            'Tipo': '1ª Ordem',
            'Const. tempo (τ)': f"{formatar_numero(tau)} s",
            'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
            'Temp. acomodação (Ts)': f"{formatar_numero(4 * tau)} s",
            'Freq. natural (ωn)': f"{formatar_numero(1/tau)} rad/s",
            'Fator amortec. (ζ)': "1.0"
        })
        return resultado

    if ordem == 2:
        wn = np.sqrt(np.prod(np.abs(polos))).real
        zeta = -np.real(polos[0]) / wn
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
        Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
        Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
        Tp = np.pi / wd if wd > 0 else float('inf')
        Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')

        resultado.update({
            'Tipo': '2ª Ordem',
            'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
            'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
            'Freq. amortec. (ωd)': f"{formatar_numero(wd)} rad/s",
            'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
            'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
            'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
            'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s"
        })
        return resultado

    # ordem superior (simplificado)
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
    polo_dom = polos_ordenados[0]
    wn = np.abs(polo_dom)
    zeta = -np.real(polo_dom) / wn if wn != 0 else 0
    omega_d = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0

    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
    Tr = (np.pi - np.arccos(zeta)) / omega_d if zeta < 1 and omega_d > 0 else float('inf')
    Tp = np.pi / omega_d if omega_d > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')

    resultado.update({
        'Tipo': f'{ordem}ª Ordem (Polo dominante)',
        'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (ωd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s",
        'Observação': 'Cálculo baseado no polo dominante'
    })
    return resultado

def estimar_tempo_final_simulacao(tf):
    polos = ctrl.poles(tf)
    if len(polos) == 0:
        return 50.0
    if any(np.real(p) > 1e-6 for p in polos):
        return 20.0
    estaveis = [np.real(p) for p in polos if np.real(p) < -1e-6]
    if not estaveis:
        return 100.0
    sigma_dom = max(estaveis)
    ts = 4 / abs(sigma_dom)
    return float(np.clip(ts * 1.5, 10, 500))

def configurar_linhas_interativas(fig):
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(line=dict(color='green', width=2, dash='dash')),
        modebar_add=['drawline','drawopenpath','drawclosedpath','drawcircle','drawrect','eraseshape']
    )
    return fig

def plot_polos_zeros(tf):
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)
    fig = go.Figure()

    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
                                 marker=dict(symbol='circle', size=12, color='blue'), name='Zeros'))

    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
                                 marker=dict(symbol='x', size=12, color='red'), name='Polos'))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Diagrama de Polos e Zeros', xaxis_title='Parte Real', yaxis_title='Parte Imaginária',
                      showlegend=True, hovermode='closest')
    return configurar_linhas_interativas(fig)

def _gerar_sinal_entrada(entrada, t):
    if entrada == 'Degrau':
        return np.ones_like(t)
    if entrada == 'Rampa':
        return t
    if entrada == 'Senoidal':
        return np.sin(2*np.pi*t)
    if entrada == 'Impulso':
        return np.concatenate([[1], np.zeros(len(t)-1)])
    return t**2

def plot_resposta_temporal(sistema, entrada):
    tfinal = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tfinal, 1000)
    u = _gerar_sinal_entrada(entrada, t)

    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=u[:len(t_out)], mode='lines', line=dict(dash='dash', color='blue'), name='Entrada'))
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines', line=dict(color='red'), name='Saída'))
    fig.update_layout(title=f'Resposta Temporal - {entrada}', xaxis_title='Tempo (s)', yaxis_title='Amplitude',
                      showlegend=True, hovermode='x unified')
    return configurar_linhas_interativas(fig)

def plot_bode(sistema, tipo='magnitude'):
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)

    if tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', name='Magnitude'))
        fig.update_layout(title='Bode - Magnitude', xaxis_title="Frequência (rad/s)", yaxis_title="Magnitude (dB)", xaxis_type='log')
        return configurar_linhas_interativas(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', name='Fase'))
    fig.update_layout(title='Bode - Fase', xaxis_title="Frequência (rad/s)", yaxis_title="Fase (deg)", xaxis_type='log')
    return configurar_linhas_interativas(fig)

def plot_lgr(sistema):
    rlist, _ = root_locus(sistema, plot=False)
    fig = go.Figure()
    for r in rlist.T:
        fig.add_trace(go.Scatter(x=np.real(r), y=np.imag(r), mode='lines', showlegend=False))
    fig.update_layout(title='Lugar Geométrico das Raízes (LGR)', xaxis_title='Parte Real', yaxis_title='Parte Imaginária')
    return configurar_linhas_interativas(fig)

def plot_nyquist(sistema):
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H.real, y=H.imag, mode='lines', name='Nyquist'))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers', name='(-1,0)', marker=dict(size=12, color='red')))
    fig.update_layout(title='Diagrama de Nyquist', xaxis_title='Parte Real', yaxis_title='Parte Imaginária')
    return configurar_linhas_interativas(fig)

# ------------------------ Blocos (originais) ------------------------

def inicializar_blocos():
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['uid', 'nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])

def adicionar_bloco(nome, tipo, numerador, denominador):
    try:
        tf, tf_symb = converter_para_tf(numerador, denominador)
        uid = str(uuid.uuid4())
        novo = pd.DataFrame([{
            'uid': uid,
            'nome': nome,
            'tipo': tipo,
            'numerador': str(numerador),
            'denominador': str(denominador),
            'tf': tf,
            'tf_simbolico': tf_symb
        }])
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco {nome} adicionado."
    except Exception as e:
        return False, f"Erro na conversão: {e}"

def remover_bloco(uid):
    df = st.session_state.blocos
    if df.empty:
        return "Nenhum bloco."
    if any(df['uid'].astype(str) == str(uid)):
        nome = df[df['uid'].astype(str) == str(uid)].iloc[0]['nome']
        st.session_state.blocos = df[df['uid'].astype(str) != str(uid)]
        return f"Bloco {nome} excluído."
    # fallback: remove por nome
    nome = str(uid)
    st.session_state.blocos = df[df['nome'].astype(str) != nome]
    return f"Bloco(s) {nome} excluído(s)."

def obter_bloco_por_tipo(tipo):
    df = st.session_state.blocos
    if df.empty:
        return None
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None

# =====================================================
# (C) CONVERSOR DO DIAGRAMA PARA TF (cadeia simples)
# =====================================================

def _tf_from_node(n: dict) -> TransferFunction:
    """Cria um TF a partir de um nó do editor."""
    t = n.get("type")
    params = n.get("params") or {}
    if t == "tf":
        num = params.get("num", "1")
        den = params.get("den", "s+1")
        tf, _ = converter_para_tf(num, den)
        return tf
    if t == "gain":
        k = params.get("k", "1")
        try:
            k_val = float(eval(str(k), {"__builtins__": {}}, {}))
        except Exception:
            k_val = 1.0
        return TransferFunction([k_val], [1])
    if t == "int":
        return TransferFunction([1], [1, 0])  # 1/s
    if t == "der":
        return TransferFunction([1, 0], [1])  # s
    if t == "sat":
        return TransferFunction([1], [1])  # aproximação: 1
    if t == "sum":
        # não convertido automaticamente (depende de topologia)
        return TransferFunction([1], [1])
    return TransferFunction([1], [1])

def compose_series_from_diagram(diagram: dict):
    """
    Converte um diagrama (nodes/edges) em uma TF em série (cadeia única).
    Retorna (tf, ordem_ids, avisos[])
    """
    nodes = diagram.get("nodes") or []
    edges = diagram.get("edges") or []
    warnings = []

    if not nodes:
        return None, [], ["Diagrama vazio."]

    # grafo dirigido src -> dst
    out_map = {}
    in_count = {}
    node_by_id = {}

    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        node_by_id[nid] = n
        out_map[nid] = []
        in_count[nid] = 0

    for e in edges:
        s = e.get("src")
        d = e.get("dst")
        if s in out_map and d in in_count:
            out_map[s].append(d)
            in_count[d] += 1

    # acha possíveis "inícios": nós com indegree 0
    starts = [nid for nid, c in in_count.items() if c == 0]
    if len(starts) != 1:
        warnings.append(f"Esperado 1 início (entrada). Encontrei {len(starts)}. Vou tentar escolher o primeiro.")
    start = starts[0] if starts else list(node_by_id.keys())[0]

    # percorre cadeia, exigindo no máximo 1 saída por nó (cadeia simples)
    order = []
    seen = set()
    cur = start
    while cur and cur not in seen:
        seen.add(cur)
        order.append(cur)
        outs = out_map.get(cur, [])
        if len(outs) == 0:
            break
        if len(outs) > 1:
            warnings.append("Há ramificação (um nó com mais de uma saída). O conversor atual suporta apenas cadeia simples.")
            break
        nxt = outs[0]
        cur = nxt

    # valida se existe nó com indegree >1 (junções)
    if any(c > 1 for c in in_count.values()):
        warnings.append("Há junções (nó com múltiplas entradas). O conversor atual suporta apenas cadeia simples.")

    # compõe TF na ordem
    tf_total = None
    for nid in order:
        n = node_by_id.get(nid, {})
        if n.get("type") == "sum":
            warnings.append("Somador detectado: ainda não é convertido automaticamente.")
        tf_n = _tf_from_node(n)
        tf_total = tf_n if tf_total is None else (tf_total * tf_n)

    return tf_total, order, warnings

# =====================================================
# (D) APP
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Modelagem e Análise de Sistemas de Controle")

    inicializar_blocos()

    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False
    if 'mostrar_ajuda' not in st.session_state:
        st.session_state.mostrar_ajuda = False
    if 'diagram_model' not in st.session_state:
        st.session_state.diagram_model = {"nodes": [], "edges": []}

    with st.sidebar:
        st.header("🧱 Adicionar Blocos (modo clássico)")
        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("➕ Adicionar", key="btn_add_block"):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            st.success(mensagem) if sucesso else st.error(mensagem)

        if not st.session_state.blocos.empty:
            st.header("🗑️ Excluir Blocos")
            options = st.session_state.blocos.apply(lambda r: f"[{r['uid']}] {r['nome']}", axis=1).tolist()
            excluir = st.selectbox("Selecionar", options, key="sb_del_block")
            if st.button("❌ Excluir", key="btn_del_block"):
                uid = excluir.split(']')[0].replace('[','').strip()
                st.success(remover_bloco(uid))
                st.rerun()

        st.header("⚙️ Configurações")
        if st.button("🔢 Habilitar Cálculo de Erro" if not st.session_state.calculo_erro_habilitado else "❌ Desabilitar Cálculo de Erro",
                     key="btn_toggle_error"):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
            st.rerun()

    tab_sim, tab_editor = st.tabs(["📈 Simulação", "🧩 Editor Visual (nível Simulink)"])

    with tab_editor:
        st.caption("Monte o diagrama visualmente. **Para simular em Malha Aberta**, conecte em cadeia (entrada → ... → saída).")
        model = visual_blocks_editor(model=st.session_state.diagram_model, height=680, key="editor_simulink")
        st.session_state.diagram_model = model

        with st.expander("📦 Modelo (JSON)"):
            st.json(model)

        tf_diag, order_ids, warns = compose_series_from_diagram(model)
        if warns:
            st.warning("⚠️ " + " | ".join(warns))
        if tf_diag is not None:
            st.success("✅ Função equivalente (cadeia simples) pronta para Malha Aberta.")
            st.code(str(tf_diag))

            if st.button("➡️ Usar como Planta (modo clássico)", key="btn_use_as_planta"):
                # cria/atualiza um bloco Planta chamado "Planta_Diagrama"
                # remove plantas antigas com esse nome
                df = st.session_state.blocos
                if not df.empty:
                    st.session_state.blocos = df[df['nome'].astype(str) != "Planta_Diagrama"]
                uid = str(uuid.uuid4())
                novo = pd.DataFrame([{
                    'uid': uid,
                    'nome': "Planta_Diagrama",
                    'tipo': "Planta",
                    'numerador': "DIAGRAMA",
                    'denominador': "DIAGRAMA",
                    'tf': tf_diag,
                    'tf_simbolico': ("DIAGRAMA", "DIAGRAMA")
                }])
                st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
                st.success("✅ Planta_Diagrama adicionada/atualizada. Vá para a aba Simulação.")
        else:
            st.info("Crie e conecte blocos para gerar a função equivalente.")

    with tab_sim:
        if st.session_state.calculo_erro_habilitado:
            st.subheader("📊 Cálculo de Erro Estacionário")
            c1, c2 = st.columns(2)
            with c1:
                num_erro = st.text_input("Numerador", value="", key="num_erro")
            with c2:
                den_erro = st.text_input("Denominador", value="", key="den_erro")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔍 Calcular Erro Estacionário", key="btn_calc_ess"):
                    try:
                        G, _ = converter_para_tf(num_erro, den_erro)
                        tipo_sis, Kp, Kv, Ka = constantes_de_erro(G)
                        df_res = pd.DataFrame([{"Tipo": tipo_sis, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                        st.dataframe(
                            df_res.style.format({
                                "Kp": lambda x: formatar_numero(x),
                                "Kv": lambda x: formatar_numero(x),
                                "Ka": lambda x: formatar_numero(x)
                            }),
                            height=120,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Erro: {e}")
            with b2:
                if st.button("🗑️ Remover Plantas", key="btn_rm_plants"):
                    df = st.session_state.blocos
                    if df.empty:
                        st.warning("Nenhum bloco.")
                    else:
                        st.session_state.blocos = df[df['tipo'] != 'Planta']
                        st.success("Plantas removidas!")
        else:
            st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na barra lateral para ativar esta funcionalidade")

        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("🔍 Tipo de Sistema")
            tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"], key="sb_tipo_malha")
            usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False, key="cb_gain")

            if usar_ganho:
                K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1, key="sl_gain")
            else:
                K = 1.0

            st.subheader("📊 Análises desejadas")
            analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
            analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0], key="ms_analises")
            entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS, key="sb_input")

        with col1:
            st.subheader("📈 Resultados da Simulação")

            if st.button("▶️ Executar Simulação", use_container_width=True, key="btn_run"):
                try:
                    df = st.session_state.blocos
                    planta = obter_bloco_por_tipo('Planta')
                    controlador = obter_bloco_por_tipo('Controlador')
                    sensor = obter_bloco_por_tipo('Sensor')

                    # Se não houver planta no modo clássico, tenta usar diagrama
                    if planta is None:
                        tf_diag, _, warns = compose_series_from_diagram(st.session_state.diagram_model)
                        if tf_diag is not None:
                            planta = tf_diag
                            st.info("🧩 Usando a Planta do Editor Visual (diagrama) porque não há Planta no modo clássico.")
                            if warns:
                                st.warning("⚠️ " + " | ".join(warns))
                        else:
                            st.error("Adicione pelo menos uma Planta (modo clássico) OU conecte uma cadeia no Editor Visual.")
                            st.stop()

                    ganho_tf = TransferFunction([K], [1])

                    if tipo_malha == "Malha Aberta":
                        sistema = ganho_tf * planta
                        st.info(f"🔧 Sistema em Malha Aberta com K = {K:.2f}")
                    else:
                        sistema = calcular_malha_fechada(ganho_tf * planta, controlador, sensor)
                        st.info(f"🔧 Sistema em Malha Fechada com K = {K:.2f}")

                    for analise in analises:
                        st.markdown(f"### 🔎 {analise}")
                        if analise == 'Resposta no tempo':
                            fig = plot_resposta_temporal(sistema, entrada)
                            st.plotly_chart(fig, use_container_width=True)
                        elif analise == 'Desempenho':
                            des = calcular_desempenho(sistema)
                            for k, v in des.items():
                                st.markdown(f"**{k}:** {v}")
                        elif analise == 'Diagrama De Bode Magnitude':
                            st.plotly_chart(plot_bode(sistema, 'magnitude'), use_container_width=True)
                        elif analise == 'Diagrama De Bode Fase':
                            st.plotly_chart(plot_bode(sistema, 'fase'), use_container_width=True)
                        elif analise == 'Diagrama de Polos e Zeros':
                            st.plotly_chart(plot_polos_zeros(sistema), use_container_width=True)
                        elif analise == 'LGR':
                            st.plotly_chart(plot_lgr(sistema), use_container_width=True)
                        elif analise == 'Nyquist':
                            st.plotly_chart(plot_nyquist(sistema), use_container_width=True)

                except Exception as e:
                    st.error(f"Erro durante a simulação: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    st.sidebar.markdown("---")
    st.sidebar.info("🧩 Use a aba **Editor Visual** para montar o diagrama no estilo Simulink. Para simular automaticamente, conecte em cadeia (série).")

if __name__ == "__main__":
    main()
