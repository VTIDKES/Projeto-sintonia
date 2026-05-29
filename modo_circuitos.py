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
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0;font-family:Segoe UI,Arial,sans-serif}body{background:#f8f8f6;color:#2c2c2a}
#app{display:grid;grid-template-columns:150px minmax(0,1fr);height:740px;border:1px solid #c8c6be;border-radius:10px;overflow:hidden}
#palette{background:#f1efe8;padding:10px;overflow:auto;border-right:1px solid #c8c6be}
h4{font-size:12px;margin:10px 0 8px;color:#5f5e5a;text-transform:uppercase;letter-spacing:.04em}.item{width:100%;border:1px solid #c8c6be;border-left:4px solid #185fa5;border-radius:7px;background:#fff;margin:0 0 7px;padding:8px 6px;cursor:pointer;font-size:12px;font-weight:650;color:#444441}.item.mec{border-left-color:#0f6e56}.item.src{border-left-color:#854f0b}.item.node{border-left-color:#444441}
#stage{position:relative;background:#fafaf8;overflow:hidden}#cv{width:100%;height:100%;display:block}#hint{position:absolute;left:10px;bottom:8px;color:#888780;font-size:11px;pointer-events:none}#toolbar{position:absolute;right:10px;top:10px;display:flex;gap:6px;z-index:2;flex-wrap:wrap}button{border:1px solid #c8c6be;border-radius:6px;background:#fff;color:#444441;padding:6px 9px;font-size:12px;cursor:pointer}button:hover{border-color:#185fa5;color:#185fa5;background:#e6f1fb}
label{display:block;margin-top:8px;color:#5f5e5a;font-size:11px;font-weight:650}input{width:100%;border:1px solid #c8c6be;border-radius:6px;background:#fff;color:#2c2c2a;padding:6px 7px;font-size:12px}.empty{margin-top:16px;color:#888780;font-size:12px;line-height:1.45;text-align:center}.selected-title{font-size:13px;font-weight:750;margin:8px 0;color:#2c2c2a}
#prop{position:absolute;left:14px;top:14px;width:220px;z-index:3;display:none;background:#fff;border:1px solid #c8c6be;border-radius:8px;box-shadow:0 14px 32px rgba(0,0,0,.16);padding:10px}#prop.open{display:block}
#formula-panel{position:absolute;right:10px;bottom:28px;z-index:2;max-width:min(560px,calc(100% - 24px));background:rgba(255,255,255,.94);border:1px solid #c8c6be;border-radius:8px;box-shadow:0 10px 26px rgba(0,0,0,.12);padding:10px 12px;color:#2c2c2a}#formula-title{font-size:11px;font-weight:800;color:#5f5e5a;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}#formula-body{font-size:13px;line-height:1.45}.formula{font-family:Cambria Math,Georgia,serif;font-size:16px;color:#185fa5}
</style></head><body><div id="app"><aside id="palette"><h4>Eletricos</h4><button class="item" data-type="resistor">Resistor</button><button class="item" data-type="capacitor">Capacitor</button><button class="item" data-type="indutor">Indutor</button><h4>Fontes</h4><button class="item src" data-type="fonte_v">Fonte V</button><button class="item src" data-type="forca">Forca</button><h4>Conexoes</h4><button class="item node" data-type="junction">No</button><h4>Mecanicos</h4><button class="item mec" data-type="massa">Massa</button><button class="item mec" data-type="mola">Mola</button><button class="item mec" data-type="amortecedor">Amortecedor</button></aside><main id="stage"><div id="toolbar"><button id="arrange">Organizar</button><button id="undo">Desfazer fio</button><button id="clearWires">Limpar fios</button><button id="clear">Limpar</button></div><canvas id="cv"></canvas><div id="prop"></div><div id="formula-panel"><div id="formula-title">Funcao de transferencia detectada</div><div id="formula-body"></div></div><div id="hint">Clique em um terminal e depois em outro para criar fios. Use nos para paralelo e malhas fechadas.</div></main></div>
<script>
const initialElements=__INITIAL_ELEMENTS__;const types={resistor:{label:"Resistor",symbol:"R",unit:"ohm",value:1000,color:"#185fa5"},capacitor:{label:"Capacitor",symbol:"C",unit:"uF",value:10,color:"#185fa5"},indutor:{label:"Indutor",symbol:"L",unit:"mH",value:100,color:"#185fa5"},fonte_v:{label:"Fonte V",symbol:"Vs",unit:"V",value:5,color:"#854f0b"},forca:{label:"Forca",symbol:"F",unit:"N",value:1,color:"#854f0b"},massa:{label:"Massa",symbol:"m",unit:"kg",value:1,color:"#0f6e56"},mola:{label:"Mola",symbol:"k",unit:"N/m",value:10,color:"#0f6e56"},amortecedor:{label:"Amortecedor",symbol:"b",unit:"N.s/m",value:2,color:"#0f6e56"},junction:{label:"No de conexao",symbol:"No",unit:"",value:0,color:"#444441"}};
const canvas=document.getElementById("cv"),ctx=canvas.getContext("2d"),stage=document.getElementById("stage"),prop=document.getElementById("prop"),formulaBody=document.getElementById("formula-body");const W=94,H=64;let nextId=1,nextEdge=1,selected=null,drag=null,dx=0,dy=0,pending=null,edges=[];let elements=(Array.isArray(initialElements)?initialElements:[]).filter(e=>types[e.type]).map((e,i)=>({id:nextId++,type:e.type,value:+e.value||types[e.type].value,x:+e.x||160+i*130,y:+e.y||300,rotation:+e.rotation||0}));
function resize(){const r=stage.getBoundingClientRect();canvas.width=r.width;canvas.height=r.height;draw()}new ResizeObserver(resize).observe(stage);resize();function round(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath()}function find(id){return elements.find(e=>e.id===id)}function ports(e){return e.type==="junction"?["node"]:["left","right","top","bottom"]}function rot(x,y,d){const a=(d||0)*Math.PI/180;return{x:x*Math.cos(a)-y*Math.sin(a),y:x*Math.sin(a)+y*Math.cos(a)}}function point(e,p){if(e.type==="junction"||p==="node")return{x:e.x,y:e.y,vx:0,vy:0};const l={left:{x:-W/2,y:0,vx:-1,vy:0},right:{x:W/2,y:0,vx:1,vy:0},top:{x:0,y:-H/2,vx:0,vy:-1},bottom:{x:0,y:H/2,vx:0,vy:1}}[p],pos=rot(l.x,l.y,e.rotation),v=rot(l.vx,l.vy,e.rotation);return{x:e.x+pos.x,y:e.y+pos.y,vx:v.x,vy:v.y}}
function drawShape(e){const t=types[e.type];ctx.strokeStyle=t.color;ctx.fillStyle=t.color;ctx.lineWidth=2;ctx.lineCap="round";ctx.lineJoin="round";if(e.type==="resistor"){ctx.beginPath();ctx.moveTo(-38,0);[[-28,0],[-23,-11],[-15,11],[-7,-11],[1,11],[9,-11],[17,11],[25,0],[38,0]].forEach(([x,y])=>ctx.lineTo(x,y));ctx.stroke()}else if(e.type==="capacitor"){ctx.beginPath();ctx.moveTo(-38,0);ctx.lineTo(-8,0);ctx.stroke();ctx.beginPath();ctx.moveTo(-8,-16);ctx.lineTo(-8,16);ctx.lineWidth=3;ctx.stroke();ctx.beginPath();ctx.moveTo(8,-16);ctx.lineTo(8,16);ctx.stroke();ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(8,0);ctx.lineTo(38,0);ctx.stroke()}else if(e.type==="indutor"){ctx.beginPath();ctx.moveTo(-38,0);ctx.lineTo(-28,0);ctx.stroke();ctx.beginPath();ctx.moveTo(-28,0);for(let i=0;i<5;i++)ctx.arc(-23+i*10,0,5,Math.PI,0);ctx.lineTo(38,0);ctx.stroke()}else if(e.type==="fonte_v"){ctx.beginPath();ctx.moveTo(-42,0);ctx.lineTo(-18,0);ctx.stroke();ctx.beginPath();ctx.arc(0,0,18,0,Math.PI*2);ctx.fillStyle="#faeeda";ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(18,0);ctx.lineTo(42,0);ctx.stroke();ctx.fillStyle="#633806";ctx.font="10px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("AC",0,-4);ctx.fillText("Vs",0,8)}else if(e.type==="forca"){ctx.beginPath();ctx.moveTo(-42,0);ctx.lineTo(24,0);ctx.lineWidth=3;ctx.stroke();ctx.beginPath();ctx.moveTo(24,-10);ctx.lineTo(42,0);ctx.lineTo(24,10);ctx.closePath();ctx.fill()}else if(e.type==="massa"){ctx.fillStyle="#e1f5ee";ctx.strokeStyle=t.color;ctx.lineWidth=2;round(-28,-18,56,36,4);ctx.fill();ctx.stroke();ctx.fillStyle="#085041";ctx.font="bold 17px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("m",0,1)}else if(e.type==="mola"){ctx.beginPath();ctx.moveTo(-42,0);ctx.lineTo(-30,0);ctx.stroke();ctx.beginPath();ctx.moveTo(-30,0);for(let i=0;i<7;i++){const x=-25+i*8;ctx.bezierCurveTo(x-4,-13,x+4,-13,x+4,0);ctx.bezierCurveTo(x+4,13,x+12,13,x+8,0)}ctx.stroke();ctx.beginPath();ctx.moveTo(30,0);ctx.lineTo(42,0);ctx.stroke()}else if(e.type==="amortecedor"){ctx.beginPath();ctx.moveTo(-42,0);ctx.lineTo(-22,0);ctx.stroke();ctx.strokeRect(-22,-15,34,30);ctx.beginPath();ctx.moveTo(-4,0);ctx.lineTo(42,0);ctx.lineWidth=3;ctx.stroke();ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(12,-15);ctx.lineTo(12,15);ctx.stroke()}else if(e.type==="junction"){ctx.beginPath();ctx.arc(0,0,7,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(-20,0);ctx.lineTo(20,0);ctx.moveTo(0,-20);ctx.lineTo(0,20);ctx.stroke()}}
function symbol(e){const t=types[e.type];ctx.save();ctx.translate(e.x,e.y);if(e===selected){ctx.save();ctx.strokeStyle="rgba(17,24,39,.45)";ctx.setLineDash([5,4]);ctx.lineWidth=1.4;round(-52,-38,104,76,8);ctx.stroke();ctx.restore()}drawShape(e);if(e.type!=="junction"){ctx.strokeStyle=t.color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(-48,0);ctx.lineTo(-42,0);ctx.moveTo(42,0);ctx.lineTo(48,0);ctx.stroke()}ctx.restore();ctx.save();ports(e).forEach(p=>{const q=point(e,p);ctx.fillStyle=pending&&pending.id===e.id&&pending.port===p?"#d85a30":t.color;ctx.beginPath();ctx.arc(q.x,q.y,e.type==="junction"?6:4,0,Math.PI*2);ctx.fill()});ctx.restore();if(e.type!=="junction"){ctx.save();ctx.fillStyle=t.color;ctx.font="bold 12px Arial";ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(t.symbol,e.x,e.y+H/2+5);ctx.fillStyle="#5f5e5a";ctx.font="11px Arial";ctx.fillText(`${Number(e.value).toPrecision(4)} ${t.unit}`,e.x,e.y+H/2+20);ctx.restore()}}
function drawWire(a,b){const as={x:a.x+(a.vx||0)*20,y:a.y+(a.vy||0)*20},bs={x:b.x+(b.vx||0)*20,y:b.y+(b.vy||0)*20},mx=(a.x+b.x)/2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(as.x,as.y);ctx.lineTo(mx,as.y);ctx.lineTo(mx,bs.y);ctx.lineTo(bs.x,bs.y);ctx.lineTo(b.x,b.y);ctx.stroke()}function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="rgba(0,0,0,.05)";for(let x=0;x<canvas.width;x+=24){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}for(let y=0;y<canvas.height;y+=24){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}ctx.strokeStyle="#444441";ctx.lineWidth=2;edges.forEach(e=>{const a=find(e.from.id),b=find(e.to.id);if(a&&b)drawWire(point(a,e.from.port),point(b,e.to.port))});elements.forEach(symbol);if(pending){const e=find(pending.id);if(e){const p=point(e,pending.port);ctx.strokeStyle="#d85a30";ctx.fillStyle="#fff8ed";ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,9,0,Math.PI*2);ctx.fill();ctx.stroke()}}updateFormula()}
function hitPort(x,y){for(let i=elements.length-1;i>=0;i--){const e=elements[i];for(const p of ports(e)){const q=point(e,p);if(Math.hypot(x-q.x,y-q.y)<=11)return{id:e.id,port:p}}}return null}function hit(x,y){for(let i=elements.length-1;i>=0;i--){const e=elements[i];if(e.type==="junction"){if(Math.hypot(x-e.x,y-e.y)<=18)return e}else if(x>=e.x-W/2&&x<=e.x+W/2&&y>=e.y-H/2&&y<=e.y+H/2)return e}return null}
function valuesByType(){const by={};elements.forEach(e=>{if(e.type!=="junction"&&by[e.type]===undefined)by[e.type]=Number(e.value)});return by}function fmt(n){if(!Number.isFinite(n))return "-";if(Math.abs(n)>=1e4||Math.abs(n)<1e-3)return n.toExponential(3);return Number(n.toPrecision(5)).toString()}function updateFormula(){const by=valuesByType();let title="Monte um arranjo livre no canvas.",formula="";if(by.resistor&&by.capacitor&&by.indutor){const R=by.resistor,L=by.indutor*1e-3,C=by.capacitor*1e-6,Vs=by.fonte_v||1,wn=1/Math.sqrt(L*C),z=R/2*Math.sqrt(C/L);title="Modelo RLC detectado";formula=`<span class="formula">G(s) = ${fmt(Vs*wn*wn)} / (s^2 + ${fmt(2*z*wn)}s + ${fmt(wn*wn)})</span>`}else if(by.resistor&&by.capacitor){const R=by.resistor,C=by.capacitor*1e-6,Vs=by.fonte_v||1,t=R*C;title="Modelo RC detectado";formula=`<span class="formula">G(s) = ${fmt(Vs)} / (${fmt(t)}s + 1)</span>`}else if(by.massa&&by.mola&&by.amortecedor!==undefined){const m=by.massa,k=by.mola,b=by.amortecedor,F=by.forca||1;title="Modelo massa-mola-amortecedor detectado";formula=`<span class="formula">G(s) = ${fmt(F/m)} / (s^2 + ${fmt(b/m)}s + ${fmt(k/m)})</span>`}formulaBody.innerHTML=`<strong>${title}</strong>${formula?`<br>${formula}`:"<br>Use os elementos e nos para serie, paralelo e malha fechada visual."}`}
function renderProp(){if(!selected){prop.classList.remove("open");prop.innerHTML="";return}const t=types[selected.type],valueField=selected.type==="junction"?"":`<label>Valor (${t.unit})</label><input id="val" type="number" step="any" value="${selected.value}">`;prop.classList.add("open");prop.innerHTML=`<div class="selected-title">${t.label}</div><div class="empty" style="margin-top:6px;text-align:left">Clique em um terminal e depois em outro para criar conexao.</div>${valueField}<label>X</label><input id="x" type="number" value="${Math.round(selected.x)}"><label>Y</label><input id="y" type="number" value="${Math.round(selected.y)}"><button id="del" style="margin-top:10px;width:100%">Remover</button>`;const val=document.getElementById("val");if(val)val.oninput=e=>{const n=Number(e.target.value);if(Number.isFinite(n))selected.value=n;draw()};document.getElementById("x").oninput=e=>{selected.x=+e.target.value||selected.x;draw()};document.getElementById("y").oninput=e=>{selected.y=+e.target.value||selected.y;draw()};document.getElementById("del").onclick=()=>{elements=elements.filter(e=>e!==selected);edges=edges.filter(e=>e.from.id!==selected.id&&e.to.id!==selected.id);selected=null;pending=null;renderProp();draw()}}
document.querySelectorAll(".item").forEach(b=>b.onclick=()=>{const n=elements.length,e={id:nextId++,type:b.dataset.type,value:types[b.dataset.type].value,x:150+n*100,y:canvas.height/2,rotation:0};elements.push(e);selected=e;renderProp();draw()});canvas.onmousedown=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,p=hitPort(x,y);if(p){if(pending){if(!(pending.id===p.id&&pending.port===p.port))edges.push({id:nextEdge++,from:pending,to:p});pending=null}else pending=p;selected=find(p.id);drag=null;renderProp();draw();return}selected=hit(x,y);if(selected){drag=selected;dx=x-selected.x;dy=y-selected.y}else drag=null;renderProp();draw()};canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drag){drag.x=Math.round((x-dx)/10)*10;drag.y=Math.round((y-dy)/10)*10;draw()}else canvas.style.cursor=hitPort(x,y)?"crosshair":hit(x,y)?"grab":"default"};window.onmouseup=()=>drag=null;document.getElementById("arrange").onclick=()=>{elements.forEach((e,i)=>{e.x=130+i*125;e.y=canvas.height/2});draw()};document.getElementById("undo").onclick=()=>{edges.pop();pending=null;draw()};document.getElementById("clearWires").onclick=()=>{edges=[];pending=null;draw()};document.getElementById("clear").onclick=()=>{elements=[];edges=[];pending=null;selected=null;renderProp();draw()};renderProp();draw();
</script></body></html>
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


def _format_latex_number(value):
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
        return str(int(round(value)))
    return f"{value:.4g}"


def _poly_to_latex(coeffs):
    coeffs = [float(c) for c in coeffs]
    degree = len(coeffs) - 1
    terms = []
    for index, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-12:
            continue

        power = degree - index
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        coeff_text = _format_latex_number(abs_coeff)

        if power == 0:
            body = coeff_text
        elif power == 1:
            body = "s" if abs(abs_coeff - 1) < 1e-12 else f"{coeff_text}s"
        else:
            body = f"s^{{{power}}}" if abs(abs_coeff - 1) < 1e-12 else f"{coeff_text}s^{{{power}}}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    result = f"-{first_body}" if first_sign == "-" else first_body
    for sign, body in terms[1:]:
        result += f" {sign} {body}"
    return result


def _tf_to_latex(tf_obj):
    num = _poly_to_latex(tf_obj.num[0][0])
    den = _poly_to_latex(tf_obj.den[0][0])
    return rf"G(s)=\frac{{{num}}}{{{den}}}"


def _build_elements(system, values):
    templates = PRESETS[system]
    result = []
    for item in templates:
        element = dict(item)
        if element["type"] in values:
            element["value"] = values[element["type"]]
        result.append(element)
    return result


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
    with st.sidebar:
        st.header("Navegação")
        if st.button("← Voltar à Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.caption("O canvas abaixo é livre: arraste elementos, mova, gire e conecte pelos terminais.")

    render_guia_janela("Guia")

    st.title("Modo Simulação com Elementos")
    st.caption(
        "Desenhe circuitos ou sistemas massa-mola-amortecedor do zero. "
        "Use nós para série, paralelo e malhas fechadas."
    )

    html_template = _load_circuit_editor_html()
    html_content = html_template.replace(
        "__INITIAL_ELEMENTS__",
        json.dumps([], ensure_ascii=False),
    )
    components.html(html_content, height=790, scrolling=False)
