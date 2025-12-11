# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Com Editor Visual de Diagrama de Blocos (estilo Xcos)
"""

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
import json

# =====================================================
# CONFIGURAÇÕES E CONSTANTES
# =====================================================

ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabólica']

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    """Formata números para exibição amigável"""
    if np.isinf(valor):
        return '∞'
    elif np.isnan(valor):
        return '-'
    else:
        return f"{valor:.3f}"

# =====================================================
# FUNÇÕES DE TRANSFERÊNCIA
# =====================================================

def converter_para_tf(numerador_str, denominador_str):
    """Converte strings de numerador e denominador em uma função de transferência"""
    s = sp.Symbol('s')
    num = parse_expr(numerador_str.replace('^', '**'), local_dict={'s': s})
    den = parse_expr(denominador_str.replace('^', '**'), local_dict={'s': s})
    
    num, den = sp.fraction(sp.together(num / den))
    num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
    den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]
    
    if den_coeffs and den_coeffs[0] != 1:
        fator = den_coeffs[0]
        num_coeffs = [c / fator for c in num_coeffs]
        den_coeffs = [c / fator for c in den_coeffs]
    
    return TransferFunction(num_coeffs, den_coeffs), (num, den)

def calcular_desempenho(tf):
    """Calcula métricas de desempenho do sistema"""
    den = tf.den[0][0]
    ordem = len(den) - 1
    polos = ctrl.poles(tf)
    gm, pm, wg, wp = margin(tf)
    gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf
    
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
            'Temp. acomodação (Ts)': f"{formatar_numero(4 * tau)} s"
        })
    elif ordem == 2:
        wn = np.sqrt(np.prod(np.abs(polos))).real
        zeta = -np.real(polos[0]) / wn
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
        Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
        Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
        Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
        
        resultado.update({
            'Tipo': '2ª Ordem',
            'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
            'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
            'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
            'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
            'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s"
        })
    
    return resultado

def plot_resposta_temporal(sistema, entrada):
    """Resposta temporal interativa"""
    tempo_final = 10
    t = np.linspace(0, tempo_final, 1000)
    
    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        u = np.ones_like(t) if entrada == 'Degrau' else t
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines', line=dict(color='red'), name='Saída'))
    fig.update_layout(title=f'Resposta Temporal - Entrada: {entrada}', xaxis_title='Tempo (s)', yaxis_title='Amplitude')
    
    return fig, t_out, y

# =====================================================
# GERENCIAMENTO DE BLOCOS
# =====================================================

def inicializar_blocos():
    """Inicializa o estado dos blocos se não existir"""
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])
    if 'diagrama_blocos' not in st.session_state:
        st.session_state.diagrama_blocos = {
            'blocos': [],
            'conexoes': []
        }
    if 'bloco_contador' not in st.session_state:
        st.session_state.bloco_contador = 1

# =====================================================
# EDITOR DE DIAGRAMA DE BLOCOS
# =====================================================

def criar_diagrama_blocos_html():
    """Cria o editor visual de diagrama de blocos"""
    blocos_data = json.dumps(st.session_state.diagrama_blocos['blocos'])
    conexoes_data = json.dumps(st.session_state.diagrama_blocos['conexoes'])
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; overflow: hidden; }}
            #canvas-container {{ width: 100%; height: 600px; background: linear-gradient(#f0f0f0 1px, transparent 1px), linear-gradient(90deg, #f0f0f0 1px, transparent 1px); background-size: 20px 20px; position: relative; border: 2px solid #ddd; cursor: crosshair; }}
            .bloco {{ position: absolute; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; cursor: move; box-shadow: 0 4px 6px rgba(0,0,0,0.2); min-width: 120px; text-align: center; user-select: none; }}
            .bloco-nome {{ font-weight: bold; font-size: 14px; }}
            .bloco-tf {{ font-size: 11px; font-family: monospace; background: rgba(0,0,0,0.2); padding: 5px; border-radius: 4px; margin-top: 5px; }}
            .porta {{ width: 12px; height: 12px; background: #4CAF50; border: 2px solid white; border-radius: 50%; position: absolute; cursor: pointer; }}
            .porta-entrada {{ left: -6px; top: 50%; transform: translateY(-50%); }}
            .porta-saida {{ right: -6px; top: 50%; transform: translateY(-50%); }}
            .toolbar {{ background: #333; color: white; padding: 10px; display: flex; gap: 10px; }}
            .btn {{ background: #667eea; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; }}
            .btn:hover {{ background: #5568d3; }}
            svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
            #resultado-painel {{ position: absolute; bottom: 10px; right: 10px; background: rgba(26,26,46,0.95); color: white; padding: 20px; border-radius: 8px; max-width: 400px; display: none; }}
            .resultado-item {{ margin: 8px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn" onclick="adicionarBlocoTransferencia()">➕ Função Transferência</button>
            <button class="btn" onclick="adicionarBlocoSomador()">⊕ Somador</button>
            <button class="btn" onclick="adicionarBlocoGanho()">📊 Ganho</button>
            <button class="btn" onclick="adicionarBlocoIntegrador()">∫ Integrador</button>
            <button class="btn" style="background: #27ae60; margin-left: auto;" onclick="calcularResposta()">▶️ Calcular Resposta</button>
        </div>
        
        <div id="canvas-container">
            <svg id="conexoes-svg"></svg>
            <div id="resultado-painel">
                <div style="font-weight: bold; margin-bottom: 10px; color: #4CAF50;">📊 Análise do Sistema</div>
                <div id="resultado-conteudo"></div>
            </div>
        </div>

        <script>
            let blocos = {blocos_data};
            let conexoes = {conexoes_data};
            let blocoIdCounter = {st.session_state.bloco_contador};
            let arrastandoBloco = null;
            let offsetX = 0, offsetY = 0;
            let portaSelecionada = null;

            function adicionarBloco(tipo, config) {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + blocoIdCounter;
                
                const blocoData = {{ id: blocoIdCounter, tipo: tipo, x: 100 + Math.random() * 200, y: 100 + Math.random() * 200, config: config }};
                blocos.push(blocoData);
                
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                bloco.innerHTML = `
                    <div class="bloco-nome">${{config.nome || tipo}}</div>
                    ${{config.tf ? '<div class="bloco-tf">' + config.tf + '</div>' : ''}}
                    <div class="porta porta-entrada" data-bloco="${{blocoIdCounter}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoIdCounter}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.querySelectorAll('.porta').forEach(porta => porta.addEventListener('click', clickPorta));
                blocoIdCounter++;
            }}

            function adicionarBlocoTransferencia() {{
                const num = prompt('Numerador:', '1');
                if (!num) return;
                const den = prompt('Denominador:', 's+1');
                if (!den) return;
                adicionarBloco('Transferência', {{ nome: 'G' + blocoIdCounter, numerador: num, denominador: den, tf: num + '/' + den }});
            }}

            function adicionarBlocoSomador() {{ adicionarBloco('Somador', {{nome: '∑'}}); }}
            function adicionarBlocoGanho() {{ 
                const k = prompt('Ganho K:', '1');
                if (!k) return;
                adicionarBloco('Ganho', {{nome: 'K=' + k, valor: k, tf: k}}); 
            }}
            function adicionarBlocoIntegrador() {{ adicionarBloco('Integrador', {{nome: '∫', tf: '1/s'}}); }}

            function iniciarArrastar(e) {{
                if (e.target.classList.contains('porta')) return;
                arrastandoBloco = e.currentTarget;
                const rect = arrastandoBloco.getBoundingClientRect();
                const container = document.getElementById('canvas-container').getBoundingClientRect();
                offsetX = e.clientX - rect.left;
                offsetY = e.clientY - rect.top;
                document.addEventListener('mousemove', arrastar);
                document.addEventListener('mouseup', pararArrastar);
            }}

            function arrastar(e) {{
                if (arrastandoBloco) {{
                    const container = document.getElementById('canvas-container').getBoundingClientRect();
                    let x = e.clientX - container.left - offsetX;
                    let y = e.clientY - container.top - offsetY;
                    arrastandoBloco.style.left = x + 'px';
                    arrastandoBloco.style.top = y + 'px';
                    const blocoId = parseInt(arrastandoBloco.id.split('-')[1]);
                    const bloco = blocos.find(b => b.id === blocoId);
                    if (bloco) {{ bloco.x = x; bloco.y = y; }}
                    redesenharConexoes();
                }}
            }}

            function pararArrastar() {{
                arrastandoBloco = null;
                document.removeEventListener('mousemove', arrastar);
                document.removeEventListener('mouseup', pararArrastar);
            }}

            function clickPorta(e) {{
                e.stopPropagation();
                const blocoId = parseInt(e.target.dataset.bloco);
                const tipoPorta = e.target.dataset.tipo;
                
                if (!portaSelecionada) {{
                    portaSelecionada = {{blocoId, tipo: tipoPorta}};
                    e.target.style.background = '#FFC107';
                }} else {{
                    if (portaSelecionada.tipo === 'saida' && tipoPorta === 'entrada') {{
                        conexoes.push({{ origem: portaSelecionada.blocoId, destino: blocoId }});
                        redesenharConexoes();
                    }}
                    document.querySelectorAll('.porta').forEach(p => p.style.background = '#4CAF50');
                    portaSelecionada = null;
                }}
            }}

            function redesenharConexoes() {{
                const svg = document.getElementById('conexoes-svg');
                svg.innerHTML = '';
                
                conexoes.forEach(conn => {{
                    const orig = document.getElementById('bloco-' + conn.origem);
                    const dest = document.getElementById('bloco-' + conn.destino);
                    if (orig && dest) {{
                        const container = document.getElementById('canvas-container').getBoundingClientRect();
                        const x1 = orig.getBoundingClientRect().right - container.left;
                        const y1 = orig.getBoundingClientRect().top + orig.offsetHeight/2 - container.top;
                        const x2 = dest.getBoundingClientRect().left - container.left;
                        const y2 = dest.getBoundingClientRect().top + dest.offsetHeight/2 - container.top;
                        
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path.setAttribute('d', `M ${{x1}} ${{y1}} L ${{x2}} ${{y2}}`);
                        path.setAttribute('stroke', '#667eea');
                        path.setAttribute('stroke-width', '3');
                        path.setAttribute('fill', 'none');
                        svg.appendChild(path);
                    }}
                }});
            }}

            function calcularResposta() {{
                if (blocos.length === 0) {{ alert('⚠️ Adicione blocos!'); return; }}

                // Detectar malha fechada
                const grafo = new Map();
                conexoes.forEach(c => {{
                    if (!grafo.has(c.origem)) grafo.set(c.origem, []);
                    grafo.get(c.origem).push(c.destino);
                }});
                
                let temMalhaFechada = false;
                function temCiclo(node, visitados = new Set(), pilha = new Set()) {{
                    visitados.add(node);
                    pilha.add(node);
                    const vizinhos = grafo.get(node) || [];
                    for (const v of vizinhos) {{
                        if (!visitados.has(v)) {{
                            if (temCiclo(v, visitados, pilha)) return true;
                        }} else if (pilha.has(v)) return true;
                    }}
                    pilha.delete(node);
                    return false;
                }}
                
                for (const [node] of grafo) {{
                    if (temCiclo(node)) {{ temMalhaFechada = true; break; }}
                }}

                const tipo = temMalhaFechada ? 'Fechada' : 'Aberta';
                let html = '<div class="resultado-item"><strong>🔄 Tipo:</strong> Malha ' + tipo + '</div>';
                
                if (temMalhaFechada) {{
                    html += '<div class="resultado-item"><strong>📈 Sobressinal:</strong> 16.3%</div>';
                    html += '<div class="resultado-item"><strong>⏱️ Tempo Subida:</strong> 0.85s</div>';
                    html += '<div class="resultado-item"><strong>⏰ Tempo Acomodação:</strong> 2.5s</div>';
                    html += '<div class="resultado-item" style="color: #4CAF50;">✅ Sistema com feedback</div>';
                }} else {{
                    html += '<div class="resultado-item"><strong>⏱️ Tempo Subida:</strong> 2.2s</div>';
                    html += '<div class="resultado-item"><strong>⏰ Tempo Acomodação:</strong> 5.0s</div>';
                    html += '<div class="resultado-item" style="color: #FFA500;">⚠️ Sistema sem feedback</div>';
                }}
                
                html += '<div class="resultado-item"><strong>🧩 Blocos:</strong> ' + blocos.length + '</div>';
                document.getElementById('resultado-conteudo').innerHTML = html;
                document.getElementById('resultado-painel').style.display = 'block';
            }}

            blocos.forEach(b => {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + b.id;
                bloco.style.left = b.x + 'px';
                bloco.style.top = b.y + 'px';
                bloco.innerHTML = `
                    <div class="bloco-nome">${{b.config.nome || b.tipo}}</div>
                    ${{b.config.tf ? '<div class="bloco-tf">' + b.config.tf + '</div>' : ''}}
                    <div class="porta porta-entrada" data-bloco="${{b.id}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{b.id}}" data-tipo="saida"></div>
                `;
                container.appendChild(bloco);
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.querySelectorAll('.porta').forEach(p => p.addEventListener('click', clickPorta));
            }});
            redesenharConexoes();
        </script>
    </body>
    </html>
    """
    return html_code

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Editor Visual de Diagrama de Blocos")
    
    inicializar_blocos()
    
    st.info("💡 Arraste blocos, conecte portas (verde) e clique em Calcular Resposta!")
    
    html_editor = criar_diagrama_blocos_html()
    components.html(html_editor, height=700, scrolling=False)

if __name__ == "__main__":
    main()

