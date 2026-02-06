# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Com Editor Visual de Diagrama de Blocos (estilo Xcos/Simulink)
Versão Aprimorada
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

def tipo_do_sistema(G):
    """Determina o tipo do sistema (número de integradores)"""
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    tipo = sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))
    return tipo

def constantes_de_erro(G):
    """Calcula as constantes de erro (Kp, Kv, Ka)"""
    s = ctrl.tf('s')
    G_min = ctrl.minreal(G, verbose=False)
    tipo = tipo_do_sistema(G_min)
    
    Kp = Kv = Ka = np.inf
    try:
        if tipo == 0:
            Kp = ctrl.dcgain(G_min)
        elif tipo == 1:
            Kv = ctrl.dcgain(s * G_min)
        elif tipo >= 2:
            Ka = ctrl.dcgain(s**2 * G_min)
    except Exception:
        pass
    
    if tipo == 0:
        Kv = Ka = np.inf
    elif tipo == 1:
        Kp = 0
        Ka = np.inf
    elif tipo >= 2:
        Kp = Kv = 0
    
    return tipo, Kp, Kv, Ka

def calcular_malha_fechada(planta, controlador=None, sensor=None):
    """Calcula a função de transferência de malha fechada"""
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])
    
    G = controlador * planta
    H = sensor
    return ctrl.feedback(G, H)

# =====================================================
# ANÁLISE DE SISTEMAS
# =====================================================

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
        return _desempenho_ordem1(polos, resultado)
    elif ordem == 2:
        return _desempenho_ordem2(polos, resultado)
    elif ordem >= 3:
        return _desempenho_ordem_superior(polos, ordem, resultado)

def _desempenho_ordem1(polos, resultado):
    """Calcula desempenho para sistemas de 1ª ordem"""
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

def _desempenho_ordem2(polos, resultado):
    """Calcula desempenho para sistemas de 2ª ordem"""
    wn = np.sqrt(np.prod(np.abs(polos))).real
    zeta = -np.real(polos[0]) / wn
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
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

def _desempenho_ordem_superior(polos, ordem, resultado):
    """Calcula desempenho para sistemas de ordem superior"""
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
    polo_dominante = None
    par_dominante = None
    
    for i in range(len(polos_ordenados) - 1):
        p1, p2 = polos_ordenados[i], polos_ordenados[i+1]
        if np.isclose(p1.real, p2.real, atol=1e-2) and np.isclose(p1.imag, -p2.imag, atol=1e-2):
            par_dominante = (p1, p2)
            break
    
    if par_dominante:
        sigma = -np.real(par_dominante[0])
        omega_d = np.abs(np.imag(par_dominante[0]))
        wn = np.sqrt(sigma**2 + omega_d**2)
        zeta = sigma / wn if wn > 0 else 0
    else:
        polo_dominante = polos_ordenados[0]
        wn = np.abs(polo_dominante)
        zeta = -np.real(polo_dominante) / wn if wn != 0 else 0
        omega_d = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Tr = (np.pi - np.arccos(zeta)) / omega_d if zeta < 1 and omega_d > 0 else float('inf')
    Tp = np.pi / omega_d if omega_d > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
    
    resultado.update({
        'Tipo': f'{ordem}ª Ordem (Par dominante)' if par_dominante else f'{ordem}ª Ordem (Polo dominante)',
        'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (ωd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s",
        'Observação': 'Cálculo baseado no par dominante' if par_dominante else 'Cálculo baseado no polo dominante'
    })
    return resultado

def estimar_tempo_final_simulacao(tf):
    """Estima o tempo final para simulação baseado nos polos do sistema"""
    polos = ctrl.poles(tf)
    if len(polos) == 0:
        return 50.0
    if any(np.real(p) > 1e-6 for p in polos):
        return 20.0
    partes_reais_estaveis = [np.real(p) for p in polos if np.real(p) < -1e-6]
    if not partes_reais_estaveis:
        return 100.0
    sigma_dominante = max(partes_reais_estaveis)
    ts_estimado = 4 / abs(sigma_dominante)
    tempo_final = ts_estimado * 1.5
    return np.clip(tempo_final, a_min=10, a_max=500)

# =====================================================
# FUNÇÕES DE PLOTAGEM
# =====================================================

def configurar_linhas_interativas(fig):
    """Adiciona suporte para desenhar linhas horizontais e verticais em gráficos"""
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(
            line=dict(color='green', width=2, dash='dash')
        ),
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    )
    return fig

def plot_polos_zeros(tf, fig=None):
    """Diagrama de Polos e Zeros interativo"""
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)
    
    if fig is None:
        fig = go.Figure()
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros),
            y=np.imag(zeros),
            mode='markers',
            marker=dict(symbol='circle', size=12, color='blue'),
            name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos),
            y=np.imag(polos),
            mode='markers',
            marker=dict(symbol='x', size=12, color='red'),
            name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig.update_layout(
        title='Diagrama de Polos e Zeros (Interativo)',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )
    
    fig = configurar_linhas_interativas(fig)
    return fig

def _gerar_sinal_entrada(entrada, t):
    """Gera sinais de entrada para simulação"""
    sinais = {
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2*np.pi*t),
        'Impulso': np.concatenate([[1], np.zeros(len(t)-1)]),
        'Parabólica': t**2
    }
    return sinais[entrada]

def plot_resposta_temporal(sistema, entrada):
    """Resposta temporal interativa"""
    tempo_final = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tempo_final, 1000)
    u = _gerar_sinal_entrada(entrada, t)
    
    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_out,
        y=u[:len(t_out)],
        mode='lines',
        line=dict(dash='dash', color='blue'),
        name='Entrada',
        hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=t_out,
        y=y,
        mode='lines',
        line=dict(color='red'),
        name='Saída',
        hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        showlegend=True,
        hovermode='x unified'
    )
    
    fig = configurar_linhas_interativas(fig)
    return fig, t_out, y

def plot_bode(sistema, tipo='both'):
    """Diagrama de Bode interativo"""
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)
    
    if tipo == 'both':
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Diagrama de Bode - Magnitude', 'Diagrama de Bode - Fase'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=w, y=mag,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Magnitude',
                hovertemplate='Freq: %{x:.2f} rad/s<br>Magnitude: %{y:.2f} dB<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=w, y=phase,
                mode='lines',
                line=dict(color='red', width=3),
                name='Fase',
                hovertemplate='Freq: %{x:.2f} rad/s<br>Fase: %{y:.2f}°<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)
        
        fig.update_layout(height=700, title_text="Diagrama de Bode")
        
    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=mag,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Magnitude',
            hovertemplate='Freq: %{x:.2f} rad/s<br>Magnitude: %{y:.2f} dB<extra></extra>'
        ))
        fig.update_layout(
            title='Diagrama de Bode - Magnitude',
            xaxis_title="Frequência (rad/s)",
            yaxis_title="Magnitude (dB)",
            xaxis_type='log'
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=phase,
            mode='lines',
            line=dict(color='red', width=3),
            name='Fase',
            hovertemplate='Freq: %{x:.2f} rad/s<br>Fase: %{y:.2f}°<extra></extra>'
        ))
        fig.update_layout(
            title='Diagrama de Bode - Fase',
            xaxis_title="Frequência (rad/s)",
            yaxis_title="Fase (deg)",
            xaxis_type='log'
        )
    
    fig = configurar_linhas_interativas(fig)
    return fig

def plot_lgr(sistema):
    """Lugar Geométrico das Raízes interativo"""
    rlist, klist = root_locus(sistema, plot=False)
    
    fig = go.Figure()
    
    for i, r in enumerate(rlist.T):
        fig.add_trace(go.Scatter(
            x=np.real(r),
            y=np.imag(r),
            mode='lines',
            line=dict(color='blue', width=1),
            name=f'Ramo {i+1}',
            showlegend=False,
            hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))
    
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros),
            y=np.imag(zeros),
            mode='markers',
            marker=dict(symbol='circle', size=10, color='green'),
            name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos),
            y=np.imag(polos),
            mode='markers',
            marker=dict(symbol='x', size=12, color='red'),
            name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig.update_layout(
        title='Lugar Geométrico das Raízes (LGR)',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )
    
    fig = configurar_linhas_interativas(fig)
    return fig

def plot_nyquist(sistema):
    """Diagrama de Nyquist interativo"""
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=H.real,
        y=H.imag,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Nyquist',
        hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=H.real,
        y=-H.imag,
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Reflexo simétrico',
        hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1],
        y=[0],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto crítico (-1,0)',
        hovertemplate='Ponto Crítico<br>Real: -1<br>Imaginário: 0<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title='Diagrama de Nyquist',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )
    
    fig = configurar_linhas_interativas(fig)
    
    polos = ctrl.poles(sistema)
    polos_spd = sum(1 for p in polos if np.real(p) > 0)
    voltas = 0
    Z = polos_spd + voltas
    
    return fig, polos_spd, voltas, Z

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

def adicionar_bloco(nome, tipo, numerador, denominador):
    """Adiciona um novo bloco ao sistema"""
    try:
        tf, tf_symb = converter_para_tf(numerador, denominador)
        novo = pd.DataFrame([{
            'nome': nome,
            'tipo': tipo,
            'numerador': numerador,
            'denominador': denominador,
            'tf': tf,
            'tf_simbolico': tf_symb
        }])
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco {nome} adicionado."
    except Exception as e:
        return False, f"Erro na conversão: {e}"

def remover_bloco(nome):
    """Remove um bloco pelo nome"""
    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['nome'] != nome]
    return f"Bloco {nome} excluído."

def obter_bloco_por_tipo(tipo):
    """Obtém o primeiro bloco de um tipo específico"""
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None

# =====================================================
# EDITOR DE DIAGRAMA DE BLOCOS - ESTILO XCOS/SIMULINK
# =====================================================

def criar_diagrama_blocos_html():
    """Cria o editor visual de diagrama de blocos estilo Xcos/Simulink"""
    blocos_data = json.dumps(st.session_state.diagrama_blocos['blocos'])
    conexoes_data = json.dumps(st.session_state.diagrama_blocos['conexoes'])
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
                background: #2c3e50;
            }}
            
            /* ===== BARRA DE FERRAMENTAS ===== */
            .toolbar {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 12px 15px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                align-items: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                border-bottom: 2px solid #3498db;
            }}
            
            .toolbar-section {{
                display: flex;
                gap: 8px;
                align-items: center;
                padding: 0 10px;
                border-right: 1px solid rgba(255,255,255,0.2);
            }}
            
            .toolbar-section:last-child {{
                border-right: none;
            }}
            
            .toolbar-label {{
                font-size: 11px;
                opacity: 0.8;
                margin-right: 5px;
            }}
            
            .btn {{
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            
            .btn:hover {{
                background: linear-gradient(135deg, #2980b9, #3498db);
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
            
            .btn:active {{
                transform: translateY(0);
            }}
            
            .btn-success {{
                background: linear-gradient(135deg, #27ae60, #229954);
            }}
            
            .btn-success:hover {{
                background: linear-gradient(135deg, #229954, #27ae60);
            }}
            
            .btn-danger {{
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }}
            
            .btn-danger:hover {{
                background: linear-gradient(135deg, #c0392b, #e74c3c);
            }}
            
            .btn-warning {{
                background: linear-gradient(135deg, #f39c12, #e67e22);
            }}
            
            .btn-warning:hover {{
                background: linear-gradient(135deg, #e67e22, #f39c12);
            }}
            
            /* ===== ÁREA DE TRABALHO ===== */
            #canvas-container {{
                width: 100%;
                height: calc(100vh - 120px);
                background: 
                    linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px),
                    linear-gradient(rgba(255,255,255,0.02) 2px, transparent 2px),
                    linear-gradient(90deg, rgba(255,255,255,0.02) 2px, transparent 2px);
                background-size: 20px 20px, 20px 20px, 100px 100px, 100px 100px;
                background-position: 0 0, 0 0, 0 0, 0 0;
                position: relative;
                cursor: default;
                overflow: hidden;
            }}
            
            /* ===== ESTILOS DE BLOCOS ===== */
            .bloco {{
                position: absolute;
                background: white;
                border: 2px solid #34495e;
                border-radius: 6px;
                cursor: move;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                min-width: 140px;
                user-select: none;
                transition: all 0.2s ease;
                display: flex;
                flex-direction: column;
            }}
            
            .bloco:hover {{
                box-shadow: 0 6px 16px rgba(0,0,0,0.4);
                transform: translateY(-2px);
            }}
            
            .bloco.selecionado {{
                border: 3px solid #3498db;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3), 0 6px 16px rgba(0,0,0,0.4);
                z-index: 1000;
            }}
            
            .bloco-header {{
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                color: white;
                padding: 6px 10px;
                border-radius: 4px 4px 0 0;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .bloco-body {{
                padding: 12px;
                background: white;
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            
            .bloco-nome {{
                font-weight: 700;
                font-size: 16px;
                text-align: center;
                color: #2c3e50;
                margin-bottom: 8px;
            }}
            
            .bloco-tf {{
                font-size: 11px;
                font-family: 'Courier New', monospace;
                background: #ecf0f1;
                padding: 8px;
                border-radius: 4px;
                text-align: center;
                color: #34495e;
                border: 1px solid #bdc3c7;
                word-break: break-all;
            }}
            
            /* Cores específicas por tipo de bloco */
            .bloco-transferencia .bloco-header {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            }}
            
            .bloco-somador .bloco-header {{
                background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
            }}
            
            .bloco-ganho .bloco-header {{
                background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
            }}
            
            .bloco-integrador .bloco-header {{
                background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            }}
            
            .bloco-derivador .bloco-header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }}
            
            .bloco-atraso .bloco-header {{
                background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            }}
            
            /* ===== PORTAS DE CONEXÃO ===== */
            .porta {{
                width: 14px;
                height: 14px;
                background: #2ecc71;
                border: 3px solid white;
                border-radius: 50%;
                position: absolute;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                z-index: 10;
            }}
            
            .porta:hover {{
                background: #27ae60;
                transform: scale(1.4);
                box-shadow: 0 3px 10px rgba(46, 204, 113, 0.6);
            }}
            
            .porta-ativa {{
                background: #f39c12;
                animation: pulse 0.6s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.3); }}
            }}
            
            .porta-entrada {{
                left: -7px;
                top: 50%;
                transform: translateY(-50%);
            }}
            
            .porta-saida {{
                right: -7px;
                top: 50%;
                transform: translateY(-50%);
            }}
            
            /* ===== SVG PARA CONEXÕES ===== */
            svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }}
            
            .conexao-linha {{
                stroke: #3498db;
                stroke-width: 3;
                fill: none;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
            }}
            
            .conexao-selecionada {{
                stroke: #e74c3c;
                stroke-width: 4;
            }}
            
            /* ===== PAINEL LATERAL ===== */
            .painel-lateral {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                max-width: 280px;
                font-size: 12px;
                backdrop-filter: blur(10px);
                border: 2px solid #3498db;
            }}
            
            .painel-titulo {{
                font-weight: 700;
                font-size: 14px;
                color: #2c3e50;
                margin-bottom: 10px;
                padding-bottom: 8px;
                border-bottom: 2px solid #3498db;
            }}
            
            .painel-item {{
                margin: 6px 0;
                padding: 6px;
                background: #ecf0f1;
                border-radius: 4px;
                color: #34495e;
            }}
            
            .painel-item strong {{
                color: #2c3e50;
            }}
            
            /* ===== MINIMAP ===== */
            .minimap {{
                position: absolute;
                bottom: 15px;
                left: 15px;
                width: 200px;
                height: 150px;
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid #3498db;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            
            /* ===== STATUS BAR ===== */
            .status-bar {{
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                color: white;
                padding: 8px 15px;
                font-size: 11px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-top: 2px solid #3498db;
            }}
            
            .status-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .status-badge {{
                background: #3498db;
                padding: 3px 8px;
                border-radius: 12px;
                font-weight: 600;
            }}
            
            /* ===== TOOLTIP ===== */
            .tooltip {{
                position: absolute;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 11px;
                pointer-events: none;
                z-index: 10000;
                opacity: 0;
                transition: opacity 0.2s;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
            
            .tooltip.show {{
                opacity: 1;
            }}
            
            /* ===== BOTÃO FECHAR BLOCO ===== */
            .btn-fechar-bloco {{
                background: none;
                border: none;
                color: white;
                font-size: 16px;
                cursor: pointer;
                padding: 0;
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 3px;
                transition: background 0.2s;
            }}
            
            .btn-fechar-bloco:hover {{
                background: rgba(255, 255, 255, 0.2);
            }}
        </style>
    </head>
    <body>
        <!-- BARRA DE FERRAMENTAS -->
        <div class="toolbar">
            <div class="toolbar-section">
                <span class="toolbar-label">BLOCOS:</span>
                <button class="btn" onclick="adicionarBlocoTransferencia()">
                    📦 Função Transferência
                </button>
                <button class="btn btn-warning" onclick="adicionarBlocoSomador()">
                    ⊕ Somador
                </button>
                <button class="btn btn-success" onclick="adicionarBlocoGanho()">
                    📊 Ganho
                </button>
                <button class="btn" onclick="adicionarBlocoIntegrador()" style="background: linear-gradient(135deg, #9b59b6, #8e44ad);">
                    ∫ Integrador
                </button>
                <button class="btn" onclick="adicionarBlocoDerivador()" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    d/dt Derivador
                </button>
                <button class="btn btn-warning" onclick="adicionarBlocoAtraso()">
                    ⏱️ Atraso
                </button>
            </div>
            
            <div class="toolbar-section">
                <span class="toolbar-label">EDIÇÃO:</span>
                <button class="btn btn-danger" onclick="removerSelecionado()">
                    🗑️ Deletar
                </button>
                <button class="btn" onclick="duplicarSelecionado()">
                    📋 Duplicar
                </button>
            </div>
            
            <div class="toolbar-section">
                <span class="toolbar-label">AÇÕES:</span>
                <button class="btn btn-danger" onclick="limparDiagrama()">
                    🔄 Limpar Tudo
                </button>
                <button class="btn btn-success" onclick="autoOrganizar()">
                    🎯 Auto-organizar
                </button>
            </div>
        </div>
        
        <!-- ÁREA DE TRABALHO -->
        <div id="canvas-container">
            <svg id="conexoes-svg"></svg>
            
            <!-- PAINEL LATERAL DE INFORMAÇÕES -->
            <div class="painel-lateral">
                <div class="painel-titulo">📊 Informações do Sistema</div>
                <div class="painel-item"><strong>Blocos:</strong> <span id="info-blocos">0</span></div>
                <div class="painel-item"><strong>Conexões:</strong> <span id="info-conexoes">0</span></div>
                <div class="painel-item"><strong>Selecionado:</strong> <span id="info-selecionado">Nenhum</span></div>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #bdc3c7;">
                <div style="font-size: 10px; color: #7f8c8d; line-height: 1.4;">
                    <strong>💡 Dicas:</strong><br>
                    • Arraste blocos para mover<br>
                    • Clique em portas verdes para conectar<br>
                    • Use Ctrl+D para duplicar<br>
                    • Delete para remover selecionado
                </div>
            </div>
        </div>
        
        <!-- BARRA DE STATUS -->
        <div class="status-bar">
            <div class="status-item">
                <span>Sistema de Controle - Editor Xcos/Simulink</span>
            </div>
            <div class="status-item">
                <span>Zoom: <span class="status-badge">100%</span></span>
                <span>Grid: <span class="status-badge">ON</span></span>
            </div>
        </div>
        
        <!-- TOOLTIP -->
        <div id="tooltip" class="tooltip"></div>

        <script>
            // ===== VARIÁVEIS GLOBAIS =====
            let blocos = {blocos_data};
            let conexoes = {conexoes_data};
            let blocoIdCounter = {st.session_state.bloco_contador};
            let blocoSelecionado = null;
            let portaSelecionada = null;
            let arrastandoBloco = null;
            let offsetX = 0, offsetY = 0;
            let conexaoSelecionada = null;
            
            // ===== CONFIGURAÇÕES DE BLOCOS =====
            const CONFIGURACOES_BLOCOS = {{
                'Transferência': {{
                    cor: '#3498db',
                    icone: '📦',
                    largura: 160,
                    altura: 100
                }},
                'Somador': {{
                    cor: '#e67e22',
                    icone: '⊕',
                    largura: 80,
                    altura: 80
                }},
                'Ganho': {{
                    cor: '#27ae60',
                    icone: '📊',
                    largura: 100,
                    altura: 80
                }},
                'Integrador': {{
                    cor: '#9b59b6',
                    icone: '∫',
                    largura: 100,
                    altura: 80
                }},
                'Derivador': {{
                    cor: '#e74c3c',
                    icone: 'd/dt',
                    largura: 100,
                    altura: 80
                }},
                'Atraso': {{
                    cor: '#f39c12',
                    icone: '⏱️',
                    largura: 120,
                    altura: 80
                }}
            }};
            
            // ===== FUNÇÕES DE ADIÇÃO DE BLOCOS =====
            function adicionarBloco(tipo, config) {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                const blocoConfig = CONFIGURACOES_BLOCOS[tipo] || {{}};
                
                bloco.className = `bloco bloco-${{tipo.toLowerCase()}}`;
                bloco.id = 'bloco-' + blocoIdCounter;
                
                const blocoData = {{
                    id: blocoIdCounter,
                    tipo: tipo,
                    x: 100 + (blocos.length * 30) % 400,
                    y: 100 + (blocos.length * 30) % 300,
                    config: config,
                    largura: blocoConfig.largura || 140,
                    altura: blocoConfig.altura || 100
                }};
                
                blocos.push(blocoData);
                
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                bloco.style.width = blocoData.largura + 'px';
                bloco.style.height = blocoData.altura + 'px';
                
                let nomeDisplay = config.nome || tipo;
                let tfDisplay = config.tf || '';
                let icone = blocoConfig.icone || '📦';
                
                bloco.innerHTML = `
                    <div class="bloco-header">
                        <span>${{icone}} ${{tipo}}</span>
                        <button class="btn-fechar-bloco" onclick="removerBlocoEspecifico(${{blocoIdCounter}}); event.stopPropagation();">×</button>
                    </div>
                    <div class="bloco-body">
                        <div class="bloco-nome">${{nomeDisplay}}</div>
                        ${{tfDisplay ? '<div class="bloco-tf">' + tfDisplay + '</div>' : ''}}
                    </div>
                    <div class="porta porta-entrada" data-bloco="${{blocoIdCounter}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoIdCounter}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.addEventListener('click', selecionarBloco);
                
                const portas = bloco.querySelectorAll('.porta');
                portas.forEach(porta => {{
                    porta.addEventListener('click', clickPorta);
                    porta.addEventListener('mouseenter', mostrarTooltipPorta);
                    porta.addEventListener('mouseleave', esconderTooltip);
                }});
                
                blocoIdCounter++;
                atualizarInfo();
                salvarEstado();
            }}
            
            function adicionarBlocoTransferencia() {{
                const num = prompt('Numerador (ex: 1, s+1):', '1');
                if (num === null) return;
                const den = prompt('Denominador (ex: s+1, s^2+2*s+1):', 's+1');
                if (den === null) return;
                adicionarBloco('Transferência', {{
                    nome: 'G' + blocoIdCounter,
                    numerador: num,
                    denominador: den,
                    tf: num + ' / (' + den + ')'
                }});
            }}
            
            function adicionarBlocoSomador() {{
                adicionarBloco('Somador', {{nome: 'Σ'}});
            }}
            
            function adicionarBlocoGanho() {{
                const ganho = prompt('Valor do ganho K:', '1');
                if (ganho === null) return;
                adicionarBloco('Ganho', {{nome: 'K=' + ganho, valor: ganho, tf: ganho}});
            }}
            
            function adicionarBlocoIntegrador() {{
                adicionarBloco('Integrador', {{nome: '1/s', tf: '1/s'}});
            }}
            
            function adicionarBlocoDerivador() {{
                adicionarBloco('Derivador', {{nome: 's', tf: 's'}});
            }}
            
            function adicionarBlocoAtraso() {{
                const tau = prompt('Constante de tempo τ (segundos):', '1');
                if (tau === null) return;
                adicionarBloco('Atraso', {{
                    nome: 'e^(-' + tau + 's)',
                    tau: tau,
                    tf: 'e^(-' + tau + 's)'
                }});
            }}
            
            // ===== ARRASTAR E SOLTAR =====
            function iniciarArrastar(e) {{
                if (e.target.classList.contains('porta') || e.target.classList.contains('btn-fechar-bloco')) return;
                e.stopPropagation();
                arrastandoBloco = e.currentTarget;
                const rect = arrastandoBloco.getBoundingClientRect();
                const container = document.getElementById('canvas-container').getBoundingClientRect();
                offsetX = e.clientX - rect.left;
                offsetY = e.clientY - rect.top;
                
                arrastandoBloco.style.zIndex = '1001';
                
                document.addEventListener('mousemove', arrastar);
                document.addEventListener('mouseup', pararArrastar);
            }}
            
            function arrastar(e) {{
                if (arrastandoBloco) {{
                    const container = document.getElementById('canvas-container').getBoundingClientRect();
                    let x = e.clientX - container.left - offsetX;
                    let y = e.clientY - container.top - offsetY;
                    
                    // Snap to grid (opcional)
                    x = Math.round(x / 20) * 20;
                    y = Math.round(y / 20) * 20;
                    
                    x = Math.max(0, Math.min(x, container.width - arrastandoBloco.offsetWidth));
                    y = Math.max(0, Math.min(y, container.height - arrastandoBloco.offsetHeight));
                    
                    arrastandoBloco.style.left = x + 'px';
                    arrastandoBloco.style.top = y + 'px';
                    
                    const blocoId = parseInt(arrastandoBloco.id.split('-')[1]);
                    const bloco = blocos.find(b => b.id === blocoId);
                    if (bloco) {{
                        bloco.x = x;
                        bloco.y = y;
                    }}
                    
                    redesenharConexoes();
                }}
            }}
            
            function pararArrastar() {{
                if (arrastandoBloco) {{
                    arrastandoBloco.style.zIndex = '';
                }}
                arrastandoBloco = null;
                document.removeEventListener('mousemove', arrastar);
                document.removeEventListener('mouseup', pararArrastar);
                salvarEstado();
            }}
            
            // ===== SELEÇÃO =====
            function selecionarBloco(e) {{
                if (e.target.classList.contains('porta') || e.target.classList.contains('btn-fechar-bloco')) return;
                e.stopPropagation();
                
                document.querySelectorAll('.bloco').forEach(b => b.classList.remove('selecionado'));
                e.currentTarget.classList.add('selecionado');
                blocoSelecionado = parseInt(e.currentTarget.id.split('-')[1]);
                
                const bloco = blocos.find(b => b.id === blocoSelecionado);
                if (bloco) {{
                    document.getElementById('info-selecionado').textContent = bloco.config.nome || bloco.tipo;
                }}
            }}
            
            // ===== CONEXÕES =====
            function clickPorta(e) {{
                e.stopPropagation();
                const blocoId = parseInt(e.target.dataset.bloco);
                const tipoPorta = e.target.dataset.tipo;
                
                if (!portaSelecionada) {{
                    portaSelecionada = {{blocoId, tipo: tipoPorta, elemento: e.target}};
                    e.target.classList.add('porta-ativa');
                }} else {{
                    if (portaSelecionada.blocoId === blocoId) {{
                        // Clicou na mesma porta, cancela
                        portaSelecionada.elemento.classList.remove('porta-ativa');
                        portaSelecionada = null;
                        return;
                    }}
                    
                    if (portaSelecionada.tipo === 'saida' && tipoPorta === 'entrada') {{
                        // Verifica se já existe conexão
                        const jaExiste = conexoes.some(c => 
                            c.origem === portaSelecionada.blocoId && c.destino === blocoId
                        );
                        
                        if (!jaExiste) {{
                            conexoes.push({{
                                origem: portaSelecionada.blocoId,
                                destino: blocoId
                            }});
                            redesenharConexoes();
                        }}
                    }} else if (portaSelecionada.tipo === 'entrada' && tipoPorta === 'saida') {{
                        const jaExiste = conexoes.some(c => 
                            c.origem === blocoId && c.destino === portaSelecionada.blocoId
                        );
                        
                        if (!jaExiste) {{
                            conexoes.push({{
                                origem: blocoId,
                                destino: portaSelecionada.blocoId
                            }});
                            redesenharConexoes();
                        }}
                    }} else {{
                        mostrarNotificacao('Conecte saída → entrada ou entrada ← saída', 'warning');
                    }}
                    
                    document.querySelectorAll('.porta').forEach(p => p.classList.remove('porta-ativa'));
                    portaSelecionada = null;
                    atualizarInfo();
                    salvarEstado();
                }}
            }}
            
            function redesenharConexoes() {{
                const svg = document.getElementById('conexoes-svg');
                svg.innerHTML = '';
                
                // Criar definições de marcadores
                const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                
                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', 'arrowhead');
                marker.setAttribute('markerWidth', '10');
                marker.setAttribute('markerHeight', '10');
                marker.setAttribute('refX', '9');
                marker.setAttribute('refY', '3');
                marker.setAttribute('orient', 'auto');
                
                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', '0 0, 10 3, 0 6');
                polygon.setAttribute('fill', '#3498db');
                
                marker.appendChild(polygon);
                defs.appendChild(marker);
                svg.appendChild(defs);
                
                // Desenhar conexões
                conexoes.forEach((conexao, index) => {{
                    const blocoOrigem = document.getElementById('bloco-' + conexao.origem);
                    const blocoDestino = document.getElementById('bloco-' + conexao.destino);
                    
                    if (blocoOrigem && blocoDestino) {{
                        const rectOrigem = blocoOrigem.getBoundingClientRect();
                        const rectDestino = blocoDestino.getBoundingClientRect();
                        const container = document.getElementById('canvas-container').getBoundingClientRect();
                        
                        const x1 = rectOrigem.right - container.left;
                        const y1 = rectOrigem.top + rectOrigem.height/2 - container.top;
                        const x2 = rectDestino.left - container.left;
                        const y2 = rectDestino.top + rectDestino.height/2 - container.top;
                        
                        const dx = x2 - x1;
                        const curva = Math.min(Math.abs(dx) / 2, 100);
                        
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        const d = `M ${{x1}} ${{y1}} C ${{x1 + curva}} ${{y1}}, ${{x2 - curva}} ${{y2}}, ${{x2}} ${{y2}}`;
                        
                        path.setAttribute('d', d);
                        path.setAttribute('class', 'conexao-linha');
                        path.setAttribute('data-index', index);
                        path.setAttribute('marker-end', 'url(#arrowhead)');
                        path.style.pointerEvents = 'stroke';
                        path.style.cursor = 'pointer';
                        
                        path.addEventListener('click', (e) => {{
                            if (confirm('Deseja remover esta conexão?')) {{
                                conexoes.splice(index, 1);
                                redesenharConexoes();
                                atualizarInfo();
                                salvarEstado();
                            }}
                        }});
                        
                        svg.appendChild(path);
                    }}
                }});
            }}
            
            // ===== REMOÇÃO =====
            function removerSelecionado() {{
                if (blocoSelecionado !== null) {{
                    removerBlocoEspecifico(blocoSelecionado);
                }} else {{
                    mostrarNotificacao('Selecione um bloco primeiro!', 'warning');
                }}
            }}
            
            function removerBlocoEspecifico(blocoId) {{
                const blocoEl = document.getElementById('bloco-' + blocoId);
                if (blocoEl) {{
                    blocoEl.remove();
                    blocos = blocos.filter(b => b.id !== blocoId);
                    conexoes = conexoes.filter(c => c.origem !== blocoId && c.destino !== blocoId);
                    redesenharConexoes();
                    blocoSelecionado = null;
                    document.getElementById('info-selecionado').textContent = 'Nenhum';
                    atualizarInfo();
                    salvarEstado();
                }}
            }}
            
            function duplicarSelecionado() {{
                if (blocoSelecionado !== null) {{
                    const bloco = blocos.find(b => b.id === blocoSelecionado);
                    if (bloco) {{
                        const novoConfig = {{ ...bloco.config }};
                        novoConfig.nome = novoConfig.nome + '_copy';
                        adicionarBloco(bloco.tipo, novoConfig);
                    }}
                }} else {{
                    mostrarNotificacao('Selecione um bloco primeiro!', 'warning');
                }}
            }}
            
            function limparDiagrama() {{
                if (confirm('⚠️ Deseja realmente limpar todo o diagrama?\\nEsta ação não pode ser desfeita.')) {{
                    blocos = [];
                    conexoes = [];
                    blocoSelecionado = null;
                    document.getElementById('canvas-container').innerHTML = '<svg id="conexoes-svg"></svg><div class="painel-lateral"><div class="painel-titulo">📊 Informações do Sistema</div><div class="painel-item"><strong>Blocos:</strong> <span id="info-blocos">0</span></div><div class="painel-item"><strong>Conexões:</strong> <span id="info-conexoes">0</span></div><div class="painel-item"><strong>Selecionado:</strong> <span id="info-selecionado">Nenhum</span></div><hr style="margin: 10px 0; border: none; border-top: 1px solid #bdc3c7;"><div style="font-size: 10px; color: #7f8c8d; line-height: 1.4;"><strong>💡 Dicas:</strong><br>• Arraste blocos para mover<br>• Clique em portas verdes para conectar<br>• Use Ctrl+D para duplicar<br>• Delete para remover selecionado</div></div>';
                    atualizarInfo();
                    salvarEstado();
                }}
            }}
            
            // ===== AUTO-ORGANIZAR =====
            function autoOrganizar() {{
                if (blocos.length === 0) return;
                
                const espacamentoX = 200;
                const espacamentoY = 150;
                const startX = 50;
                const startY = 100;
                
                blocos.forEach((bloco, index) => {{
                    const linha = Math.floor(index / 4);
                    const coluna = index % 4;
                    
                    bloco.x = startX + (coluna * espacamentoX);
                    bloco.y = startY + (linha * espacamentoY);
                    
                    const blocoEl = document.getElementById('bloco-' + bloco.id);
                    if (blocoEl) {{
                        blocoEl.style.left = bloco.x + 'px';
                        blocoEl.style.top = bloco.y + 'px';
                    }}
                }});
                
                redesenharConexoes();
                salvarEstado();
            }}
            
            // ===== UTILITÁRIOS =====
            function atualizarInfo() {{
                document.getElementById('info-blocos').textContent = blocos.length;
                document.getElementById('info-conexoes').textContent = conexoes.length;
            }}
            
            function mostrarTooltipPorta(e) {{
                const tooltip = document.getElementById('tooltip');
                const tipo = e.target.dataset.tipo;
                tooltip.textContent = tipo === 'entrada' ? 'Porta de Entrada' : 'Porta de Saída';
                tooltip.style.left = (e.pageX + 10) + 'px';
                tooltip.style.top = (e.pageY - 30) + 'px';
                tooltip.classList.add('show');
            }}
            
            function esconderTooltip() {{
                document.getElementById('tooltip').classList.remove('show');
            }}
            
            function mostrarNotificacao(mensagem, tipo = 'info') {{
                alert(mensagem);
            }}
            
            function salvarEstado() {{
                window.parent.postMessage({{
                    type: 'salvar_diagrama',
                    blocos: blocos,
                    conexoes: conexoes,
                    contador: blocoIdCounter
                }}, '*');
            }}
            
            // ===== ATALHOS DE TECLADO =====
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Delete' && blocoSelecionado !== null) {{
                    removerSelecionado();
                }}
                
                if (e.ctrlKey && e.key === 'd') {{
                    e.preventDefault();
                    duplicarSelecionado();
                }}
            }});
            
            // ===== INICIALIZAÇÃO =====
            blocos.forEach(blocoData => {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                const blocoConfig = CONFIGURACOES_BLOCOS[blocoData.tipo] || {{}};
                
                bloco.className = `bloco bloco-${{blocoData.tipo.toLowerCase()}}`;
                bloco.id = 'bloco-' + blocoData.id;
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                bloco.style.width = (blocoData.largura || 140) + 'px';
                bloco.style.height = (blocoData.altura || 100) + 'px';
                
                let nomeDisplay = blocoData.config.nome || blocoData.tipo;
                let tfDisplay = blocoData.config.tf || '';
                let icone = blocoConfig.icone || '📦';
                
                bloco.innerHTML = `
                    <div class="bloco-header">
                        <span>${{icone}} ${{blocoData.tipo}}</span>
                        <button class="btn-fechar-bloco" onclick="removerBlocoEspecifico(${{blocoData.id}}); event.stopPropagation();">×</button>
                    </div>
                    <div class="bloco-body">
                        <div class="bloco-nome">${{nomeDisplay}}</div>
                        ${{tfDisplay ? '<div class="bloco-tf">' + tfDisplay + '</div>' : ''}}
                    </div>
                    <div class="porta porta-entrada" data-bloco="${{blocoData.id}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoData.id}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.addEventListener('click', selecionarBloco);
                
                const portas = bloco.querySelectorAll('.porta');
                portas.forEach(porta => {{
                    porta.addEventListener('click', clickPorta);
                    porta.addEventListener('mouseenter', mostrarTooltipPorta);
                    porta.addEventListener('mouseleave', esconderTooltip);
                }});
            }});

            redesenharConexoes();
            atualizarInfo();
        </script>
    </body>
    </html>
    """
    return html_code

def processar_diagrama_blocos():
    """Processa o diagrama de blocos e calcula o sistema equivalente"""
    blocos_diagrama = st.session_state.diagrama_blocos['blocos']
    conexoes_diagrama = st.session_state.diagrama_blocos['conexoes']
    
    if not blocos_diagrama:
        return None, "Nenhum bloco no diagrama"
    
    try:
        # Criar dicionário de funções de transferência
        tfs = {}
        for bloco in blocos_diagrama:
            bloco_id = bloco['id']
            config = bloco['config']
            
            if bloco['tipo'] == 'Transferência':
                tf, _ = converter_para_tf(config['numerador'], config['denominador'])
                tfs[bloco_id] = tf
            elif bloco['tipo'] == 'Ganho':
                tfs[bloco_id] = TransferFunction([float(config['valor'])], [1])
            elif bloco['tipo'] == 'Integrador':
                tfs[bloco_id] = TransferFunction([1], [1, 0])
            elif bloco['tipo'] == 'Derivador':
                tfs[bloco_id] = TransferFunction([1, 0], [1])
            elif bloco['tipo'] == 'Somador':
                tfs[bloco_id] = TransferFunction([1], [1])
            elif bloco['tipo'] == 'Atraso':
                # Aproximação de Padé para atraso de primeira ordem
                tau = float(config.get('tau', 1))
                # Aproximação: e^(-tau*s) ≈ (1 - tau*s/2) / (1 + tau*s/2)
                num = [1, -tau/2]
                den = [1, tau/2]
                tfs[bloco_id] = TransferFunction(num, den)
        
        # Se não há conexões, multiplica todos em série
        if not conexoes_diagrama:
            if len(blocos_diagrama) == 1:
                return tfs[blocos_diagrama[0]['id']], "Sistema de bloco único"
            
            # Multiplicação em série de todos os blocos
            sistema_final = tfs[blocos_diagrama[0]['id']]
            for i in range(1, len(blocos_diagrama)):
                sistema_final = sistema_final * tfs[blocos_diagrama[i]['id']]
            
            return sistema_final, f"Sistema com {len(blocos_diagrama)} blocos em série"
        
        # Processar conexões (simplificado - assume conexão em série)
        blocos_processados = set()
        sistema_parcial = None
        
        # Encontrar primeiro bloco (sem entrada)
        blocos_com_entrada = {c['destino'] for c in conexoes_diagrama}
        blocos_sem_entrada = [b['id'] for b in blocos_diagrama if b['id'] not in blocos_com_entrada]
        
        if blocos_sem_entrada:
            primeiro_bloco = blocos_sem_entrada[0]
            sistema_parcial = tfs[primeiro_bloco]
            blocos_processados.add(primeiro_bloco)
            
            # Seguir a cadeia de conexões
            bloco_atual = primeiro_bloco
            while True:
                proxima_conexao = next((c for c in conexoes_diagrama if c['origem'] == bloco_atual), None)
                if not proxima_conexao:
                    break
                
                proximo_bloco = proxima_conexao['destino']
                if proximo_bloco not in blocos_processados:
                    sistema_parcial = sistema_parcial * tfs[proximo_bloco]
                    blocos_processados.add(proximo_bloco)
                    bloco_atual = proximo_bloco
                else:
                    break
        else:
            # Fallback: multiplicar todos
            sistema_parcial = tfs[blocos_diagrama[0]['id']]
            for i in range(1, len(blocos_diagrama)):
                sistema_parcial = sistema_parcial * tfs[blocos_diagrama[i]['id']]
        
        num_blocos = len(blocos_processados) if blocos_processados else len(blocos_diagrama)
        return sistema_parcial, f"Sistema processado com {num_blocos} bloco(s) e {len(conexoes_diagrama)} conexão(ões)"
    
    except Exception as e:
        return None, f"Erro ao processar diagrama: {str(e)}"

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Sistema de Controle - Xcos/Simulink", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🎛️ Sistema de Modelagem e Análise de Controle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Editor Visual Avançado - Inspirado em Xcos/Simulink</p>', unsafe_allow_html=True)
    
    inicializar_blocos()
    
    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False
    
    if 'mostrar_ajuda' not in st.session_state:
        st.session_state.mostrar_ajuda = False
    
    if 'modo_editor' not in st.session_state:
        st.session_state.modo_editor = 'classico'
    
    # Seletor de modo
    st.sidebar.header("🎛️ Modo de Trabalho")
    modo = st.sidebar.radio(
        "Escolha o modo:",
        ['📝 Clássico (Lista)', '🎨 Editor Visual (Xcos/Simulink)'],
        index=0 if st.session_state.modo_editor == 'classico' else 1,
        help="Clássico: adicionar blocos por formulário\nEditor Visual: arrastar e conectar blocos graficamente estilo Xcos/Simulink"
    )
    
    if modo == '🎨 Editor Visual (Xcos/Simulink)':
        st.session_state.modo_editor = 'visual'
    else:
        st.session_state.modo_editor = 'classico'
    
    # =====================================================
    # MODO EDITOR VISUAL
    # =====================================================
    if st.session_state.modo_editor == 'visual':
        st.markdown("---")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info("🎨 **Modo Editor Visual Ativo**\n\nConstrua sistemas de controle visualmente!")
        with col_info2:
            st.success(f"📦 **Blocos:** {len(st.session_state.diagrama_blocos['blocos'])}")
        with col_info3:
            st.success(f"🔗 **Conexões:** {len(st.session_state.diagrama_blocos['conexoes'])}")
        
        html_editor = criar_diagrama_blocos_html()
        resultado = components.html(html_editor, height=750, scrolling=False)
        
        # Atualizar o estado se houver resultado do componente
        if resultado is not None and isinstance(resultado, dict):
            if resultado.get('type') == 'salvar_diagrama':
                st.session_state.diagrama_blocos['blocos'] = resultado.get('blocos', [])
                st.session_state.diagrama_blocos['conexoes'] = resultado.get('conexoes', [])
                st.session_state.bloco_contador = resultado.get('contador', st.session_state.bloco_contador)
        
        st.markdown("---")
        
        # Script para receber mensagens do iframe
        components.html("""
        <script>
            window.addEventListener('message', function(event) {
                if (event.data.type === 'salvar_diagrama') {
                    // Enviar dados de volta ao Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        data: event.data
                    }, '*');
                }
            });
        </script>
        """, height=0)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Processar e Analisar Sistema", type="primary", use_container_width=True):
                if len(st.session_state.diagrama_blocos['blocos']) == 0:
                    st.warning("⚠️ Adicione blocos ao diagrama primeiro!")
                else:
                    with st.spinner("🔄 Processando diagrama..."):
                        sistema, msg = processar_diagrama_blocos()
                        
                        if sistema:
                            st.success(f"✅ {msg}")
                            
                            # Exibir função de transferência
                            st.markdown("### 📐 Função de Transferência do Sistema")
                            st.code(f"G(s) = {sistema}", language="python")
                            
                            # Métricas de Desempenho
                            with st.expander("📊 **Métricas de Desempenho**", expanded=True):
                                desempenho = calcular_desempenho(sistema)
                                col_a, col_b = st.columns(2)
                                items = list(desempenho.items())
                                mid = len(items) // 2
                                
                                with col_a:
                                    for chave, valor in items[:mid]:
                                        st.metric(chave, valor)
                                
                                with col_b:
                                    for chave, valor in items[mid:]:
                                        st.metric(chave, valor)
                            
                            # Análise de Estabilidade
                            with st.expander("🔍 **Análise de Estabilidade**", expanded=True):
                                polos = ctrl.poles(sistema)
                                zeros = ctrl.zeros(sistema)
                                
                                col_1, col_2 = st.columns(2)
                                with col_1:
                                    st.markdown("**Polos do Sistema:**")
                                    for i, polo in enumerate(polos):
                                        if np.isreal(polo):
                                            st.text(f"p{i+1} = {polo.real:.4f}")
                                        else:
                                            st.text(f"p{i+1} = {polo.real:.4f} ± {abs(polo.imag):.4f}j")
                                
                                with col_2:
                                    st.markdown("**Zeros do Sistema:**")
                                    if len(zeros) > 0:
                                        for i, zero in enumerate(zeros):
                                            if np.isreal(zero):
                                                st.text(f"z{i+1} = {zero.real:.4f}")
                                            else:
                                                st.text(f"z{i+1} = {zero.real:.4f} ± {abs(zero.imag):.4f}j")
                                    else:
                                        st.text("Nenhum zero")
                                
                                # Verificar estabilidade
                                estavel = all(np.real(p) < 0 for p in polos)
                                if estavel:
                                    st.success("✅ **Sistema ESTÁVEL** - Todos os polos no semiplano esquerdo")
                                else:
                                    st.error("❌ **Sistema INSTÁVEL** - Polos no semiplano direito detectados")
                            
                            # Gráficos de Análise
                            st.markdown("### 📈 Análises Gráficas")
                            
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "📈 Resposta Temporal", 
                                "📊 Diagrama de Bode", 
                                "🎯 Polos e Zeros",
                                "🔄 Lugar das Raízes (LGR)",
                                "🌀 Nyquist"
                            ])
                            
                            with tab1:
                                entrada_sinal = st.selectbox("Sinal de Entrada:", INPUT_SIGNALS, key="entrada_visual")
                                fig, t, y = plot_resposta_temporal(sistema, entrada_sinal)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Informações adicionais
                                st.markdown("**Informações da Resposta:**")
                                col_i1, col_i2, col_i3 = st.columns(3)
                                with col_i1:
                                    st.metric("Valor Final", f"{y[-1]:.3f}")
                                with col_i2:
                                    st.metric("Valor Máximo", f"{np.max(y):.3f}")
                                with col_i3:
                                    st.metric("Tempo de Simulação", f"{t[-1]:.2f} s")
                            
                            with tab2:
                                fig = plot_bode(sistema, 'both')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Margens de estabilidade
                                gm, pm, wg, wp = margin(sistema)
                                st.markdown("**Margens de Estabilidade:**")
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                with col_m1:
                                    gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf
                                    st.metric("Margem de Ganho", f"{formatar_numero(gm_db)} dB")
                                with col_m2:
                                    st.metric("Margem de Fase", f"{formatar_numero(pm)}°")
                                with col_m3:
                                    st.metric("ωg (fase)", f"{formatar_numero(wg)} rad/s")
                                with col_m4:
                                    st.metric("ωp (ganho)", f"{formatar_numero(wp)} rad/s")
                            
                            with tab3:
                                fig = plot_polos_zeros(sistema)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab4:
                                try:
                                    fig = plot_lgr(sistema)
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.info("💡 O LGR mostra como os polos do sistema variam com o ganho K")
                                except Exception as e:
                                    st.warning(f"⚠️ Não foi possível gerar o LGR: {str(e)}")
                            
                            with tab5:
                                try:
                                    fig, polos_spd, voltas, Z = plot_nyquist(sistema)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    col_n1, col_n2, col_n3 = st.columns(3)
                                    with col_n1:
                                        st.metric("Polos SPD (P)", polos_spd)
                                    with col_n2:
                                        st.metric("Voltas (N)", voltas)
                                    with col_n3:
                                        if Z == 0:
                                            st.success(f"Z = {Z} ✅ ESTÁVEL")
                                        else:
                                            st.error(f"Z = {Z} ❌ INSTÁVEL")
                                except Exception as e:
                                    st.warning(f"⚠️ Não foi possível gerar Nyquist: {str(e)}")
                        else:
                            st.error(f"❌ {msg}")
        
        with col2:
            if st.button("💾 Exportar Diagrama", use_container_width=True):
                diagrama_json = json.dumps(st.session_state.diagrama_blocos, indent=2)
                st.download_button(
                    label="📥 Baixar JSON",
                    data=diagrama_json,
                    file_name="diagrama_sistema_controle.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            uploaded_file = st.file_uploader("📤 Importar Diagrama", type=['json'], key="import_diagram")
            if uploaded_file is not None:
                try:
                    diagrama_importado = json.load(uploaded_file)
                    st.session_state.diagrama_blocos = diagrama_importado
                    st.success("✅ Diagrama importado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao importar: {e}")
        
        with col4:
            if st.button("📖 Tutorial", use_container_width=True):
                st.info("""
                **🎓 Tutorial do Editor Visual**
                
                **Adicionando Blocos:**
                • Clique nos botões da barra superior
                • Cada tipo tem cor e ícone específicos
                
                **Conectando Blocos:**
                1. Clique na porta verde de **saída** (direita)
                2. Clique na porta verde de **entrada** (esquerda)
                3. A conexão é criada automaticamente
                
                **Movendo Blocos:**
                • Arraste qualquer bloco pela área
                • Os blocos se alinham à grade
                
                **Editando:**
                • **Deletar:** Selecione e pressione Delete ou clique no X
                • **Duplicar:** Selecione e pressione Ctrl+D
                • **Auto-organizar:** Clique no botão da barra
                
                **Tipos de Blocos:**
                • 📦 **Função Transferência:** G(s) = num/den
                • ⊕ **Somador:** Soma sinais
                • 📊 **Ganho:** Multiplicador K
                • ∫ **Integrador:** 1/s
                • d/dt **Derivador:** s
                • ⏱️ **Atraso:** e^(-τs)
                """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Estatísticas do Diagrama")
        st.sidebar.metric("Total de Blocos", len(st.session_state.diagrama_blocos['blocos']))
        st.sidebar.metric("Total de Conexões", len(st.session_state.diagrama_blocos['conexoes']))
        
        if st.session_state.diagrama_blocos['blocos']:
            st.sidebar.markdown("### 📋 Blocos no Diagrama")
            for bloco in st.session_state.diagrama_blocos['blocos']:
                st.sidebar.text(f"• {bloco['tipo']}: {bloco['config'].get('nome', 'N/A')}")
        
        return
    
    # =====================================================
    # MODO CLÁSSICO (código original mantido)
    # =====================================================
    
    with st.sidebar:
        st.header("🧱 Adicionar Blocos")
        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")
        
        if st.button("➕ Adicionar", use_container_width=True):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            if sucesso:
                st.success(mensagem)
            else:
                st.error(mensagem)
        
        if not st.session_state.blocos.empty:
            st.header("🗑️ Excluir Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("❌ Excluir", use_container_width=True):
                mensagem = remover_bloco(excluir)
                st.success(mensagem)
        
        st.header("⚙️ Configurações")
        if st.button("🔢 Habilitar Cálculo de Erro" if not st.session_state.calculo_erro_habilitado else "❌ Desabilitar Cálculo de Erro", use_container_width=True):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
            st.rerun()
    
    if st.session_state.calculo_erro_habilitado:
        st.subheader("📊 Cálculo de Erro Estacionário")
        col1, col2 = st.columns(2)
        with col1:
            num_erro = st.text_input("Numerador", value="", key="num_erro")
        with col2:
            den_erro = st.text_input("Denominador", value="", key="den_erro")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔍 Calcular Erro Estacionário"):
                try:
                    G, _ = converter_para_tf(num_erro, den_erro)
                    tipo, Kp, Kv, Ka = constantes_de_erro(G)
                    
                    df = pd.DataFrame([{"Tipo": tipo, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                    st.subheader("📊 Resultado")
                    st.dataframe(
                        df.style.format({
                            "Kp": lambda x: formatar_numero(x),
                            "Kv": lambda x: formatar_numero(x),
                            "Ka": lambda x: formatar_numero(x)
                        }),
                        height=120,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
        
        with btn_col2:
            if st.button("🗑️ Remover Planta", key="remover_planta"):
                if not st.session_state.blocos.empty:
                    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['tipo'] != 'Planta']
                    st.success("Plantas removidas!")
                else:
                    st.warning("Nenhuma planta para remover")
    else:
        st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na barra lateral")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔍 Tipo de Sistema")
        tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False)
        
        if usar_ganho:
            K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1)
            st.info(f"✅ Ganho K: {K:.2f}")
        else:
            K = 1.0
        
        st.subheader("📊 Análises")
        analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
        analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)
    
    with col1:
        st.subheader("📈 Resultados")
        
        if st.button("▶️ Executar Simulação", use_container_width=True):
            try:
                df = st.session_state.blocos
                if df.empty:
                    st.warning("Adicione blocos primeiro.")
                    st.stop()
                
                planta = obter_bloco_por_tipo('Planta')
                controlador = obter_bloco_por_tipo('Controlador')
                sensor = obter_bloco_por_tipo('Sensor')
                
                if planta is None:
                    st.error("Adicione pelo menos uma Planta.")
                    st.stop()
                
                ganho_tf = TransferFunction([K], [1])
                
                if tipo_malha == "Malha Aberta":
                    sistema = ganho_tf * planta
                    st.info(f"🔧 Sistema em Malha Aberta com K = {K:.2f}")
                else:
                    planta_com_ganho = ganho_tf * planta
                    sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                    st.info(f"🔧 Sistema em Malha Fechada com K = {K:.2f}")
                
                for analise in analises:
                    st.markdown(f"### 🔎 {analise}")
                    
                    if analise == 'Resposta no tempo':
                        fig, t_out, y = plot_resposta_temporal(sistema, entrada)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analise == 'Desempenho':
                        desempenho = calcular_desempenho(sistema)
                        for chave, valor in desempenho.items():
                            st.markdown(f"**{chave}:** {valor}")
                    
                    elif analise == 'Diagrama De Bode Magnitude':
                        fig = plot_bode(sistema, 'magnitude')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analise == 'Diagrama De Bode Fase':
                        fig = plot_bode(sistema, 'fase')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analise == 'Diagrama de Polos e Zeros':
                        fig = plot_polos_zeros(sistema)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analise == 'LGR':
                        fig = plot_lgr(sistema)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analise == 'Nyquist':
                        fig, polos_spd, voltas, Z = plot_nyquist(sistema)
                        st.markdown(f"**Polos SPD (P):** {polos_spd}")
                        st.markdown(f"**Voltas (N):** {voltas}")
                        st.markdown(f"**Z = {Z} → {'✅ Estável' if Z == 0 else '❌ Instável'}**")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Experimente!")
    st.sidebar.success("""
    **Editor Visual Xcos/Simulink**
    
    Construa sistemas de controle
    de forma visual e intuitiva!
    
    Clique no modo Editor Visual
    para começar! 🎨
    """)

if __name__ == "__main__":
    main()
