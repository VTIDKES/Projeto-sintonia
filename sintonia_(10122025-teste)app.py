# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Analise de Sistemas de Controle
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

# =====================================================
# CONFIGURACOES E CONSTANTES
# =====================================================

ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabolica']

# =====================================================
# FUNCOES AUXILIARES
# =====================================================

def formatar_numero(valor):
    if np.isinf(valor):
        return '∞'
    elif np.isnan(valor):
        return '-'
    else:
        return f"{valor:.3f}"

# =====================================================
# FUNCOES DE TRANSFERENCIA
# =====================================================

def converter_para_tf(numerador_str, denominador_str):
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
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    tipo = sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))
    return tipo

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
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])
    G = controlador * planta
    H = sensor
    return ctrl.feedback(G, H)

# =====================================================
# ANALISE DE SISTEMAS
# =====================================================

def calcular_desempenho(tf):
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
    tau = -1 / polos[0].real
    resultado.update({
        'Tipo': '1a Ordem',
        'Const. tempo (t)': f"{formatar_numero(tau)} s",
        'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(4 * tau)} s",
        'Freq. natural (wn)': f"{formatar_numero(1/tau)} rad/s",
        'Fator amortec. (z)': "1.0"
    })
    return resultado

def _desempenho_ordem2(polos, resultado):
    wn = np.sqrt(np.prod(np.abs(polos))).real
    zeta = -np.real(polos[0]) / wn
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
    Tp = np.pi / wd if wd > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
    resultado.update({
        'Tipo': '2a Ordem',
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(wd)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s"
    })
    return resultado

def _desempenho_ordem_superior(polos, ordem, resultado):
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
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
        'Tipo': f'{ordem}a Ordem (Par dominante)' if par_dominante else f'{ordem}a Ordem (Polo dominante)',
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s",
    })
    return resultado

def estimar_tempo_final_simulacao(tf):
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
# FUNCOES DE PLOTAGEM
# =====================================================

def configurar_linhas_interativas(fig):
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(line=dict(color='green', width=2, dash='dash')),
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    )
    return fig

def plot_polos_zeros(tf, fig=None):
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)
    if fig is None:
        fig = go.Figure()
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=12, color='blue'), name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Diagrama de Polos e Zeros', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    return fig

def _gerar_sinal_entrada(entrada, t):
    sinais = {
        'Degrau': np.ones_like(t), 'Rampa': t, 'Senoidal': np.sin(2*np.pi*t),
        'Impulso': np.concatenate([[1], np.zeros(len(t)-1)]), 'Parabolica': t**2
    }
    return sinais[entrada]

def plot_resposta_temporal(sistema, entrada):
    tempo_final = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tempo_final, 1000)
    u = _gerar_sinal_entrada(entrada, t)
    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=u[:len(t_out)], mode='lines',
        line=dict(dash='dash', color='blue'), name='Entrada',
        hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines',
        line=dict(color='red'), name='Saida',
        hovertemplate='Tempo: %{x:.2f}s<br>Saida: %{y:.3f}<extra></extra>'))
    fig.update_layout(title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Amplitude', showlegend=True, hovermode='x unified')
    fig = configurar_linhas_interativas(fig)
    return fig, t_out, y

def plot_bode(sistema, tipo='both'):
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)
    if tipo == 'both':
        fig = make_subplots(rows=2, cols=1,
            subplot_titles=('Bode - Magnitude', 'Bode - Fase'), vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase', showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)
        fig.update_layout(height=700, title_text="Diagrama de Bode")
    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='blue', width=3), name='Magnitude'))
        fig.update_layout(title='Bode - Magnitude', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Magnitude (dB)", xaxis_type='log')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='red', width=3), name='Fase'))
        fig.update_layout(title='Bode - Fase', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Fase (deg)", xaxis_type='log')
    fig = configurar_linhas_interativas(fig)
    return fig

def plot_lgr(sistema):
    rlist, klist = root_locus(sistema, plot=False)
    fig = go.Figure()
    for i, r in enumerate(rlist.T):
        fig.add_trace(go.Scatter(x=np.real(r), y=np.imag(r), mode='lines',
            line=dict(color='blue', width=1), name=f'Ramo {i+1}', showlegend=False))
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=10, color='green'), name='Zeros'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Lugar Geometrico das Raizes (LGR)',
        xaxis_title='Parte Real', yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    return fig

def plot_nyquist(sistema):
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H.real, y=H.imag, mode='lines',
        line=dict(color='blue', width=2), name='Nyquist'))
    fig.add_trace(go.Scatter(x=H.real, y=-H.imag, mode='lines',
        line=dict(dash='dash', color='gray', width=1), name='Reflexo simetrico'))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'), name='Ponto critico (-1,0)'))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(title='Diagrama de Nyquist', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    polos = ctrl.poles(sistema)
    polos_spd = sum(1 for p in polos if np.real(p) > 0)
    voltas = 0
    Z = polos_spd + voltas
    return fig, polos_spd, voltas, Z

# =====================================================
# GERENCIAMENTO DE BLOCOS (MODO CLASSICO)
# =====================================================

def inicializar_blocos():
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])
    if 'modo_editor' not in st.session_state:
        st.session_state.modo_editor = 'classico'
    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False

def adicionar_bloco(nome, tipo, numerador, denominador):
    try:
        tf, tf_symb = converter_para_tf(numerador, denominador)
        novo = pd.DataFrame([{
            'nome': nome, 'tipo': tipo, 'numerador': numerador,
            'denominador': denominador, 'tf': tf, 'tf_simbolico': tf_symb
        }])
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco {nome} adicionado."
    except Exception as e:
        return False, f"Erro na conversao: {e}"

def remover_bloco(nome):
    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['nome'] != nome]
    return f"Bloco {nome} excluido."

def obter_bloco_por_tipo(tipo):
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None


# =====================================================
# EDITOR VISUAL - HTML COMPLETO INLINE
# Nao depende de nenhum arquivo externo
# =====================================================

def _load_visual_editor_html():
    return r'''<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
:root{--bg:#0e1117;--sf:#1a1d2e;--sf2:#252840;--bd:#333654;--tx:#e0e4f0;--txm:#8890b0;
--acc:#5b6be0;--grn:#34d399;--red:#f87171;--yel:#fbbf24;--blu:#60a5fa;--pur:#a78bfa;--pnk:#f472b6}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:auto;min-height:100%;font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);overflow-y:auto}
.app{display:flex;flex-direction:column;min-height:100%}
.toolbar{background:var(--sf);border-bottom:1px solid var(--bd);padding:6px 10px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;min-height:44px;z-index:50;position:sticky;top:0}
.toolbar .sep{width:1px;height:26px;background:var(--bd);margin:0 3px}
.toolbar .lbl{font-size:10px;color:var(--txm);text-transform:uppercase;letter-spacing:.5px;margin-right:2px;white-space:nowrap}
.tb{display:inline-flex;align-items:center;gap:4px;background:var(--sf2);color:#c8cad8;border:1px solid var(--bd);
border-radius:6px;padding:6px 10px;font-size:11px;cursor:pointer;white-space:nowrap;transition:all .15s;min-height:34px;touch-action:manipulation}
.tb:hover,.tb:active{background:#2f3349;border-color:var(--acc);color:#fff}
.workspace{display:flex;height:460px;min-height:320px;flex-shrink:0}
.canvas-wrap{flex:1;position:relative;overflow:hidden;background:radial-gradient(circle,#141722,#0e1117);touch-action:none}
.canvas-wrap::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,#1e2235 1px,transparent 1px);background-size:24px 24px;opacity:.45;pointer-events:none}
.canvas{position:absolute;inset:0}
.wires{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2}
.wires path{pointer-events:visibleStroke;cursor:pointer}
.block{position:absolute;z-index:10;border-radius:10px;cursor:grab;user-select:none;min-width:120px;box-shadow:0 4px 16px rgba(0,0,0,.35)}
.block:active{cursor:grabbing}
.block.sel{box-shadow:0 0 0 2px var(--yel),0 0 20px rgba(251,191,36,.2)!important}
.block-header{font-size:9px;text-transform:uppercase;letter-spacing:.7px;padding:6px 10px 2px;opacity:.6}
.block-body{padding:4px 10px 8px;font-weight:700;font-size:13px;color:#fff}
.block-tf-disp{font-family:monospace;font-size:10px;margin-top:3px;padding:3px 6px;background:rgba(0,0,0,.3);border-radius:4px;color:#a0b8d8;text-align:center}
.block-tf-disp .tf-num{border-bottom:1px solid rgba(255,255,255,.2);padding-bottom:2px;margin-bottom:2px}
.block-tf{background:linear-gradient(135deg,#1e3a5f,#152844);border:1px solid #2a5a8f}
.block-gain{background:linear-gradient(135deg,#2d1f4e,#1f1635);border:1px solid #5a3d8f}
.block-int{background:linear-gradient(135deg,#3d3a1a,#2a2812);border:1px solid #8a7a2d}
.block-der{background:linear-gradient(135deg,#2a2a2a,#1e1e1e);border:1px solid #555}
.block-pid{background:linear-gradient(135deg,#2a1f4e,#1a1535);border:1px solid #5548a0}
.block-sensor{background:linear-gradient(135deg,#4a1a2d,#351020);border:1px solid #8a2d50}
.block-input{background:linear-gradient(135deg,#1a3d1a,#0f250f);border:1px solid #2d8a2d;border-radius:20px}
.block-output{background:linear-gradient(135deg,#3d1a1a,#250f0f);border:1px solid #8a2d2d;border-radius:20px}
.block-sum{background:linear-gradient(135deg,#1a3d3a,#122a28);border:2px solid #2d8a70;border-radius:50%;min-width:56px;width:56px;height:56px;display:flex;align-items:center;justify-content:center}
.block-sum .block-header{display:none}.block-sum .block-body{padding:0;font-size:20px;text-align:center}
.block-branch{background:#3b82f6;border:2px solid #2563eb;border-radius:50%;min-width:20px;width:20px;height:20px}
.block-branch .block-header,.block-branch .block-body{display:none}
.port{position:absolute;width:14px;height:14px;border-radius:50%;cursor:crosshair;z-index:20;transition:transform .12s}
.port::after{content:"";position:absolute;inset:3px;border-radius:50%;background:currentColor}
.port-in{border:2px solid var(--grn);color:var(--grn);background:var(--bg)}
.port-out{border:2px solid var(--blu);color:var(--blu);background:var(--bg)}
.port:hover,.port:active{transform:scale(1.5)!important;box-shadow:0 0 10px currentColor}
.port.active{border-color:var(--yel);color:var(--yel);transform:scale(1.4)!important}
.sign-label{position:absolute;font-size:11px;font-weight:700;color:var(--grn);pointer-events:none}
.panel{width:240px;background:var(--sf);border-left:1px solid var(--bd);display:flex;flex-direction:column;overflow-y:auto;z-index:30;font-size:12px}
.panel-section{padding:10px 12px;border-bottom:1px solid var(--bd)}
.panel-section h4{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--txm);margin-bottom:8px}
.pg{margin-bottom:8px}
.pg label{display:block;font-size:10px;text-transform:uppercase;color:var(--txm);margin-bottom:3px}
.pg input{width:100%;padding:6px 8px;background:var(--sf2);border:1px solid var(--bd);border-radius:5px;color:var(--tx);font-size:12px;outline:none}
.pg input:focus{border-color:var(--acc)}
.hint{font-size:11px;color:var(--txm);line-height:1.5}
.hint b{color:var(--tx)}
.mode-bar{display:flex;gap:0;background:var(--sf);border-top:1px solid var(--bd)}
.mode-btn{flex:1;padding:12px;background:none;border:none;border-bottom:3px solid transparent;color:var(--txm);font-size:14px;font-weight:700;cursor:pointer;transition:all .15s;touch-action:manipulation;letter-spacing:.3px}
.mode-btn.active{color:var(--grn);border-bottom-color:var(--grn);background:rgba(52,211,153,.06)}
.mode-btn:hover{color:var(--tx)}
.manual-sec{background:var(--sf);padding:20px;border-top:1px solid var(--bd);display:none}
.manual-sec.vis{display:block}
.manual-sec h4{font-size:12px;text-transform:uppercase;color:var(--acc);margin-bottom:14px;letter-spacing:.5px}
.man-tabs{display:flex;gap:8px;margin-bottom:16px}
.man-tab{padding:8px 16px;background:var(--sf2);border:1px solid var(--bd);border-radius:8px;color:var(--txm);font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;touch-action:manipulation}
.man-tab.active{background:var(--acc);border-color:var(--acc);color:#fff}
.man-tab:hover:not(.active){border-color:var(--acc);color:var(--tx)}
.man-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px}
.man-row .pg input{font-size:14px;padding:10px 12px}
.man-row .pg label{font-size:11px;margin-bottom:4px}
.man-hint{font-size:11px;color:var(--txm);margin-top:6px;line-height:1.6}
.man-hint code{background:var(--sf2);padding:1px 5px;border-radius:3px;color:var(--blu);font-size:11px}
.calc-bar{background:var(--sf);border-top:1px solid var(--bd);border-bottom:2px solid #22c55e;padding:12px 20px;text-align:center}
.calc-bar button{background:#16a34a;border:2px solid #22c55e;color:#fff;font-weight:700;font-size:16px;padding:14px 40px;border-radius:10px;cursor:pointer;letter-spacing:1px;box-shadow:0 0 20px rgba(34,197,94,.3);touch-action:manipulation}
.calc-bar button:hover{background:#22c55e;transform:scale(1.03)}
.results{background:var(--sf);padding:0;display:none}
.results.vis{display:block}
.results-hdr{padding:16px 20px 10px;display:flex;align-items:center;justify-content:space-between}
.results-hdr h3{font-size:16px;color:var(--grn)}
.rbody{padding:0 20px 20px}
.rcard{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:16px}
.rcard h4{font-size:12px;text-transform:uppercase;color:var(--acc);margin-bottom:10px}
.tf-disp{font-family:monospace;text-align:center;padding:12px;background:rgba(0,0,0,.3);border-radius:6px;font-size:14px;line-height:1.8}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.mbox{background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center}
.mbox .ml{font-size:10px;text-transform:uppercase;color:var(--txm);margin-bottom:4px}
.mbox .mv{font-size:16px;font-weight:700;color:var(--grn)}
.ebox{background:#3a1520;border:1px solid #8a3050;color:#ff8fa3;padding:12px;border-radius:8px;font-size:13px}
.pzl{font-family:monospace;font-size:12px;line-height:1.8}
@media(max-width:768px){
.workspace{flex-direction:column;height:350px}
.panel{width:100%;max-height:150px;border-left:none;border-top:1px solid var(--bd)}
.toolbar{padding:4px 6px;gap:3px}.tb{padding:5px 7px;font-size:10px;min-height:30px}
.calc-bar button{font-size:14px;padding:10px 24px;width:100%}
.mgrid{grid-template-columns:repeat(2,1fr)}
.man-row{grid-template-columns:1fr}.man-tabs{flex-wrap:wrap}
}
</style></head><body><div class="app">
<div class="toolbar" id="diag-toolbar">
<span class="lbl">Sinais:</span>
<button class="tb" style="background:#16382a;border-color:#2d8a55;color:var(--grn)" data-add="input">R(s)</button>
<button class="tb" style="background:#381620;border-color:#8a2d3a;color:var(--red)" data-add="output">Y(s)</button>
<div class="sep"></div>
<span class="lbl">Blocos:</span>
<button class="tb" style="background:#162038;border-color:#2d558a;color:var(--blu)" data-add="tf">G(s)</button>
<button class="tb" style="background:#201638;border-color:#5a2d8a;color:var(--pur)" data-add="gain">K</button>
<button class="tb" data-add="sum">&Sigma;</button>
<button class="tb" data-add="int">1/s</button>
<button class="tb" data-add="der">s</button>
<button class="tb" style="background:#201638;border-color:#5a2d8a;color:var(--pur)" data-add="pid">PID</button>
<button class="tb" style="background:#381628;border-color:#8a2d5a;color:var(--pnk)" data-add="sensor">H(s)</button>
<button class="tb" data-add="branch">Ramif</button>
<div class="sep"></div>
<button class="tb" style="background:#3a1520;border-color:#8a3050;color:#ff8fa3" id="btnDel">Del</button>
<button class="tb" style="background:#3a1520;border-color:#8a3050;color:#ff8fa3" id="btnClear">Limpar</button>
<button class="tb" id="btnAuto">Auto</button>
</div>
<div class="workspace" id="diag-workspace">
<div class="canvas-wrap" id="cw"><div class="canvas" id="cv"></div>
<svg class="wires" id="wires"><defs>
<marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#5b6be0"/></marker>
<marker id="ar2" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#e8a035"/></marker>
</defs></svg></div>
<div class="panel" id="panel">
<div class="panel-section"><h4>Sistema</h4>
<div style="display:grid;grid-template-columns:1fr auto;gap:4px 8px;font-size:12px">
<span style="color:var(--txm)">Blocos:</span><span id="pB" style="font-weight:600">0</span>
<span style="color:var(--txm)">Conexoes:</span><span id="pE" style="font-weight:600">0</span>
</div></div>
<div class="panel-section"><h4>Parametros</h4><div id="pA"><span class="hint">Selecione um bloco.</span></div></div>
<div class="panel-section"><h4>Dicas</h4><div class="hint">
<b>Conectar:</b> porta <span style="color:var(--blu)">azul</span> &rarr; porta <span style="color:var(--grn)">verde</span><br>
<b>Calcular:</b> botao verde abaixo<br><b>Mobile:</b> toque e arraste!
</div></div></div></div>

<div class="mode-bar">
<button class="mode-btn active" id="modeDiag" onclick="setMode('diag')">&#9638; Diagrama de Blocos</button>
<button class="mode-btn" id="modeMan" onclick="setMode('manual')">&#9998; Entrada Manual</button>
</div>

<div class="manual-sec" id="manualSec">
<div class="man-tabs">
<button class="man-tab active" id="subDirect" onclick="setSubMode('direct')">T(s) Direta</button>
<button class="man-tab" id="subClosed" onclick="setSubMode('closed')">Malha Fechada G/(1+GH)</button>
<button class="man-tab" id="subOpen" onclick="setSubMode('open')">Malha Aberta G*H</button>
</div>
<div id="manDirect">
<h4>Funcao de Transferencia T(s) = Num / Den</h4>
<div class="man-row">
<div class="pg"><label>Numerador</label><input id="manNum" value="1" placeholder="ex: s+1"></div>
<div class="pg"><label>Denominador</label><input id="manDen" value="s^2+2s+1" placeholder="ex: s^2+2s+1"></div>
</div>
<div class="man-hint">Formato: <code>s^2+3s+1</code> ou <code>2s^3+s+5</code>. Use <code>^</code> para potencias.</div>
</div>
<div id="manClosed" style="display:none">
<h4>Malha Fechada: T(s) = G(s) / (1 + G(s)&middot;H(s))</h4>
<div class="man-row">
<div class="pg"><label>G(s) Numerador</label><input id="manGN" value="10" placeholder="ex: 10"></div>
<div class="pg"><label>G(s) Denominador</label><input id="manGD" value="s^2+3s+1" placeholder="ex: s^2+3s+1"></div>
</div>
<div class="man-row">
<div class="pg"><label>H(s) Numerador</label><input id="manHN" value="1" placeholder="ex: 1"></div>
<div class="pg"><label>H(s) Denominador</label><input id="manHD" value="1" placeholder="ex: s+1"></div>
</div>
<div class="man-hint">Calcula <code>T(s) = G/(1+GH)</code> com realimentacao unitaria quando H=1.</div>
</div>
<div id="manOpen" style="display:none">
<h4>Malha Aberta: L(s) = G(s) &middot; H(s)</h4>
<div class="man-row">
<div class="pg"><label>G(s) Numerador</label><input id="manOGN" value="10" placeholder="ex: 10"></div>
<div class="pg"><label>G(s) Denominador</label><input id="manOGD" value="s^2+3s+1" placeholder="ex: s^2+3s+1"></div>
</div>
<div class="man-row">
<div class="pg"><label>H(s) Numerador</label><input id="manOHN" value="1" placeholder="ex: 1"></div>
<div class="pg"><label>H(s) Denominador</label><input id="manOHD" value="1" placeholder="ex: 1"></div>
</div>
<div class="man-hint">Analisa a funcao de transferencia de malha aberta <code>L(s) = G*H</code>.</div>
</div>
</div>

<div class="calc-bar"><button id="btnCalcMain" onclick="onCalc()">&#9654; CALCULAR DIAGRAMA</button></div>

<div class="results" id="res"><div class="results-hdr"><h3>Resultados</h3>
<button onclick="document.getElementById('res').classList.remove('vis')" style="background:none;border:1px solid var(--bd);color:var(--txm);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px">Fechar</button>
</div><div class="rbody" id="rb"></div></div></div>

<script>
/* ===== POLY MATH ===== */
function pTrim(p){var a=p.slice();while(a.length>1&&Math.abs(a[a.length-1])<1e-14)a.pop();return a.length?a:[0]}
function pAdd(a,b){var r=[];for(var i=0;i<Math.max(a.length,b.length);i++)r.push((a[i]||0)+(b[i]||0));return pTrim(r)}
function pSub(a,b){var r=[];for(var i=0;i<Math.max(a.length,b.length);i++)r.push((a[i]||0)-(b[i]||0));return pTrim(r)}
function pMul(a,b){var r=[];for(var i=0;i<a.length+b.length-1;i++)r.push(0);for(var i=0;i<a.length;i++)for(var j=0;j<b.length;j++)r[i+j]+=a[i]*b[j];return pTrim(r)}
function pScl(a,k){return a.map(function(c){return c*k})}
function pEv(p,x){var r=0;for(var i=p.length-1;i>=0;i--)r=r*x+p[i];return r}
function cMul(a,b){return{r:a.r*b.r-a.i*b.i,i:a.r*b.i+a.i*b.r}}
function cDiv(a,b){var d=b.r*b.r+b.i*b.i;return d<1e-30?{r:0,i:0}:{r:(a.r*b.r+a.i*b.i)/d,i:(a.i*b.r-a.r*b.i)/d}}
function cAbs(a){return Math.sqrt(a.r*a.r+a.i*a.i)}
function cEvP(p,z){var r={r:0,i:0};for(var i=p.length-1;i>=0;i--){r=cMul(r,z);r.r+=p[i]}return r}

/* ===== PF (poly fraction) ===== */
function pfC(c){return{n:[c],d:[1]}}
function pfZ(a){var t=pTrim(a.n);return t.length===1&&Math.abs(t[0])<1e-14}
function pfSub(a,b){return{n:pSub(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)}}
function pfMul(a,b){return{n:pMul(a.n,b.n),d:pMul(a.d,b.d)}}
function pfDiv(a,b){return{n:pMul(a.n,b.d),d:pMul(a.d,b.n)}}

/* ===== PARSER ===== */
function parseP(s){
  s=(s||"").replace(/\s+/g,"");if(!s)return[0];
  if(!/s/i.test(s)){var v=parseFloat(s);return[isNaN(v)?0:v]}
  var n="";for(var i=0;i<s.length;i++){if(s[i]==="-"&&i>0&&s[i-1]!=="^"&&s[i-1]!=="+"&&s[i-1]!=="-"&&s[i-1]!=="*")n+="+";n+=s[i]}
  var ts=n.split("+").filter(function(t){return t}),co={};
  ts.forEach(function(t){t=t.replace(/\*/g,"");var m;
    if(m=t.match(/^([+-]?\d*\.?\d*)s\^(\d+)$/)){var c=m[1]===""||m[1]==="+"?1:m[1]==="-"?-1:parseFloat(m[1]);co[parseInt(m[2])]=(co[parseInt(m[2])]||0)+c}
    else if(m=t.match(/^([+-]?\d*\.?\d*)s$/)){var c=m[1]===""||m[1]==="+"?1:m[1]==="-"?-1:parseFloat(m[1]);co[1]=(co[1]||0)+c}
    else{var c=parseFloat(t);if(!isNaN(c))co[0]=(co[0]||0)+c}});
  var ks=Object.keys(co).map(Number);if(!ks.length)return[0];var mx=Math.max.apply(null,ks),r=[];
  for(var i=0;i<=mx;i++)r.push(co[i]||0);return pTrim(r)}

/* ===== FORMAT ===== */
function fN(n){if(Math.abs(n-Math.round(n))<1e-8)return Math.round(n).toString();return n.toFixed(4).replace(/0+$/,"").replace(/\.$/,"")}
function fP(c){c=pTrim(c);var ts=[];for(var i=c.length-1;i>=0;i--){var v=c[i];if(Math.abs(v)<1e-10)continue;var t;
  if(i===0)t=fN(v);else if(i===1){t=Math.abs(v-1)<1e-10?"s":Math.abs(v+1)<1e-10?"-s":fN(v)+"s"}
  else{t=Math.abs(v-1)<1e-10?"s^"+i:Math.abs(v+1)<1e-10?"-s^"+i:fN(v)+"s^"+i}ts.push(t)}
  if(!ts.length)return"0";var s=ts[0];for(var i=1;i<ts.length;i++)s+=ts[i][0]==="-"?" - "+ts[i].slice(1):" + "+ts[i];return s}

/* ===== BLOCK TF ===== */
function bTF(nd){var p=nd.params||{},t=nd.type;
  if(t==="tf"||t==="sensor")return{n:parseP(p.num||"1"),d:parseP(p.den||"1")};
  if(t==="gain")return pfC(parseFloat(p.k)||1);
  if(t==="int")return{n:[1],d:[0,1]};if(t==="der")return{n:[0,1],d:[1]};
  if(t==="pid"){var kp=parseFloat(p.kp)||0,ki=parseFloat(p.ki)||0,kd=parseFloat(p.kd)||0;
    if(ki===0&&kd===0)return pfC(kp||1);if(ki===0)return{n:[kp,kd],d:[1]};return{n:[ki,kp,kd],d:[0,1]}}
  return pfC(1)}

/* ===== SOLVER ===== */
function solve(nodes,edges){
  if(!nodes.length)return{e:"Adicione blocos."};
  var inp=null,out=null;nodes.forEach(function(n){if(n.type==="input")inp=n;if(n.type==="output")out=n});
  if(!inp)return{e:"Adicione Entrada R(s)."};if(!out)return{e:"Adicione Saida Y(s)."};if(!edges.length)return{e:"Conecte os blocos."};
  var N=nodes.length,ix={};nodes.forEach(function(n,i){ix[n.id]=i});
  var A=[],b=[];for(var i=0;i<N;i++){A.push([]);for(var j=0;j<N;j++)A[i].push(pfC(0));b.push(pfC(0))}
  for(var i=0;i<N;i++){var nd=nodes[i];A[i][i]=pfC(1);
    if(nd.type==="input"){b[i]=pfC(1);continue}
    var inc=edges.filter(function(e){return e.dst===nd.id});
    if(nd.type==="sum"){var sg=((nd.params||{}).signs||"+ -").trim().split(/\s+/);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;var pi=parseInt((e.dstPort||"in0").replace("in",""))||0;
        var sign=sg[pi]==="-"?-1:1;A[i][si]=pfSub(A[i][si],pfC(sign))})}
    else{var tf=bTF(nd);inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;A[i][si]=pfSub(A[i][si],tf)})}}
  for(var c=0;c<N;c++){var pv=-1;for(var r=c;r<N;r++)if(!pfZ(A[r][c])){pv=r;break}
    if(pv<0)return{e:"Sistema singular."};if(pv!==c){var t=A[c];A[c]=A[pv];A[pv]=t;var t2=b[c];b[c]=b[pv];b[pv]=t2}
    for(var r=c+1;r<N;r++){if(pfZ(A[r][c]))continue;var f=pfDiv(A[r][c],A[c][c]);
      for(var j=c;j<N;j++)A[r][j]=pfSub(A[r][j],pfMul(f,A[c][j]));b[r]=pfSub(b[r],pfMul(f,b[c]))}}
  var x=[];for(var i=0;i<N;i++)x.push(pfC(0));
  for(var i=N-1;i>=0;i--){var s=b[i];for(var j=i+1;j<N;j++)s=pfSub(s,pfMul(A[i][j],x[j]));x[i]=pfDiv(s,A[i][i])}
  var oi=ix[out.id],tf={n:pTrim(x[oi].n),d:pTrim(x[oi].d)};
  if(tf.d[tf.d.length-1]<0){tf.n=pScl(tf.n,-1);tf.d=pScl(tf.d,-1)}
  var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
  return{tf:tf}}

/* ===== ROOTS ===== */
function roots(poly){poly=pTrim(poly);var n=poly.length-1;if(n<=0)return[];
  var lc=poly[n],p=poly.map(function(c){return c/lc});
  if(n===1)return[{r:-p[0],i:0}];
  if(n===2){var bb=p[1],cc=p[0],d=bb*bb-4*cc;
    if(d>=0)return[{r:(-bb+Math.sqrt(d))/2,i:0},{r:(-bb-Math.sqrt(d))/2,i:0}];
    return[{r:-bb/2,i:Math.sqrt(-d)/2},{r:-bb/2,i:-Math.sqrt(-d)/2}]}
  var rs=[];for(var i=0;i<n;i++){var a=2*Math.PI*i/n+.4;rs.push({r:(1+.5*i/n)*Math.cos(a),i:(1+.5*i/n)*Math.sin(a)})}
  for(var it=0;it<1500;it++){var mx=0;for(var i=0;i<n;i++){var v=cEvP(p,rs[i]),pr={r:1,i:0};
    for(var j=0;j<n;j++)if(j!==i)pr=cMul(pr,{r:rs[i].r-rs[j].r,i:rs[i].i-rs[j].i});
    if(cAbs(pr)<1e-30)continue;var dl=cDiv(v,pr);rs[i].r-=dl.r;rs[i].i-=dl.i;mx=Math.max(mx,cAbs(dl))}if(mx<1e-12)break}
  rs.forEach(function(r){if(Math.abs(r.i)<1e-8)r.i=0});return rs}

/* ===== STEP RESPONSE (Stehfest) ===== */
function fact(n){var r=1;for(var i=2;i<=n;i++)r*=i;return r}
var SN=14,SV=[];(function(){for(var i=1;i<=SN;i++){var s=0,k0=Math.floor((i+1)/2),k1=Math.min(i,SN/2);
  for(var k=k0;k<=k1;k++)s+=Math.pow(k,SN/2)*fact(2*k)/(fact(SN/2-k)*fact(k)*fact(k-1)*fact(i-k)*fact(2*k-i));
  SV.push(Math.pow(-1,SN/2+i)*s)}})();
function stepResp(tf,tMax,nP){var ln2=Math.LN2,ts=[],ys=[];
  for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;
    for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv)*sv;if(Math.abs(dv)<1e-30)continue;s+=SV[j]*nv/dv}
    ys.push(s*ln2/t)}return{t:ts,y:ys}}

/* ===== BODE ===== */
function bode(tf,wMin,wMax,nP){var fs=[],ms=[],ps=[],lm=Math.log10(wMin),lx=Math.log10(wMax);
  for(var i=0;i<nP;i++){var w=Math.pow(10,lm+i*(lx-lm)/(nP-1));fs.push(w);
    var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc),mg=cAbs(T);
    ms.push(mg>1e-30?20*Math.log10(mg):-600);ps.push(Math.atan2(T.i,T.r)*180/Math.PI)}
  for(var i=1;i<ps.length;i++){while(ps[i]-ps[i-1]>180)ps[i]-=360;while(ps[i]-ps[i-1]<-180)ps[i]+=360}
  return{w:fs,m:ms,p:ps}}

/* ===== AUTO RANGES ===== */
function autoT(tf){var ps=roots(tf.d);if(!ps.length)return 10;var m=Infinity;
  ps.forEach(function(p){if(Math.abs(p.r)>1e-6)m=Math.min(m,Math.abs(p.r))});return m===Infinity?10:Math.min(100,Math.max(2,7/m))}
function autoW(tf){var ps=roots(tf.d).concat(roots(tf.n)),fs=ps.map(function(p){return Math.sqrt(p.r*p.r+p.i*p.i)}).filter(function(f){return f>1e-6});
  if(!fs.length)return{a:.01,b:1000};return{a:Math.max(.001,Math.min.apply(null,fs)/20),b:Math.min(1e5,Math.max.apply(null,fs)*20)}}

/* ===== PERF ===== */
function perf(t,y){if(!y.length)return{};var l=y.slice(Math.floor(y.length*.9)),yf=l.reduce(function(a,b){return a+b},0)/l.length;
  var ym=Math.max.apply(null,y),os=Math.abs(yf)>1e-6?Math.max(0,(ym-yf)/Math.abs(yf)*100):0;
  var t10=null,t90=null;if(yf>0)for(var i=0;i<y.length;i++){if(t10===null&&y[i]>=yf*.1)t10=t[i];if(t90===null&&y[i]>=yf*.9){t90=t[i];break}}
  var tr=t10!==null&&t90!==null?t90-t10:NaN,ts2=NaN;
  if(Math.abs(yf)>1e-6)for(var i=y.length-1;i>=0;i--)if(Math.abs(y[i]-yf)>.02*Math.abs(yf)){ts2=i<y.length-1?t[i+1]:t[i];break}
  return{"Valor Final":fN(yf),"Sobressinal":fN(os)+"%","T. Subida":isNaN(tr)?"N/A":fN(tr)+"s","T. Acomod.":isNaN(ts2)?"N/A":fN(ts2)+"s","Pico":fN(ym)}}

/* ===== CANVAS CHART ===== */
function chart(id,xD,yD,xL,yL,col,logX){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=280,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
  var vY=yD.filter(function(v){return isFinite(v)});if(!vY.length)return;
  var xA=logX?xD.map(function(x){return Math.log10(x)}):xD;
  var x0=Math.min.apply(null,xA),x1=Math.max.apply(null,xA),y0=Math.min.apply(null,vY),y1=Math.max.apply(null,vY);
  var yp=(y1-y0)*.1||1;y0-=yp;y1+=yp;
  function mX(x){return mg.l+((logX?Math.log10(x):x)-x0)/(x1-x0)*pw}
  function mY(y){return mg.t+ph-(y-y0)/(y1-y0)*ph}
  ctx.fillStyle="#0e1117";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#252840";ctx.lineWidth=.5;
  for(var i=0;i<=5;i++){var gy=mg.t+ph*i/5;ctx.beginPath();ctx.moveTo(mg.l,gy);ctx.lineTo(w-mg.r,gy);ctx.stroke();
    ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="right";ctx.fillText(fN(y1-(y1-y0)*i/5),mg.l-5,gy+4)}
  ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();var st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(yD[i])){st=false;continue}
    var px=mX(xD[i]),py=Math.max(mg.t,Math.min(mg.t+ph,mY(yD[i])));if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}

/* ===== RESULTS ===== */
function fC(c){if(Math.abs(c.i)<1e-8)return fN(c.r);return fN(c.r)+(c.i>=0?" + ":" - ")+fN(Math.abs(c.i))+"j"}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}

function showRes(tf){var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");rb.innerHTML="";
  var ns=fP(tf.n),ds=fP(tf.d);
  rb.innerHTML+='<div class="rcard"><h4>T(s)</h4><div class="tf-disp"><div style="display:inline-block;text-align:center"><div style="padding:0 8px">'+esc(ns)+'</div><div style="border-top:2px solid var(--acc);padding:4px 8px 0">'+esc(ds)+'</div></div></div></div>';
  var ps=roots(tf.d),zs=roots(tf.n),stb=ps.every(function(p){return p.r<1e-6});
  var h='<div class="rcard"><h4>Polos e Zeros</h4><div style="display:flex;gap:20px;flex-wrap:wrap"><div><b style="color:var(--red)">Polos:</b><div class="pzl">';
  ps.forEach(function(p){h+='<div style="color:var(--red)">'+fC(p)+'</div>'});if(!ps.length)h+="<div>-</div>";
  h+='</div></div><div><b style="color:var(--blu)">Zeros:</b><div class="pzl">';
  zs.forEach(function(z){h+='<div style="color:var(--blu)">'+fC(z)+'</div>'});if(!zs.length)h+="<div>-</div>";
  h+='</div></div></div><div style="margin-top:8px;padding:6px 10px;border-radius:6px;font-weight:700;font-size:13px;'+(stb?'background:#16382a;color:#34d399">ESTAVEL':'background:#3a1520;color:#f87171">INSTAVEL')+'</div></div>';
  rb.innerHTML+=h;
  var tM=autoT(tf),sr=stepResp(tf,tM,400);
  rb.innerHTML+='<div class="rcard"><h4>Resposta ao Degrau</h4><div><canvas id="cStep"></canvas></div></div>';
  chart("cStep",sr.t,sr.y,"Tempo (s)","Amplitude","#5b6be0",false);
  var pf=perf(sr.t,sr.y);h='<div class="rcard"><h4>Desempenho</h4><div class="mgrid">';
  Object.keys(pf).forEach(function(k){h+='<div class="mbox"><div class="ml">'+k+'</div><div class="mv">'+pf[k]+'</div></div>'});
  rb.innerHTML+=h+'</div></div>';
  var wr=autoW(tf),bd=bode(tf,wr.a,wr.b,400);
  rb.innerHTML+='<div class="rcard"><h4>Bode - Magnitude</h4><div><canvas id="cBM"></canvas></div></div>';
  rb.innerHTML+='<div class="rcard"><h4>Bode - Fase</h4><div><canvas id="cBP"></canvas></div></div>';
  chart("cBM",bd.w,bd.m,"w (rad/s)","dB","#60a5fa",true);chart("cBP",bd.w,bd.p,"w (rad/s)","graus","#f472b6",true);
  rd.scrollIntoView({behavior:"smooth"})}

/* ===== MODE SWITCHING ===== */
var curMode="diag",curSubMode="direct";
function setMode(m){curMode=m;
  document.getElementById("modeDiag").classList.toggle("active",m==="diag");
  document.getElementById("modeMan").classList.toggle("active",m==="manual");
  document.getElementById("diag-workspace").style.display=m==="diag"?"flex":"none";
  document.getElementById("diag-toolbar").style.display=m==="diag"?"flex":"none";
  document.getElementById("manualSec").classList.toggle("vis",m==="manual");
  document.getElementById("btnCalcMain").innerHTML=m==="diag"?"&#9654; CALCULAR DIAGRAMA":"&#9654; CALCULAR T(s)"}
function setSubMode(m){curSubMode=m;
  document.getElementById("subDirect").classList.toggle("active",m==="direct");
  document.getElementById("subClosed").classList.toggle("active",m==="closed");
  document.getElementById("subOpen").classList.toggle("active",m==="open");
  document.getElementById("manDirect").style.display=m==="direct"?"block":"none";
  document.getElementById("manClosed").style.display=m==="closed"?"block":"none";
  document.getElementById("manOpen").style.display=m==="open"?"block":"none"}

function onCalc(){
  if(curMode==="manual"){
    var tf;
    if(curSubMode==="direct"){
      tf={n:parseP(document.getElementById("manNum").value),d:parseP(document.getElementById("manDen").value)};
    } else if(curSubMode==="closed"){
      var gn=parseP(document.getElementById("manGN").value),gd=parseP(document.getElementById("manGD").value);
      var hn=parseP(document.getElementById("manHN").value),hd=parseP(document.getElementById("manHD").value);
      tf={n:pMul(gn,hd),d:pAdd(pMul(gd,hd),pMul(gn,hn))};
    } else {
      var gn=parseP(document.getElementById("manOGN").value),gd=parseP(document.getElementById("manOGD").value);
      var hn=parseP(document.getElementById("manOHN").value),hd=parseP(document.getElementById("manOHD").value);
      tf={n:pMul(gn,hn),d:pMul(gd,hd)};
    }
    var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
    showRes(tf);return;
  }
  var r=solve(model.nodes,model.edges);
  if(r.e){var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");rb.innerHTML='<div class="ebox">'+esc(r.e)+'</div>';rd.scrollIntoView({behavior:"smooth"});return}
  showRes(r.tf)}

/* ===== EDITOR STATE ===== */
var model={nodes:[],edges:[]},selId=null,dragSt=null,conSt=null;
var cw=document.getElementById("cw"),cv=document.getElementById("cv"),wSvg=document.getElementById("wires");
function nxtId(){var m=0;model.nodes.forEach(function(n){var v=parseInt(n.id.replace("n",""))||0;if(v>m)m=v});return"n"+(m+1)}
function ptr(e){if(e.touches&&e.touches.length)return{x:e.touches[0].clientX,y:e.touches[0].clientY};if(e.changedTouches&&e.changedTouches.length)return{x:e.changedTouches[0].clientX,y:e.changedTouches[0].clientY};return{x:e.clientX,y:e.clientY}}
var BL={tf:"Funcao Transf.",gain:"Ganho",sum:"Somador",int:"Integrador",der:"Derivador",pid:"PID",sensor:"Sensor",input:"Entrada",output:"Saida",branch:"Ramificacao"};
function dPar(t){if(t==="tf")return{num:"1",den:"s+1"};if(t==="gain")return{k:"1"};if(t==="sum")return{signs:"+ -"};if(t==="pid")return{kp:"1",ki:"0",kd:"0"};if(t==="sensor")return{num:"1",den:"1"};if(t==="input")return{label:"R(s)"};if(t==="output")return{label:"Y(s)"};return{}}
function gPC(t,p){if(t==="input")return{i:[],o:[{id:"out0"}]};if(t==="output")return{i:[{id:"in0"}],o:[]};if(t==="branch")return{i:[{id:"in0"}],o:[{id:"out0"},{id:"out1"}]};
  if(t==="sum"){var sg=(p&&p.signs?p.signs:"+ -").trim().split(/\s+/);return{i:sg.map(function(s,i){return{id:"in"+i,sign:s}}),o:[{id:"out0"}]}}return{i:[{id:"in0"}],o:[{id:"out0"}]}}
function bTxt(n){var p=n.params||{};if(n.type==="tf")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="gain")return"K="+(p.k||"1");if(n.type==="pid")return'<div class="block-tf-disp">Kp='+(p.kp||"0")+" Ki="+(p.ki||"0")+" Kd="+(p.kd||"0")+'</div>';
  if(n.type==="sensor")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="input")return p.label||"R(s)";if(n.type==="output")return p.label||"Y(s)";if(n.type==="sum")return"\u03a3";if(n.type==="int")return"1/s";if(n.type==="der")return"s";return""}
function addB(t){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:dPar(t)};
  if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
function delSel(){if(!selId)return;model.edges=model.edges.filter(function(e){return e.src!==selId&&e.dst!==selId});model.nodes=model.nodes.filter(function(n){return n.id!==selId});selId=null;conSt=null;render()}
function clrAll(){model={nodes:[],edges:[]};selId=null;conSt=null;render();document.getElementById("res").classList.remove("vis")}
function autoLay(){if(!model.nodes.length)return;var lv={},aj={};model.edges.forEach(function(e){if(!aj[e.src])aj[e.src]=[];aj[e.src].push(e.dst)});
  var ins=model.nodes.filter(function(n){return n.type==="input"}).map(function(n){return n.id});if(!ins.length)ins=[model.nodes[0].id];
  var q=ins.map(function(id){return{id:id,l:0}}),vis={};ins.forEach(function(id){vis[id]=true});
  while(q.length){var c=q.shift();lv[c.id]=c.l;(aj[c.id]||[]).forEach(function(n){if(!vis[n]){vis[n]=true;q.push({id:n,l:c.l+1})}})}
  model.nodes.forEach(function(n){if(!(n.id in lv))lv[n.id]=3});var bl={};Object.keys(lv).forEach(function(id){var l=lv[id];if(!bl[l])bl[l]=[];bl[l].push(id)});
  Object.keys(bl).forEach(function(l){bl[l].forEach(function(id,i){var n=model.nodes.find(function(nd){return nd.id===id});if(n){n.x=50+parseInt(l)*170;n.y=50+i*100}})});render()}
function setSel(id){selId=id;document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===id)});updP()}
function pSty(t,k,i){if(t==="sum"){if(k==="in"&&i===0)return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="in"&&i===1)return"left:50%;bottom:-7px;transform:translateX(-50%)";if(k==="in"&&i>=2)return"left:-7px;top:"+(8+i*14)+"px";return"right:-7px;top:50%;transform:translateY(-50%)"}
  if(t==="branch"){if(k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="out"&&i===0)return"right:-7px;top:50%;transform:translateY(-50%)";return"left:50%;bottom:-7px;transform:translateX(-50%)"}
  if(t==="input"&&k==="out")return"right:-7px;top:50%;transform:translateY(-50%)";if(t==="output"&&k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";
  return k==="in"?"left:-7px;top:50%;transform:translateY(-50%)":"right:-7px;top:50%;transform:translateY(-50%)"}
function render(){cv.querySelectorAll(".block").forEach(function(el){el.remove()});
  model.nodes.forEach(function(n){var el=document.createElement("div");el.className="block block-"+n.type;el.dataset.id=n.id;el.style.left=n.x+"px";el.style.top=n.y+"px";
    var pc=gPC(n.type,n.params),h="";
    if(n.type==="sum")h='<div class="block-body">\u03a3</div>';else if(n.type==="branch")h="";
    else h='<div class="block-header">'+BL[n.type]+'</div><div class="block-body">'+bTxt(n)+'</div>';
    pc.i.forEach(function(p,i){h+='<div class="port port-in" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="in" style="'+pSty(n.type,"in",i)+'"></div>';
      if(n.type==="sum"&&p.sign)h+='<span class="sign-label" style="'+(i===0?"left:-20px;top:50%;transform:translateY(-50%)":i===1?"left:50%;bottom:-18px;transform:translateX(-50%)":"left:-20px;top:"+(8+i*14)+"px")+'">'+p.sign+'</span>'});
    pc.o.forEach(function(p,i){h+='<div class="port port-out" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="out" style="'+pSty(n.type,"out",i)+'"></div>'});
    el.innerHTML=h;cv.appendChild(el);
    el.addEventListener("mousedown",function(ev){onBD(ev,n.id)});el.addEventListener("touchstart",function(ev){onBD(ev,n.id)},{passive:false});
    el.querySelectorAll(".port").forEach(function(pl){pl.addEventListener("mousedown",function(ev){ev.stopPropagation();onPC(ev,pl)});
      pl.addEventListener("touchstart",function(ev){ev.stopPropagation();ev.preventDefault();onPC(ev,pl)},{passive:false})})});
  document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===selId)});rW();updP()}
function rW(){wSvg.querySelectorAll("path").forEach(function(p){p.remove()});var wr=cw.getBoundingClientRect();wSvg.setAttribute("width",wr.width);wSvg.setAttribute("height",wr.height);
  model.edges.forEach(function(e,idx){var a=gPP(e.src,e.srcPort||"out0"),b=gPP(e.dst,e.dstPort||"in0");if(!a||!b)return;
    var p=document.createElementNS("http://www.w3.org/2000/svg","path"),dx=Math.max(50,Math.abs(b.x-a.x)*.4);
    if(b.x<a.x-20){var mY=Math.max(a.y,b.y)+60;p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+40)+" "+a.y+","+(a.x+40)+" "+mY+","+((a.x+b.x)/2)+" "+mY+" S "+(b.x-40)+" "+b.y+","+b.x+" "+b.y);
      p.setAttribute("stroke","#e8a035");p.setAttribute("stroke-dasharray","8 4");p.setAttribute("marker-end","url(#ar2)")}
    else{p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+dx)+" "+a.y+","+(b.x-dx)+" "+b.y+","+b.x+" "+b.y);p.setAttribute("stroke","#5b6be0");p.setAttribute("marker-end","url(#ar)")}
    p.setAttribute("fill","none");p.setAttribute("stroke-width","2.5");p.style.pointerEvents="visibleStroke";p.style.cursor="pointer";
    p.addEventListener("click",function(){model.edges.splice(idx,1);render()});p.addEventListener("touchend",function(ev){ev.preventDefault();model.edges.splice(idx,1);render()});
    wSvg.appendChild(p)})}
function gPP(nid,pid){var bl=document.querySelector('.block[data-id="'+nid+'"]');if(!bl)return null;var pl=bl.querySelector('.port[data-port="'+pid+'"]');
  if(!pl){pl=bl.querySelector(".port.port-"+(pid.indexOf("out")===0?"out":"in"))}if(!pl)return null;var cr=cw.getBoundingClientRect(),pr=pl.getBoundingClientRect();
  return{x:pr.left+pr.width/2-cr.left,y:pr.top+pr.height/2-cr.top}}
function onBD(e,id){if(e.target.classList.contains("port"))return;if(e.button===2)return;e.preventDefault();setSel(id);var p=ptr(e),el=document.querySelector('.block[data-id="'+id+'"]');if(!el)return;var r=el.getBoundingClientRect();dragSt={id:id,ox:p.x-r.left,oy:p.y-r.top}}
function onPM(e){if(!dragSt)return;e.preventDefault();var p=ptr(e),wr=cw.getBoundingClientRect(),el=document.querySelector('.block[data-id="'+dragSt.id+'"]');if(!el)return;
  var nx=Math.max(0,Math.min(wr.width-el.offsetWidth,p.x-wr.left-dragSt.ox)),ny=Math.max(0,Math.min(wr.height-el.offsetHeight,p.y-wr.top-dragSt.oy));
  el.style.left=nx+"px";el.style.top=ny+"px";var nd=model.nodes.find(function(n){return n.id===dragSt.id});if(nd){nd.x=nx;nd.y=ny}rW()}
document.addEventListener("mousemove",onPM);document.addEventListener("mouseup",function(){dragSt=null});
document.addEventListener("touchmove",onPM,{passive:false});document.addEventListener("touchend",function(){dragSt=null});
cv.addEventListener("mousedown",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}});
cv.addEventListener("touchstart",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}},{passive:false});
function onPC(e,pl){e.preventDefault();var nid=pl.dataset.nid,pid=pl.dataset.port,kind=pl.dataset.kind;setSel(nid);
  if(conSt){if(conSt.nid===nid){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}
    var sI,sP,dI,dP;if(conSt.kind==="out"&&kind==="in"){sI=conSt.nid;sP=conSt.pid;dI=nid;dP=pid}
    else if(conSt.kind==="in"&&kind==="out"){sI=nid;sP=pid;dI=conSt.nid;dP=conSt.pid}
    else{conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}
    if(!model.edges.some(function(ed){return ed.src===sI&&ed.srcPort===sP&&ed.dst===dI&&ed.dstPort===dP}))
      model.edges.push({id:"e"+Math.random().toString(36).slice(2),src:sI,srcPort:sP,dst:dI,dstPort:dP});
    conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});render()}
  else{conSt={nid:nid,pid:pid,kind:kind};pl.classList.add("active")}}
function updP(){document.getElementById("pB").textContent=model.nodes.length;document.getElementById("pE").textContent=model.edges.length;
  var pa=document.getElementById("pA");if(!selId){pa.innerHTML='<span class="hint">Selecione um bloco.</span>';return}
  var nd=model.nodes.find(function(n){return n.id===selId});if(!nd){pa.innerHTML="";return}var p=nd.params||{};
  var h='<div style="font-size:11px;color:var(--txm);margin-bottom:6px">'+BL[nd.type]+' <b style="color:var(--tx)">'+nd.id+'</b></div>';
  if(nd.type==="tf"||nd.type==="sensor"){h+=pI("Num","num",p.num||"1");h+=pI("Den","den",p.den||"1")}
  else if(nd.type==="gain")h+=pI("K","k",p.k||"1");
  else if(nd.type==="sum")h+=pI("Sinais","signs",p.signs||"+ -");
  else if(nd.type==="pid"){h+=pI("Kp","kp",p.kp||"1");h+=pI("Ki","ki",p.ki||"0");h+=pI("Kd","kd",p.kd||"0")}
  else if(nd.type==="input"||nd.type==="output")h+=pI("Label","label",p.label||"");
  pa.innerHTML=h;pa.querySelectorAll("input[data-key]").forEach(function(inp){inp.addEventListener("input",function(){nd.params[inp.dataset.key]=inp.value;
    if(inp.dataset.key==="signs")render();else{var bl=document.querySelector('.block[data-id="'+nd.id+'"] .block-body');if(bl)bl.innerHTML=bTxt(nd)}})})}
function pI(l,k,v){return'<div class="pg"><label>'+l+'</label><input data-key="'+k+'" value="'+esc(String(v))+'"></div>'}
document.querySelectorAll(".tb[data-add]").forEach(function(b){b.addEventListener("click",function(){addB(b.dataset.add)})});
document.getElementById("btnDel").addEventListener("click",delSel);document.getElementById("btnClear").addEventListener("click",clrAll);document.getElementById("btnAuto").addEventListener("click",autoLay);
document.addEventListener("keydown",function(e){if(e.target.tagName==="INPUT")return;if(e.key==="Delete"||e.key==="Backspace")delSel();if(e.key==="Escape"){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")})}});
render();new ResizeObserver(function(){rW()}).observe(cw);
</script></body></html>'''


# =====================================================
# APLICACAO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("Modelagem e Analise de Sistemas de Controle")

    inicializar_blocos()

    # Sidebar: Modo de trabalho
    st.sidebar.header("Modo de Trabalho")
    modo = st.sidebar.radio(
        "Escolha o modo:",
        ['Classico (Lista)', 'Editor Visual / Diagrama de Blocos'],
        index=0 if st.session_state.modo_editor == 'classico' else 1
    )
    st.session_state.modo_editor = 'visual' if 'Visual' in modo else 'classico'

    # ══════════════════════════════════════════
    #  MODO EDITOR VISUAL
    # ══════════════════════════════════════════
    if st.session_state.modo_editor == 'visual':
        html_content = _load_visual_editor_html()
        components.html(html_content, height=1200, scrolling=True)

        with st.sidebar:
            st.markdown("### Como usar")
            st.markdown("""
            **Diagrama de Blocos:**
            1. Adicione blocos pela barra superior
            2. Arraste para posicionar
            3. Conecte: porta azul (saida) -> porta verde (entrada)
            4. Edite parametros no painel lateral
            5. Clique **CALCULAR** para ver resultados

            **Entrada Manual:**
            1. Clique em "Entrada Manual"
            2. Escolha: T(s) Direta, Malha Fechada ou Malha Aberta
            3. Digite numerador e denominador
            4. Clique **CALCULAR T(s)**
            """)
        return

    # ══════════════════════════════════════════
    #  MODO CLASSICO
    # ══════════════════════════════════════════

    with st.sidebar:
        st.header("Adicionar Blocos")
        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("Adicionar"):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            if sucesso:
                st.success(mensagem)
            else:
                st.error(mensagem)

        if not st.session_state.blocos.empty:
            st.header("Excluir Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("Excluir"):
                mensagem = remover_bloco(excluir)
                st.success(mensagem)

        st.header("Configuracoes")
        if st.button("Habilitar Calculo de Erro" if not st.session_state.calculo_erro_habilitado else "Desabilitar Calculo de Erro"):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
            st.rerun()

    if st.session_state.calculo_erro_habilitado:
        st.subheader("Calculo de Erro Estacionario")
        col1, col2 = st.columns(2)
        with col1:
            num_erro = st.text_input("Numerador", value="", key="num_erro")
        with col2:
            den_erro = st.text_input("Denominador", value="", key="den_erro")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Calcular Erro Estacionario"):
                try:
                    G, _ = converter_para_tf(num_erro, den_erro)
                    tipo, Kp, Kv, Ka = constantes_de_erro(G)
                    df = pd.DataFrame([{"Tipo": tipo, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                    st.subheader("Resultado")
                    st.dataframe(
                        df.style.format({
                            "Kp": lambda x: formatar_numero(x),
                            "Kv": lambda x: formatar_numero(x),
                            "Ka": lambda x: formatar_numero(x)
                        }),
                        height=120, use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro: {str(e)}")

        with btn_col2:
            if st.button("Remover Planta", key="remover_planta"):
                if not st.session_state.blocos.empty:
                    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['tipo'] != 'Planta']
                    st.success("Plantas removidas!")
                else:
                    st.warning("Nenhuma planta para remover")
    else:
        st.info("Use o botao 'Habilitar Calculo de Erro' na barra lateral")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Tipo de Sistema")
        tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustavel", value=False)

        if usar_ganho:
            K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1)
            st.info(f"Ganho K: {K:.2f}")
        else:
            K = 1.0

        st.subheader("Analises")
        analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
        analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("Resultados")

        if st.button("Executar Simulacao", use_container_width=True):
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
                    st.info(f"Sistema em Malha Aberta com K = {K:.2f}")
                else:
                    planta_com_ganho = ganho_tf * planta
                    sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                    st.info(f"Sistema em Malha Fechada com K = {K:.2f}")

                for analise in analises:
                    st.markdown(f"### {analise}")

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
                        st.markdown(f"**Z = {Z} -> {'Estavel' if Z == 0 else 'Instavel'}**")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro durante a simulacao: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dica")
    st.sidebar.info("Experimente o Editor Visual para construir sistemas graficamente!")


if __name__ == "__main__":
    main()
