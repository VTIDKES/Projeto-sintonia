# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Com Editor Visual de Diagrama de Blocos (estilo Xcos) - VERSÃO MELHORADA
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
import json
import base64

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
# PROCESSAMENTO DO DIAGRAMA DE BLOCOS
# =====================================================

def processar_diagrama_blocos_xcos(blocos_json):
    """
    Processa o diagrama de blocos do editor Xcos e calcula o sistema equivalente
    """
    try:
        # Decodificar JSON dos blocos
        diagrama = json.loads(blocos_json)
        blocos = diagrama.get('blocos', [])
        conexoes = diagrama.get('conexoes', [])
        
        if not blocos:
            return None, "❌ Nenhum bloco no diagrama"
        
        # Criar dicionário de funções de transferência
        tfs = {}
        blocos_info = {}
        
        for bloco in blocos:
            bloco_id = bloco['id']
            tipo = bloco['tipo']
            config = bloco['config']
            
            blocos_info[bloco_id] = {'tipo': tipo, 'nome': config.get('nome', f'Bloco{bloco_id}')}
            
            if tipo == 'Transferência':
                try:
                    tf, _ = converter_para_tf(config['numerador'], config['denominador'])
                    tfs[bloco_id] = tf
                except Exception as e:
                    return None, f"❌ Erro no bloco {config.get('nome')}: {str(e)}"
                    
            elif tipo == 'Ganho':
                try:
                    K = float(config['valor'])
                    tfs[bloco_id] = TransferFunction([K], [1])
                except:
                    tfs[bloco_id] = TransferFunction([1], [1])
                    
            elif tipo == 'Integrador':
                tfs[bloco_id] = TransferFunction([1], [1, 0])
                
            elif tipo == 'Somador':
                tfs[bloco_id] = TransferFunction([1], [1])
        
        # Se não há conexões, criar sistema em série simples
        if not conexoes:
            if len(blocos) == 1:
                return tfs[blocos[0]['id']], f"✅ Sistema com 1 bloco: {blocos_info[blocos[0]['id']]['nome']}"
            
            # Sistema em série
            sistema_final = tfs[blocos[0]['id']]
            for i in range(1, len(blocos)):
                sistema_final = sistema_final * tfs[blocos[i]['id']]
            
            return sistema_final, f"✅ Sistema com {len(blocos)} blocos em série"
        
        # Processar conexões para criar o sistema
        # Criar grafo de conexões
        grafo = {}
        for conexao in conexoes:
            origem = conexao['origem']
            destino = conexao['destino']
            if origem not in grafo:
                grafo[origem] = []
            grafo[origem].append(destino)
        
        # Encontrar bloco inicial (sem entrada) e final (sem saída)
        todos_ids = set(b['id'] for b in blocos)
        destinos = set(c['destino'] for c in conexoes)
        origens = set(c['origem'] for c in conexoes)
        
        blocos_iniciais = todos_ids - destinos
        blocos_finais = todos_ids - origens
        
        # Calcular sistema em cadeia direta
        if blocos_iniciais and blocos_finais:
            # Simplificação: assumir cadeia linear simples
            # Ordenar blocos pela conexão
            ordem_blocos = []
            visitados = set()
            
            def dfs(bloco_id):
                if bloco_id in visitados:
                    return
                visitados.add(bloco_id)
                ordem_blocos.append(bloco_id)
                if bloco_id in grafo:
                    for proximo in grafo[bloco_id]:
                        dfs(proximo)
            
            for inicial in blocos_iniciais:
                dfs(inicial)
            
            # Multiplicar todas as TFs em série
            if ordem_blocos:
                sistema_final = tfs[ordem_blocos[0]]
                for bloco_id in ordem_blocos[1:]:
                    sistema_final = sistema_final * tfs[bloco_id]
                
                return sistema_final, f"✅ Sistema processado com {len(ordem_blocos)} blocos conectados"
        
        # Fallback: multiplicar todos
        sistema_final = tfs[blocos[0]['id']]
        for i in range(1, len(blocos)):
            sistema_final = sistema_final * tfs[blocos[i]['id']]
        
        return sistema_final, f"✅ Sistema com {len(blocos)} blocos"
        
    except Exception as e:
        return None, f"❌ Erro ao processar diagrama: {str(e)}"

# =====================================================
# GERENCIAMENTO DE ESTADO
# =====================================================

def inicializar_estado():
    """Inicializa todas as variáveis de estado"""
    if 'blocos_xcos' not in st.session_state:
        st.session_state.blocos_xcos = []
    if 'conexoes_xcos' not in st.session_state:
        st.session_state.conexoes_xcos = []
    if 'contador_xcos' not in st.session_state:
        st.session_state.contador_xcos = 1
    if 'sistema_atual' not in st.session_state:
        st.session_state.sistema_atual = None
    if 'ultimo_diagrama' not in st.session_state:
        st.session_state.ultimo_diagrama = ""

# =====================================================
# EDITOR VISUAL XCOS
# =====================================================

def criar_editor_xcos():
    """Cria o editor visual de diagrama de blocos estilo Xcos"""
    
    # Criar chave única para forçar atualização
    chave_diagrama = st.text_input(
        "📋 Dados do Diagrama (JSON)",
        value=st.session_state.get('ultimo_diagrama', '{"blocos":[],"conexoes":[]}'),
        key='diagrama_json_input',
        label_visibility='collapsed'
    )
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
                background: #1a1a2e;
            }}
            
            .toolbar {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            
            .btn {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.3s;
                backdrop-filter: blur(10px);
            }}
            
            .btn:hover {{
                background: rgba(255,255,255,0.3);
                border-color: rgba(255,255,255,0.5);
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            
            .btn-danger {{
                background: rgba(231, 76, 60, 0.8);
                border-color: rgba(231, 76, 60, 1);
            }}
            
            .btn-danger:hover {{
                background: rgba(192, 57, 43, 0.9);
            }}
            
            .btn-success {{
                background: rgba(46, 204, 113, 0.8);
                border-color: rgba(46, 204, 113, 1);
                font-size: 16px;
                padding: 12px 30px;
            }}
            
            .btn-success:hover {{
                background: rgba(39, 174, 96, 0.9);
                transform: scale(1.05);
            }}
            
            #canvas-container {{
                width: 100%;
                height: 550px;
                background: 
                    linear-gradient(#2a2a3e 1px, transparent 1px),
                    linear-gradient(90deg, #2a2a3e 1px, transparent 1px);
                background-size: 20px 20px;
                position: relative;
                overflow: hidden;
                cursor: crosshair;
            }}
            
            .bloco {{
                position: absolute;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 12px;
                cursor: move;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                min-width: 140px;
                text-align: center;
                user-select: none;
                transition: all 0.2s;
                border: 2px solid rgba(255,255,255,0.1);
            }}
            
            .bloco:hover {{
                transform: scale(1.08);
                box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
                border-color: rgba(255,255,255,0.3);
            }}
            
            .bloco.selecionado {{
                border: 3px solid #ffd700;
                box-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
                transform: scale(1.05);
            }}
            
            .bloco-tipo {{
                font-size: 10px;
                opacity: 0.9;
                margin-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
            }}
            
            .bloco-nome {{
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 6px;
            }}
            
            .bloco-tf {{
                font-size: 12px;
                font-family: 'Courier New', monospace;
                background: rgba(0,0,0,0.3);
                padding: 6px 8px;
                border-radius: 6px;
                margin-top: 6px;
            }}
            
            .porta {{
                width: 14px;
                height: 14px;
                background: #2ecc71;
                border: 3px solid white;
                border-radius: 50%;
                position: absolute;
                cursor: pointer;
                transition: all 0.2s;
                z-index: 10;
            }}
            
            .porta:hover {{
                background: #27ae60;
                transform: scale(1.4);
                box-shadow: 0 0 15px rgba(46, 204, 113, 0.8);
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
            
            svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }}
            
            #info-panel {{
                position: absolute;
                top: 15px;
                right: 15px;
                background: rgba(26, 26, 46, 0.95);
                color: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                max-width: 280px;
                font-size: 13px;
                border: 2px solid rgba(102, 126, 234, 0.3);
                z-index: 100;
            }}
            
            #info-panel strong {{
                color: #667eea;
                font-size: 16px;
                display: block;
                margin-bottom: 12px;
            }}
            
            #info-panel div {{
                margin: 8px 0;
                padding-left: 10px;
                border-left: 3px solid #667eea;
            }}
            
            .status-bar {{
                background: #16213e;
                color: white;
                padding: 10px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
            }}
            
            .status-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .status-badge {{
                background: #667eea;
                padding: 4px 12px;
                border-radius: 12px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn" onclick="adicionarBlocoTransferencia()">➕ Função Transferência</button>
            <button class="btn" onclick="adicionarBlocoSomador()">⊕ Somador</button>
            <button class="btn" onclick="adicionarBlocoGanho()">📊 Ganho</button>
            <button class="btn" onclick="adicionarBlocoIntegrador()">∫ Integrador</button>
            <button class="btn btn-danger" onclick="removerSelecionado()">🗑️ Remover Selecionado</button>
            <button class="btn btn-danger" onclick="limparDiagrama()">🔄 Limpar Tudo</button>
            <button class="btn btn-success" onclick="exportarDiagrama()">⚡ PROCESSAR SISTEMA</button>
        </div>
        
        <div id="canvas-container">
            <svg id="conexoes-svg"></svg>
            <div id="info-panel">
                <strong>📊 Instruções</strong>
                <div>✏️ Clique nos botões para adicionar blocos</div>
                <div>🖱️ Arraste blocos para posicionar</div>
                <div>🔗 Clique nas portas verdes para conectar</div>
                <div>🎯 Clique no bloco para selecionar</div>
                <div>⚡ Clique em PROCESSAR quando pronto</div>
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <span>Blocos:</span>
                <span class="status-badge" id="status-blocos">0</span>
            </div>
            <div class="status-item">
                <span>Conexões:</span>
                <span class="status-badge" id="status-conexoes">0</span>
            </div>
            <div class="status-item" id="status-mensagem">
                <span>✅ Pronto para começar</span>
            </div>
        </div>

        <script>
            let blocos = [];
            let conexoes = [];
            let blocoIdCounter = 1;
            let blocoSelecionado = null;
            let portaSelecionada = null;
            let arrastandoBloco = null;
            let offsetX = 0, offsetY = 0;

            // Atualizar status
            function atualizarStatus() {{
                document.getElementById('status-blocos').textContent = blocos.length;
                document.getElementById('status-conexoes').textContent = conexoes.length;
                
                let msg = '✅ Pronto';
                if (blocos.length > 0) {{
                    msg = `✅ ${{blocos.length}} bloco(s) - Pronto para processar`;
                }}
                document.getElementById('status-mensagem').innerHTML = `<span>${{msg}}</span>`;
            }}

            function adicionarBloco(tipo, config) {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + blocoIdCounter;
                
                const blocoData = {{
                    id: blocoIdCounter,
                    tipo: tipo,
                    x: 150 + (blocos.length * 200) % 600,
                    y: 150 + Math.floor(blocos.length / 3) * 150,
                    config: config
                }};
                
                blocos.push(blocoData);
                
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                
                let nomeDisplay = config.nome || tipo;
                let tfDisplay = config.tf || '';
                
                bloco.innerHTML = `
                    <div class="bloco-tipo">${{tipo}}</div>
                    <div class="bloco-nome">${{nomeDisplay}}</div>
                    ${{tfDisplay ? '<div class="bloco-tf">' + tfDisplay + '</div>' : ''}}
                    <div class="porta porta-entrada" data-bloco="${{blocoIdCounter}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoIdCounter}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.addEventListener('click', selecionarBloco);
                
                const portas = bloco.querySelectorAll('.porta');
                portas.forEach(porta => {{
                    porta.addEventListener('click', clickPorta);
                }});
                
                blocoIdCounter++;
                atualizarStatus();
            }}

            function adicionarBlocoTransferencia() {{
                const num = prompt('Digite o NUMERADOR da função de transferência:\\n\\nExemplos: 1, 10, s, 2*s+1, s^2+3*s+2', '1');
                if (num === null || num.trim() === '') return;
                
                const den = prompt('Digite o DENOMINADOR da função de transferência:\\n\\nExemplos: s, s+1, s^2+2*s+1, s^2+5*s+6', 's+1');
                if (den === null || den.trim() === '') return;
                
                adicionarBloco('Transferência', {{
                    nome: 'G' + blocoIdCounter,
                    numerador: num.trim(),
                    denominador: den.trim(),
                    tf: num.trim() + ' / (' + den.trim() + ')'
                }});
            }}

            function adicionarBlocoSomador() {{
                adicionarBloco('Somador', {{nome: 'Σ' + blocoIdCounter}});
            }}

            function adicionarBlocoGanho() {{
                const ganho = prompt('Digite o valor do GANHO K:\\n\\nExemplos: 1, 5, 10, 0.5', '1');
                if (ganho === null || ganho.trim() === '') return;
                adicionarBloco('Ganho', {{
                    nome: 'K=' + ganho.trim(),
                    valor: ganho.trim(),
                    tf: ganho.trim()
                }});
            }}

            function adicionarBlocoIntegrador() {{
                adicionarBloco('Integrador', {{nome: '∫' + blocoIdCounter, tf: '1/s'}});
            }}

            function iniciarArrastar(e) {{
                if (e.target.classList.contains('porta')) return;
                e.stopPropagation();
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
                arrastandoBloco = null;
                document.removeEventListener('mousemove', arrastar);
                document.removeEventListener('mouseup', pararArrastar);
            }}

            function selecionarBloco(e) {{
                if (e.target.classList.contains('porta')) return;
                e.stopPropagation();
                
                document.querySelectorAll('.bloco').forEach(b => b.classList.remove('selecionado'));
                e.currentTarget.classList.add('selecionado');
                blocoSelecionado = parseInt(e.currentTarget.id.split('-')[1]);
            }}

            function clickPorta(e) {{
                e.stopPropagation();
                const blocoId = parseInt(e.target.dataset.bloco);
                const tipoPorta = e.target.dataset.tipo;
                
                if (!portaSelecionada) {{
                    portaSelecionada = {{blocoId, tipo: tipoPorta}};
                    e.target.style.background = '#f39c12';
                    e.target.style.boxShadow = '0 0 20px rgba(243, 156, 18, 0.8)';
                }} else {{
                    if (portaSelecionada.tipo === 'saida' && tipoPorta === 'entrada') {{
                        conexoes.push({{
                            origem: portaSelecionada.blocoId,
                            destino: blocoId
                        }});
                        redesenharConexoes();
                        atualizarStatus();
                    }} else if (portaSelecionada.tipo === 'entrada' && tipoPorta === 'saida') {{
                        conexoes.push({{
                            origem: blocoId,
                            destino: portaSelecionada.blocoId
                        }});
                        redesenharConexoes();
                        atualizarStatus();
                    }} else {{
                        alert('❌ Conecte: SAÍDA → ENTRADA');
                    }}
                    
                    document.querySelectorAll('.porta').forEach(p => {{
                        p.style.background = '#2ecc71';
                        p.style.boxShadow = '';
                    }});
                    portaSelecionada = null;
                }}
            }}

            function redesenharConexoes() {{
                const svg = document.getElementById('conexoes-svg');
                svg.innerHTML = '';
                
                // Adicionar definições de marcadores
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
                polygon.setAttribute('fill', '#667eea');
                marker.appendChild(polygon);
                defs.appendChild(marker);
                svg.appendChild(defs);
                
                conexoes.forEach(conexao => {{
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
                        
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        const dx = x2 - x1;
                        const curva = Math.abs(dx) / 2;
                        const d = `M ${{x1}} ${{y1}} C ${{x1 + curva}} ${{y1}}, ${{x2 - curva}} ${{y2}}, ${{x2}} ${{y2}}`;
                        
                        path.setAttribute('d', d);
                        path.setAttribute('stroke', '#667eea');
                        path.setAttribute('stroke-width', '4');
                        path.setAttribute('fill', 'none');
                        path.setAttribute('marker-end', 'url(#arrowhead)');
                        path.setAttribute('filter', 'drop-shadow(0 0 8px rgba(102, 126, 234, 0.6))');
                        
                        svg.appendChild(path);
                    }}
                }});
            }}

            function removerSelecionado() {{
                if (blocoSelecionado !== null) {{
                    const blocoEl = document.getElementById('bloco-' + blocoSelecionado);
                    if (blocoEl) {{
                        blocoEl.remove();
                        blocos = blocos.filter(b => b.id !== blocoSelecionado);
                        conexoes = conexoes.filter(c => c.origem !== blocoSelecionado && c.destino !== blocoSelecionado);
                        redesenharConexoes();
                        blocoSelecionado = null;
                        atualizarStatus();
                    }}
                }} else {{
                    alert('⚠️ Selecione um bloco primeiro clicando nele!');
                }}
            }}

            function limparDiagrama() {{
                if (confirm('🗑️ Deseja realmente limpar todo o diagrama?\\n\\nTodos os blocos e conexões serão removidos.')) {{
                    blocos = [];
                    conexoes = [];
                    blocoSelecionado = null;
                    document.getElementById('canvas-container').innerHTML = `
                        <svg id="conexoes-svg"></svg>
                        <div id="info-panel">
                            <strong>📊 Instruções</strong>
                            <div>✏️ Clique nos botões para adicionar blocos</div>
                            <div>🖱️ Arraste blocos para posicionar</div>
                            <div>🔗 Clique nas portas verdes para conectar</div>
                            <div>🎯 Clique no bloco para selecionar</div>
                            <div>⚡ Clique em PROCESSAR quando pronto</div>
                        </div>
                    `;
                    atualizarStatus();
                }}
            }}

            function exportarDiagrama() {{
                if (blocos.length === 0) {{
                    alert('⚠️ Adicione pelo menos um bloco antes de processar!');
                    return;
                }}
                
                const diagrama = {{
                    blocos: blocos,
                    conexoes: conexoes
                }};
                
                const json = JSON.stringify(diagrama);
                
                // Tentar enviar para o Streamlit
                try {{
                    // Método 1: Atualizar campo de texto
                    const inputField = window.parent.document.querySelector('input[aria-label="📋 Dados do Diagrama (JSON)"]');
                    if (inputField) {{
                        inputField.value = json;
                        inputField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        inputField.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                    
                    alert('✅ Diagrama exportado!\\n\\n' +
                          `📊 Blocos: ${{blocos.length}}\\n` +
                          `🔗 Conexões: ${{conexoes.length}}\\n\\n` +
                          '⚡ Agora clique no botão "PROCESSAR SISTEMA" abaixo do editor!');
                }} catch(e) {{
                    console.error('Erro ao exportar:', e);
                    alert('❌ Erro ao exportar. Tente novamente.');
                }}
            }}

            // Inicializar
            atualizarStatus();
        </script>
    </body>
    </html>
    """
    
    return html_code

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Sistema de Controle - Editor Xcos", layout="wide", initial_sidebar_state="collapsed")
    
    # CSS customizado
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎨 Editor Visual de Sistemas de Controle (Xcos Style)")
    
    inicializar_estado()
    
    # Tabs principais
    tab1, tab2 = st.tabs(["🎨 Editor Visual Xcos", "📊 Análise Clássica"])
    
    with tab1:
        st.markdown("### 🎯 Construa seu Sistema Visualmente")
        st.info("💡 **Como usar:** Adicione blocos, conecte-os arrastando entre as portas verdes, e clique em 'PROCESSAR SISTEMA' no editor.")
        
        # Editor HTML
        html_editor = criar_editor_xcos()
        st.components.v1.html(html_editor, height=700, scrolling=False)
        
        # Área de processamento
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("⚡ PROCESSAR SISTEMA COMPLETO", type="primary", use_container_width=True):
                diagrama_json = st.session_state.get('diagrama_json_input', '{"blocos":[],"conexoes":[]}')
                
                if diagrama_json and diagrama_json != '{"blocos":[],"conexoes":[]}':
                    st.session_state.ultimo_diagrama = diagrama_json
                    
                    sistema, msg = processar_diagrama_blocos_xcos(diagrama_json)
                    
                    if sistema:
                        st.success(msg)
                        st.session_state.sistema_atual = sistema
                        
                        # Mostrar função de transferência
                        st.markdown("#### 📐 Função de Transferência Resultante")
                        st.code(f"G(s) = {sistema}", language="text")
                        
                        # Análises automáticas
                        st.markdown("---")
                        st.markdown("### 📊 Análises Automáticas")
                        
                        # Desempenho
                        with st.expander("📈 Métricas de Desempenho", expanded=True):
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
                        
                        # Gráficos
                        tab_resp, tab_bode, tab_pz, tab_lgr = st.tabs([
                            "📈 Resposta ao Degrau",
                            "📊 Bode",
                            "🎯 Polos e Zeros",
                            "🔄 LGR"
                        ])
                        
                        with tab_resp:
                            fig, t, y = plot_resposta_temporal(sistema, 'Degrau')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Estatísticas da resposta
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Valor Final", f"{y[-1]:.3f}")
                            with col2:
                                st.metric("Máximo", f"{np.max(y):.3f}")
                            with col3:
                                overshoot = ((np.max(y) - y[-1]) / y[-1] * 100) if y[-1] != 0 else 0
                                st.metric("Overshoot", f"{overshoot:.1f}%")
                        
                        with tab_bode:
                            fig = plot_bode(sistema, 'both')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab_pz:
                            fig = plot_polos_zeros(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Info sobre estabilidade
                            polos = ctrl.poles(sistema)
                            estavel = all(np.real(p) < 0 for p in polos)
                            if estavel:
                                st.success("✅ Sistema ESTÁVEL (todos os polos no semiplano esquerdo)")
                            else:
                                st.error("❌ Sistema INSTÁVEL (polos no semiplano direito)")
                        
                        with tab_lgr:
                            fig = plot_lgr(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(msg)
                else:
                    st.warning("⚠️ Nenhum diagrama foi criado ainda. Use o editor acima para adicionar blocos!")
        
        with col2:
            if st.button("💾 Exportar JSON", use_container_width=True):
                diagrama_json = st.session_state.get('diagrama_json_input', '{"blocos":[],"conexoes":[]}')
                if diagrama_json != '{"blocos":[],"conexoes":[]}':
                    st.download_button(
                        label="📥 Download",
                        data=diagrama_json,
                        file_name="sistema_controle.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("Nada para exportar")
        
        with col3:
            if st.button("📖 Ajuda Rápida", use_container_width=True):
                st.info("""
                **Passos:**
                1. Adicione blocos clicando nos botões
                2. Arraste para posicionar
                3. Conecte: porta saída → porta entrada
                4. Clique em "PROCESSAR SISTEMA"
                5. Analise os resultados abaixo
                """)
    
    with tab2:
        st.markdown("### 📐 Modo Clássico - Entrada Manual")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🔢 Função de Transferência")
            num_input = st.text_input("Numerador", placeholder="Ex: 10", key="num_classico")
            den_input = st.text_input("Denominador", placeholder="Ex: s^2 + 2*s + 1", key="den_classico")
            
            if st.button("🔍 Analisar", type="primary", use_container_width=True):
                if num_input and den_input:
                    try:
                        tf, _ = converter_para_tf(num_input, den_input)
                        st.session_state.sistema_atual = tf
                        st.success(f"✅ Sistema carregado: G(s) = {tf}")
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
                else:
                    st.warning("⚠️ Preencha numerador e denominador")
        
        with col2:
            if st.session_state.sistema_atual:
                st.markdown("#### 📊 Sistema Atual")
                st.code(f"G(s) = {st.session_state.sistema_atual}", language="text")
                
                # Análise rápida
                desempenho = calcular_desempenho(st.session_state.sistema_atual)
                st.markdown("**Tipo:** " + desempenho.get('Tipo', 'N/A'))
                
                polos = ctrl.poles(st.session_state.sistema_atual)
                estavel = all(np.real(p) < 0 for p in polos)
                st.markdown("**Estabilidade:** " + ("✅ Estável" if estavel else "❌ Instável"))
        
        # Gráficos do sistema atual
        if st.session_state.sistema_atual:
            st.markdown("---")
            st.markdown("### 📈 Visualizações")
            
            tabs = st.tabs(["📈 Resposta", "📊 Bode", "🎯 Polos/Zeros", "🔄 LGR", "⭕ Nyquist"])
            
            with tabs[0]:
                entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)
                fig, t, y = plot_resposta_temporal(st.session_state.sistema_atual, entrada)
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                fig = plot_bode(st.session_state.sistema_atual, 'both')
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                fig = plot_polos_zeros(st.session_state.sistema_atual)
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:
                fig = plot_lgr(st.session_state.sistema_atual)
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[4]:
                fig, p, n, z = plot_nyquist(st.session_state.sistema_atual)
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Polos SPD: {p} | Voltas: {n} | Z: {z} → {'Estável' if z == 0 else 'Instável'}")

if __name__ == "__main__":
    main()
