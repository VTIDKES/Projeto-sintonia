# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Analise de Sistemas de Controle
Sistema de Modelagem e Analise de Sistemas de Controle v2.0
Refatorado com: tela inicial, espaco de estados, modal de blocos,
logica corrigida de serie/paralelo/feedback, simplificacao automatica.
"""

import streamlit as st
@@ -14,35 +16,89 @@
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import json
import re

# =====================================================
# CONFIGURACOES E CONSTANTES
# =====================================================
# ══════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════

ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
    "malha_aberta": [
        "Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
        "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"
    ],
    "malha_fechada": [
        "Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
        "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"
    ],
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabolica']

# =====================================================
# FUNCOES AUXILIARES
# =====================================================
BLOCK_TYPES = {
    'Planta': {'icon': 'G(s)', 'desc': 'Funcao de transferencia da planta'},
    'Controlador': {'icon': 'C(s)', 'desc': 'Controlador (PID, Lead-Lag, etc.)'},
    'Sensor': {'icon': 'H(s)', 'desc': 'Sensor na malha de realimentacao'},
    'Atuador': {'icon': 'A(s)', 'desc': 'Atuador do sistema'},
    'Pre-filtro': {'icon': 'F(s)', 'desc': 'Filtro antes do somador'},
    'Perturbacao': {'icon': 'D(s)', 'desc': 'Perturbacao/disturbio'},
}

CONNECTION_TYPES = ['Serie', 'Paralelo', 'Realimentacao Negativa', 'Realimentacao Positiva']

# ══════════════════════════════════════════════════
# INICIALIZACAO DO SESSION STATE
# ══════════════════════════════════════════════════

def inicializar_estado():
    defaults = {
        'modo_selecionado': None,
        'blocos': pd.DataFrame(columns=[
            'nome', 'tipo', 'representacao', 'numerador', 'denominador',
            'A', 'B', 'C', 'D', 'tf', 'tf_simbolico'
        ]),
        'conexoes': [],
        'calculo_erro_habilitado': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════
# FUNCOES UTILITARIAS
# ══════════════════════════════════════════════════

def formatar_numero(valor):
    if np.isinf(valor):
        return '∞'
    elif np.isnan(valor):
    if np.isnan(valor):
        return '-'
    else:
        return f"{valor:.3f}"

# =====================================================
# FUNCOES DE TRANSFERENCIA
# =====================================================
    return f"{valor:.3f}"


def parse_matrix(text):
    """Converte texto para matriz numpy. Aceita formatos:
       '[[1,2],[3,4]]'  ou  '1 2; 3 4'  ou  '1,2;3,4'
    """
    text = text.strip()
    if text.startswith('['):
        try:
            return np.array(json.loads(text), dtype=float)
        except Exception:
            text = text.replace('[', '').replace(']', '')
    rows = [r.strip() for r in text.split(';') if r.strip()]
    matrix = []
    for row in rows:
        vals = re.split(r'[,\s]+', row.strip())
        matrix.append([float(v) for v in vals if v])
    return np.array(matrix, dtype=float)


# ══════════════════════════════════════════════════
# FUNCOES DE TRANSFERENCIA E ESPACO DE ESTADOS
# ══════════════════════════════════════════════════

def converter_para_tf(numerador_str, denominador_str):
    s = sp.Symbol('s')
@@ -57,11 +113,37 @@ def converter_para_tf(numerador_str, denominador_str):
        den_coeffs = [c / fator for c in den_coeffs]
    return TransferFunction(num_coeffs, den_coeffs), (num, den)


def converter_ss_para_tf(A_str, B_str, C_str, D_str):
    """Converte espaco de estados (A,B,C,D) para funcao de transferencia.
    T(s) = C(sI - A)^(-1)B + D
    """
    A = parse_matrix(A_str)
    B = parse_matrix(B_str)
    C = parse_matrix(C_str)
    D = parse_matrix(D_str)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"Matriz A deve ser quadrada ({n}x{n})")
    if B.shape[0] != n:
        raise ValueError(f"Matriz B deve ter {n} linhas")
    if C.shape[1] != n:
        raise ValueError(f"Matriz C deve ter {n} colunas")
    ss_sys = ctrl.ss(A, B, C, D)
    tf_sys = ctrl.tf(ss_sys)
    s = sp.Symbol('s')
    num_coeffs = list(tf_sys.num[0][0])
    den_coeffs = list(tf_sys.den[0][0])
    num_sym = sum(c * s**i for i, c in enumerate(reversed(num_coeffs)))
    den_sym = sum(c * s**i for i, c in enumerate(reversed(den_coeffs)))
    return tf_sys, (num_sym, den_sym), ss_sys


def tipo_do_sistema(G):
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    tipo = sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))
    return tipo
    return sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))


def constantes_de_erro(G):
    s = ctrl.tf('s')
@@ -86,6 +168,93 @@ def constantes_de_erro(G):
        Kp = Kv = 0
    return tipo, Kp, Kv, Ka


# ══════════════════════════════════════════════════
# INTERCONEXAO DE BLOCOS (SERIE, PARALELO, FEEDBACK)
# ══════════════════════════════════════════════════

def blocos_em_serie(tf_list):
    """Serie: G_total = G1 * G2 * ... * Gn"""
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = ctrl.series(resultado, tf)[2]
    return resultado


def blocos_em_paralelo(tf_list):
    """Paralelo: G_total = G1 + G2 + ... + Gn"""
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = ctrl.parallel(resultado, tf)[2]
    return resultado


def realimentacao(G, H=None, positiva=False):
    """Feedback: T = G / (1 +/- G*H)
    Negativa (padrao): T = G / (1 + G*H)
    Positiva: T = G / (1 - G*H)
    """
    if H is None:
        H = TransferFunction([1], [1])
    sign = 1 if positiva else -1
    return ctrl.feedback(G, H, sign=sign)


def simplificar_diagrama(blocos_df, conexoes):
    """Simplifica um diagrama de blocos baseado nas conexoes definidas.
    Retorna a funcao de transferencia equivalente.
    """
    if blocos_df.empty:
        raise ValueError("Nenhum bloco definido.")

    tf_map = {}
    for _, row in blocos_df.iterrows():
        tf_map[row['nome']] = row['tf']

    if not conexoes:
        tfs = list(tf_map.values())
        if len(tfs) == 1:
            return tfs[0]
        return blocos_em_serie(tfs)

    resultado = None
    for con in conexoes:
        tipo_con = con['tipo']
        nomes = con['blocos']
        tfs = [tf_map[n] for n in nomes if n in tf_map]

        if len(tfs) < 2:
            if len(tfs) == 1:
                resultado = tfs[0] if resultado is None else ctrl.series(resultado, tfs[0])[2]
            continue

        if tipo_con == 'Serie':
            parcial = blocos_em_serie(tfs)
        elif tipo_con == 'Paralelo':
            parcial = blocos_em_paralelo(tfs)
        elif tipo_con == 'Realimentacao Negativa':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=False)
        elif tipo_con == 'Realimentacao Positiva':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=True)
        else:
            parcial = blocos_em_serie(tfs)

        if resultado is None:
            resultado = parcial
        else:
            resultado = ctrl.series(resultado, parcial)[2]

    if resultado is None:
        tfs = list(tf_map.values())
        resultado = blocos_em_serie(tfs) if len(tfs) > 1 else tfs[0]

    return ctrl.minreal(resultado, verbose=False)


def calcular_malha_fechada(planta, controlador=None, sensor=None):
    if controlador is None:
        controlador = TransferFunction([1], [1])
@@ -95,29 +264,31 @@ def calcular_malha_fechada(planta, controlador=None, sensor=None):
    H = sensor
    return ctrl.feedback(G, H)

# =====================================================

# ══════════════════════════════════════════════════
# ANALISE DE SISTEMAS
# =====================================================
# ══════════════════════════════════════════════════

def calcular_desempenho(tf):
    den = tf.den[0][0]
def calcular_desempenho(tf_sys):
    den = tf_sys.den[0][0]
    ordem = len(den) - 1
    polos = ctrl.poles(tf)
    gm, pm, wg, wp = margin(tf)
    polos = ctrl.poles(tf_sys)
    gm, pm, wg, wp = margin(tf_sys)
    gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf
    resultado = {
        'Margem de ganho': f"{formatar_numero(gm)} ({'∞' if gm == np.inf else f'{formatar_numero(gm_db)} dB'})",
        'Margem de fase': f"{formatar_numero(pm)}°",
        'Freq. cruz. fase': f"{formatar_numero(wg)} rad/s",
        'Freq. cruz. ganho': f"{formatar_numero(wp)} rad/s"
        'Freq. cruz. ganho': f"{formatar_numero(wp)} rad/s",
    }
    if ordem == 1:
        return _desempenho_ordem1(polos, resultado)
    elif ordem == 2:
        return _desempenho_ordem2(polos, resultado)
    elif ordem >= 3:
    else:
        return _desempenho_ordem_superior(polos, ordem, resultado)


def _desempenho_ordem1(polos, resultado):
    tau = -1 / polos[0].real
    resultado.update({
@@ -126,15 +297,16 @@ def _desempenho_ordem1(polos, resultado):
        'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(4 * tau)} s",
        'Freq. natural (wn)': f"{formatar_numero(1/tau)} rad/s",
        'Fator amortec. (z)': "1.0"
        'Fator amortec. (z)': "1.0",
    })
    return resultado


def _desempenho_ordem2(polos, resultado):
    wn = np.sqrt(np.prod(np.abs(polos))).real
    zeta = -np.real(polos[0]) / wn
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
    Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
    Tp = np.pi / wd if wd > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
@@ -146,16 +318,18 @@ def _desempenho_ordem2(polos, resultado):
        'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s"
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s",
    })
    return resultado


def _desempenho_ordem_superior(polos, ordem, resultado):
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
    par_dominante = None
    for i in range(len(polos_ordenados) - 1):
        p1, p2 = polos_ordenados[i], polos_ordenados[i+1]
        if np.isclose(p1.real, p2.real, atol=1e-2) and np.isclose(p1.imag, -p2.imag, atol=1e-2):
        p1, p2 = polos_ordenados[i], polos_ordenados[i + 1]
        if (np.isclose(p1.real, p2.real, atol=1e-2)
                and np.isclose(p1.imag, -p2.imag, atol=1e-2)):
            par_dominante = (p1, p2)
            break
    if par_dominante:
@@ -168,12 +342,13 @@ def _desempenho_ordem_superior(polos, ordem, resultado):
        wn = np.abs(polo_dominante)
        zeta = -np.real(polo_dominante) / wn if wn != 0 else 0
        omega_d = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
    Tr = (np.pi - np.arccos(zeta)) / omega_d if zeta < 1 and omega_d > 0 else float('inf')
    Tp = np.pi / omega_d if omega_d > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
    label = f'{ordem}a Ordem (Par dominante)' if par_dominante else f'{ordem}a Ordem (Polo dominante)'
    resultado.update({
        'Tipo': f'{ordem}a Ordem (Par dominante)' if par_dominante else f'{ordem}a Ordem (Polo dominante)',
        'Tipo': label,
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(omega_d)} rad/s",
@@ -184,8 +359,9 @@ def _desempenho_ordem_superior(polos, ordem, resultado):
    })
    return resultado

def estimar_tempo_final_simulacao(tf):
    polos = ctrl.poles(tf)

def estimar_tempo_final_simulacao(tf_sys):
    polos = ctrl.poles(tf_sys)
    if len(polos) == 0:
        return 50.0
    if any(np.real(p) > 1e-6 for p in polos):
@@ -195,48 +371,57 @@ def estimar_tempo_final_simulacao(tf):
        return 100.0
    sigma_dominante = max(partes_reais_estaveis)
    ts_estimado = 4 / abs(sigma_dominante)
    tempo_final = ts_estimado * 1.5
    return np.clip(tempo_final, a_min=10, a_max=500)
    return np.clip(ts_estimado * 1.5, a_min=10, a_max=500)


# =====================================================
# ══════════════════════════════════════════════════
# FUNCOES DE PLOTAGEM
# =====================================================
# ══════════════════════════════════════════════════

def configurar_linhas_interativas(fig):
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(line=dict(color='green', width=2, dash='dash')),
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                     'drawcircle', 'drawrect', 'eraseshape'],
    )
    return fig

def plot_polos_zeros(tf, fig=None):
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)

def plot_polos_zeros(tf_sys, fig=None):
    zeros = ctrl.zeros(tf_sys)
    polos = ctrl.poles(tf_sys)
    if fig is None:
        fig = go.Figure()
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=12, color='blue'), name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Diagrama de Polos e Zeros', xaxis_title='Parte Real',
    fig.update_layout(
        title='Diagrama de Polos e Zeros', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    return fig
    return configurar_linhas_interativas(fig)


def _gerar_sinal_entrada(entrada, t):
    sinais = {
        'Degrau': np.ones_like(t), 'Rampa': t, 'Senoidal': np.sin(2*np.pi*t),
        'Impulso': np.concatenate([[1], np.zeros(len(t)-1)]), 'Parabolica': t**2
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2 * np.pi * t),
        'Impulso': np.concatenate([[1], np.zeros(len(t) - 1)]),
        'Parabolica': t**2,
    }
    return sinais[entrada]


def plot_resposta_temporal(sistema, entrada):
    tempo_final = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tempo_final, 1000)
@@ -246,29 +431,36 @@ def plot_resposta_temporal(sistema, entrada):
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=u[:len(t_out)], mode='lines',
    fig.add_trace(go.Scatter(
        x=t_out, y=u[:len(t_out)], mode='lines',
        line=dict(dash='dash', color='blue'), name='Entrada',
        hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines',
        line=dict(color='red'), name='Saida',
    fig.add_trace(go.Scatter(
        x=t_out, y=y, mode='lines', line=dict(color='red'), name='Saida',
        hovertemplate='Tempo: %{x:.2f}s<br>Saida: %{y:.3f}<extra></extra>'))
    fig.update_layout(title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Amplitude', showlegend=True, hovermode='x unified')
    fig = configurar_linhas_interativas(fig)
    return fig, t_out, y
    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Amplitude',
        showlegend=True, hovermode='x unified')
    return configurar_linhas_interativas(fig), t_out, y


def plot_bode(sistema, tipo='both'):
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys = signal.TransferFunction(numerator, denominator)
    sys_scipy = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)
    w, mag, phase = signal.bode(sys_scipy, w)
    if tipo == 'both':
        fig = make_subplots(rows=2, cols=1,
            subplot_titles=('Bode - Magnitude', 'Bode - Fase'), vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Bode - Magnitude', 'Bode - Fase'),
            vertical_spacing=0.1)
        fig.add_trace(go.Scatter(
            x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='red', width=3),
        fig.add_trace(go.Scatter(
            x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase', showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=2, col=1)
@@ -277,52 +469,69 @@ def plot_bode(sistema, tipo='both'):
        fig.update_layout(height=700, title_text="Diagrama de Bode")
    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='blue', width=3), name='Magnitude'))
        fig.update_layout(title='Bode - Magnitude', xaxis_title="Frequencia (rad/s)",
        fig.add_trace(go.Scatter(
            x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude'))
        fig.update_layout(
            title='Bode - Magnitude', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Magnitude (dB)", xaxis_type='log')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='red', width=3), name='Fase'))
        fig.update_layout(title='Bode - Fase', xaxis_title="Frequencia (rad/s)",
        fig.add_trace(go.Scatter(
            x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase'))
        fig.update_layout(
            title='Bode - Fase', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Fase (deg)", xaxis_type='log')
    fig = configurar_linhas_interativas(fig)
    return fig
    return configurar_linhas_interativas(fig)


def plot_lgr(sistema):
    rlist, klist = root_locus(sistema, plot=False)
    fig = go.Figure()
    for i, r in enumerate(rlist.T):
        fig.add_trace(go.Scatter(x=np.real(r), y=np.imag(r), mode='lines',
            line=dict(color='blue', width=1), name=f'Ramo {i+1}', showlegend=False))
        fig.add_trace(go.Scatter(
            x=np.real(r), y=np.imag(r), mode='lines',
            line=dict(color='blue', width=1),
            name=f'Ramo {i+1}', showlegend=False))
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=10, color='green'), name='Zeros'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Lugar Geometrico das Raizes (LGR)',
        xaxis_title='Parte Real', yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    return fig
    fig.update_layout(
        title='Lugar Geometrico das Raizes (LGR)',
        xaxis_title='Parte Real', yaxis_title='Parte Imaginaria',
        showlegend=True, hovermode='closest')
    return configurar_linhas_interativas(fig)


def plot_nyquist(sistema):
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H.real, y=H.imag, mode='lines',
    fig.add_trace(go.Scatter(
        x=H.real, y=H.imag, mode='lines',
        line=dict(color='blue', width=2), name='Nyquist'))
    fig.add_trace(go.Scatter(x=H.real, y=-H.imag, mode='lines',
    fig.add_trace(go.Scatter(
        x=H.real, y=-H.imag, mode='lines',
        line=dict(dash='dash', color='gray', width=1), name='Reflexo simetrico'))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'), name='Ponto critico (-1,0)'))
    fig.add_trace(go.Scatter(
        x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto critico (-1,0)'))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(title='Diagrama de Nyquist', xaxis_title='Parte Real',
    fig.update_layout(
        title='Diagrama de Nyquist', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    polos = ctrl.poles(sistema)
@@ -331,33 +540,42 @@ def plot_nyquist(sistema):
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
# ══════════════════════════════════════════════════
# GERENCIAMENTO DE BLOCOS
# ══════════════════════════════════════════════════

def adicionar_bloco(nome, tipo, numerador, denominador):
def adicionar_bloco(nome, tipo, representacao, numerador='', denominador='',
                    A_str='', B_str='', C_str='', D_str=''):
    try:
        tf, tf_symb = converter_para_tf(numerador, denominador)
        if representacao == 'Funcao de Transferencia':
            tf_obj, tf_symb = converter_para_tf(numerador, denominador)
            ss_sys = None
        else:
            tf_obj, tf_symb, ss_sys = converter_ss_para_tf(A_str, B_str, C_str, D_str)
            numerador = str(list(tf_obj.num[0][0]))
            denominador = str(list(tf_obj.den[0][0]))

        novo = pd.DataFrame([{
            'nome': nome, 'tipo': tipo, 'numerador': numerador,
            'denominador': denominador, 'tf': tf, 'tf_simbolico': tf_symb
            'nome': nome, 'tipo': tipo, 'representacao': representacao,
            'numerador': numerador, 'denominador': denominador,
            'A': A_str, 'B': B_str, 'C': C_str, 'D': D_str,
            'tf': tf_obj, 'tf_simbolico': tf_symb,
        }])
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco {nome} adicionado."
        st.session_state.blocos = pd.concat(
            [st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco '{nome}' adicionado com sucesso."
    except Exception as e:
        return False, f"Erro na conversao: {e}"
        return False, f"Erro: {e}"


def remover_bloco(nome):
    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['nome'] != nome]
    return f"Bloco {nome} excluido."
    st.session_state.blocos = st.session_state.blocos[
        st.session_state.blocos['nome'] != nome]
    st.session_state.conexoes = [
        c for c in st.session_state.conexoes if nome not in c['blocos']]
    return f"Bloco '{nome}' removido."


def obter_bloco_por_tipo(tipo):
    df = st.session_state.blocos
@@ -366,10 +584,58 @@ def obter_bloco_por_tipo(tipo):
    return None


# =====================================================
# EDITOR VISUAL - HTML COMPLETO INLINE
# Nao depende de nenhum arquivo externo
# =====================================================
def obter_bloco_por_nome(nome):
    df = st.session_state.blocos
    match = df[df['nome'] == nome]
    if not match.empty:
        return match.iloc[0]['tf']
    return None


# ══════════════════════════════════════════════════
# EXECUCAO DE ANALISES
# ══════════════════════════════════════════════════

def executar_analises(sistema, analises, entrada, tipo_malha):
    for analise in analises:
        st.markdown(f"### {analise}")
        try:
            if analise == 'Resposta no tempo':
                fig, t_out, y = plot_resposta_temporal(sistema, entrada)
                st.plotly_chart(fig, use_container_width=True)
            elif analise == 'Desempenho':
                desempenho = calcular_desempenho(sistema)
                cols = st.columns(3)
                items = list(desempenho.items())
                for i, (chave, valor) in enumerate(items):
                    with cols[i % 3]:
                        st.metric(label=chave, value=valor)
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
                c1, c2, c3 = st.columns(3)
                c1.metric("Polos SPD (P)", polos_spd)
                c2.metric("Voltas (N)", voltas)
                c3.metric("Z = P + N", f"{Z} ({'Estavel' if Z == 0 else 'Instavel'})")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro em '{analise}': {e}")


# ══════════════════════════════════════════════════
# EDITOR VISUAL HTML (MODO CANVAS)
# ══════════════════════════════════════════════════

def _load_visual_editor_html():
    return r'''<!DOCTYPE html>
@@ -381,12 +647,41 @@ def _load_visual_editor_html():
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:auto;min-height:100%;font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);overflow-y:auto}
.app{display:flex;flex-direction:column;min-height:100%}
/* Modal Overlay */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:1000;align-items:center;justify-content:center}
.modal-overlay.vis{display:flex}
.modal{background:var(--sf);border:1px solid var(--bd);border-radius:16px;width:90%;max-width:520px;max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.5)}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--bd)}
.modal-hdr h3{font-size:16px;color:var(--grn)}
.modal-close{background:none;border:none;color:var(--txm);font-size:22px;cursor:pointer;padding:4px 8px}
.modal-close:hover{color:var(--red)}
.modal-body{padding:16px 20px}
.block-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.block-option{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:14px;cursor:pointer;transition:all .15s;text-align:center}
.block-option:hover{border-color:var(--acc);background:#2f3349;transform:scale(1.03)}
.block-option .bo-icon{font-size:22px;font-weight:700;margin-bottom:4px}
.block-option .bo-label{font-size:12px;font-weight:600;color:var(--tx)}
.block-option .bo-desc{font-size:10px;color:var(--txm);margin-top:2px}
.cat-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--acc);margin:12px 0 6px;font-weight:700}
.cat-label:first-child{margin-top:0}
/* Config sub-panel inside modal */
.cfg-panel{display:none;padding:12px 0 0}
.cfg-panel.vis{display:block}
.cfg-panel h4{font-size:12px;color:var(--txm);margin-bottom:8px}
.cfg-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.cfg-row label{font-size:10px;text-transform:uppercase;color:var(--txm);display:block;margin-bottom:3px}
.cfg-row input{width:100%;padding:8px;background:var(--sf2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);font-size:13px;outline:none}
.cfg-row input:focus{border-color:var(--acc)}
.cfg-btn{background:var(--acc);border:none;color:#fff;font-weight:700;font-size:13px;padding:10px 24px;border-radius:8px;cursor:pointer;width:100%;margin-top:8px}
.cfg-btn:hover{background:#6e7df0}
.toolbar{background:var(--sf);border-bottom:1px solid var(--bd);padding:6px 10px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;min-height:44px;z-index:50;position:sticky;top:0}
.toolbar .sep{width:1px;height:26px;background:var(--bd);margin:0 3px}
.toolbar .lbl{font-size:10px;color:var(--txm);text-transform:uppercase;letter-spacing:.5px;margin-right:2px;white-space:nowrap}
.tb{display:inline-flex;align-items:center;gap:4px;background:var(--sf2);color:#c8cad8;border:1px solid var(--bd);
border-radius:6px;padding:6px 10px;font-size:11px;cursor:pointer;white-space:nowrap;transition:all .15s;min-height:34px;touch-action:manipulation}
.tb:hover,.tb:active{background:#2f3349;border-color:var(--acc);color:#fff}
.tb-add{background:#16382a;border-color:#2d8a55;color:var(--grn);font-weight:700;font-size:13px;padding:8px 16px}
.tb-add:hover{background:#1e4a36}
.workspace{display:flex;height:460px;min-height:320px;flex-shrink:0}
.canvas-wrap{flex:1;position:relative;overflow:hidden;background:radial-gradient(circle,#141722,#0e1117);touch-action:none}
.canvas-wrap::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,#1e2235 1px,transparent 1px);background-size:24px 24px;opacity:.45;pointer-events:none}
@@ -406,6 +701,7 @@ def _load_visual_editor_html():
.block-der{background:linear-gradient(135deg,#2a2a2a,#1e1e1e);border:1px solid #555}
.block-pid{background:linear-gradient(135deg,#2a1f4e,#1a1535);border:1px solid #5548a0}
.block-sensor{background:linear-gradient(135deg,#4a1a2d,#351020);border:1px solid #8a2d50}
.block-actuator{background:linear-gradient(135deg,#1a3a4a,#102535);border:1px solid #2d708a}
.block-input{background:linear-gradient(135deg,#1a3d1a,#0f250f);border:1px solid #2d8a2d;border-radius:20px}
.block-output{background:linear-gradient(135deg,#3d1a1a,#250f0f);border:1px solid #8a2d2d;border-radius:20px}
.block-sum{background:linear-gradient(135deg,#1a3d3a,#122a28);border:2px solid #2d8a70;border-radius:50%;min-width:56px;width:56px;height:56px;display:flex;align-items:center;justify-content:center}
@@ -435,7 +731,7 @@ def _load_visual_editor_html():
.manual-sec{background:var(--sf);padding:20px;border-top:1px solid var(--bd);display:none}
.manual-sec.vis{display:block}
.manual-sec h4{font-size:12px;text-transform:uppercase;color:var(--acc);margin-bottom:14px;letter-spacing:.5px}
.man-tabs{display:flex;gap:8px;margin-bottom:16px}
.man-tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.man-tab{padding:8px 16px;background:var(--sf2);border:1px solid var(--bd);border-radius:8px;color:var(--txm);font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;touch-action:manipulation}
.man-tab.active{background:var(--acc);border-color:var(--acc);color:#fff}
.man-tab:hover:not(.active){border-color:var(--acc);color:var(--tx)}
@@ -468,22 +764,54 @@ def _load_visual_editor_html():
.calc-bar button{font-size:14px;padding:10px 24px;width:100%}
.mgrid{grid-template-columns:repeat(2,1fr)}
.man-row{grid-template-columns:1fr}.man-tabs{flex-wrap:wrap}
.block-grid{grid-template-columns:1fr}
}
</style></head><body><div class="app">
<!-- MODAL DE ADICIONAR BLOCO -->
<div class="modal-overlay" id="addModal">
<div class="modal">
<div class="modal-hdr"><h3>Adicionar Bloco</h3><button class="modal-close" onclick="closeModal()">&times;</button></div>
<div class="modal-body">
<div id="modalGrid">
<div class="cat-label">Sinais</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('input')"><div class="bo-icon" style="color:var(--grn)">R(s)</div><div class="bo-label">Entrada</div><div class="bo-desc">Sinal de referencia</div></div>
<div class="block-option" onclick="pickBlock('output')"><div class="bo-icon" style="color:var(--red)">Y(s)</div><div class="bo-label">Saida</div><div class="bo-desc">Sinal de saida</div></div>
</div>
<div class="cat-label">Blocos de Transferencia</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('tf')"><div class="bo-icon" style="color:var(--blu)">G(s)</div><div class="bo-label">Planta</div><div class="bo-desc">Funcao de transferencia</div></div>
<div class="block-option" onclick="pickBlock('gain')"><div class="bo-icon" style="color:var(--pur)">K</div><div class="bo-label">Ganho</div><div class="bo-desc">Ganho constante</div></div>
<div class="block-option" onclick="pickBlock('pid')"><div class="bo-icon" style="color:var(--pur)">PID</div><div class="bo-label">Controlador PID</div><div class="bo-desc">Kp + Ki/s + Kd*s</div></div>
<div class="block-option" onclick="pickBlock('int')"><div class="bo-icon" style="color:var(--yel)">1/s</div><div class="bo-label">Integrador</div><div class="bo-desc">Integracao pura</div></div>
<div class="block-option" onclick="pickBlock('der')"><div class="bo-icon" style="color:#aaa">s</div><div class="bo-label">Derivador</div><div class="bo-desc">Derivacao pura</div></div>
<div class="block-option" onclick="pickBlock('actuator')"><div class="bo-icon" style="color:var(--blu)">A(s)</div><div class="bo-label">Atuador</div><div class="bo-desc">Dinamica do atuador</div></div>
</div>
<div class="cat-label">Realimentacao</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('sensor')"><div class="bo-icon" style="color:var(--pnk)">H(s)</div><div class="bo-label">Sensor</div><div class="bo-desc">Malha de realimentacao</div></div>
<div class="block-option" onclick="pickBlock('sum')"><div class="bo-icon" style="color:var(--grn)">&Sigma;</div><div class="bo-label">Somador</div><div class="bo-desc">Soma/subtracao de sinais</div></div>
<div class="block-option" onclick="pickBlock('branch')"><div class="bo-icon" style="color:var(--blu)">&bull;</div><div class="bo-label">Ramificacao</div><div class="bo-desc">Divide sinal em dois</div></div>
</div>
</div>
<!-- Config panel after picking -->
<div class="cfg-panel" id="cfgPanel">
<h4 id="cfgTitle">Configurar bloco</h4>
<div id="cfgFields"></div>
<button class="cfg-btn" onclick="confirmAdd()">Adicionar ao Diagrama</button>
<button class="cfg-btn" style="background:var(--sf2);margin-top:4px;border:1px solid var(--bd)" onclick="backToGrid()">Voltar</button>
</div>
</div></div></div>
<div class="toolbar" id="diag-toolbar">
<span class="lbl">Sinais:</span>
<button class="tb" style="background:#16382a;border-color:#2d8a55;color:var(--grn)" data-add="input">R(s)</button>
<button class="tb" style="background:#381620;border-color:#8a2d3a;color:var(--red)" data-add="output">Y(s)</button>
<button class="tb tb-add" onclick="openModal()">+ Adicionar Bloco</button>
<div class="sep"></div>
<span class="lbl">Blocos:</span>
<span class="lbl">Rapido:</span>
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
@@ -517,6 +845,7 @@ def _load_visual_editor_html():
<button class="man-tab active" id="subDirect" onclick="setSubMode('direct')">T(s) Direta</button>
<button class="man-tab" id="subClosed" onclick="setSubMode('closed')">Malha Fechada G/(1+GH)</button>
<button class="man-tab" id="subOpen" onclick="setSubMode('open')">Malha Aberta G*H</button>
<button class="man-tab" id="subSS" onclick="setSubMode('ss')">Espaco de Estados</button>
</div>
<div id="manDirect">
<h4>Funcao de Transferencia T(s) = Num / Den</h4>
@@ -550,6 +879,18 @@ def _load_visual_editor_html():
</div>
<div class="man-hint">Analisa a funcao de transferencia de malha aberta <code>L(s) = G*H</code>.</div>
</div>
<div id="manSS" style="display:none">
<h4>Espaco de Estados: dx/dt = Ax + Bu, y = Cx + Du</h4>
<div class="man-row">
<div class="pg"><label>Matriz A (nxn)</label><input id="manA" value="0 1; -2 -3" placeholder="0 1; -2 -3"></div>
<div class="pg"><label>Matriz B (nx1)</label><input id="manB" value="0; 1" placeholder="0; 1"></div>
</div>
<div class="man-row">
<div class="pg"><label>Matriz C (1xn)</label><input id="manC" value="1 0" placeholder="1 0"></div>
<div class="pg"><label>Matriz D (1x1)</label><input id="manD" value="0" placeholder="0"></div>
</div>
<div class="man-hint">Use <code>;</code> para separar linhas. Ex: <code>0 1; -2 -3</code> = matriz 2x2. Converte para T(s) = C(sI-A)<sup>-1</sup>B + D</div>
</div>
</div>
<div class="calc-bar"><button id="btnCalcMain" onclick="onCalc()">&#9654; CALCULAR DIAGRAMA</button></div>
@@ -559,6 +900,38 @@ def _load_visual_editor_html():
</div><div class="rbody" id="rb"></div></div></div>
<script>
/* ===== MODAL ===== */
var pendingType=null;
function openModal(){document.getElementById("addModal").classList.add("vis");document.getElementById("modalGrid").style.display="block";document.getElementById("cfgPanel").classList.remove("vis");pendingType=null}
function closeModal(){document.getElementById("addModal").classList.remove("vis")}
function backToGrid(){document.getElementById("modalGrid").style.display="block";document.getElementById("cfgPanel").classList.remove("vis")}
function pickBlock(t){
  var simple=["input","output","int","der","branch"];
  if(simple.indexOf(t)>=0){addB(t);closeModal();return}
  pendingType=t;
  document.getElementById("modalGrid").style.display="none";
  var cp=document.getElementById("cfgPanel");cp.classList.add("vis");
  var cf=document.getElementById("cfgFields");var h="";
  var titles={tf:"Planta G(s)",gain:"Ganho K",pid:"Controlador PID",sensor:"Sensor H(s)",sum:"Somador",actuator:"Atuador A(s)"};
  document.getElementById("cfgTitle").textContent="Configurar: "+(titles[t]||t);
  if(t==="tf"||t==="sensor"||t==="actuator"){
    h+='<div class="cfg-row"><div><label>Numerador</label><input id="cfgNum" value="1" placeholder="ex: s+1"></div>';
    h+='<div><label>Denominador</label><input id="cfgDen" value="s+1" placeholder="ex: s^2+2s+1"></div></div>'}
  else if(t==="gain"){h+='<div class="cfg-row"><div><label>Ganho K</label><input id="cfgK" value="1" placeholder="ex: 10"></div><div></div></div>'}
  else if(t==="pid"){
    h+='<div class="cfg-row"><div><label>Kp</label><input id="cfgKp" value="1"></div><div><label>Ki</label><input id="cfgKi" value="0"></div></div>';
    h+='<div class="cfg-row"><div><label>Kd</label><input id="cfgKd" value="0"></div><div></div></div>'}
  else if(t==="sum"){h+='<div class="cfg-row"><div><label>Sinais (+ ou -)</label><input id="cfgSigns" value="+ -" placeholder="+ - +"></div><div></div></div>'}
  cf.innerHTML=h}
function confirmAdd(){
  if(!pendingType)return;
  var t=pendingType,p={};
  if(t==="tf"||t==="sensor"||t==="actuator"){p.num=(document.getElementById("cfgNum")||{}).value||"1";p.den=(document.getElementById("cfgDen")||{}).value||"1"}
  else if(t==="gain"){p.k=(document.getElementById("cfgK")||{}).value||"1"}
  else if(t==="pid"){p.kp=(document.getElementById("cfgKp")||{}).value||"1";p.ki=(document.getElementById("cfgKi")||{}).value||"0";p.kd=(document.getElementById("cfgKd")||{}).value||"0"}
  else if(t==="sum"){p.signs=(document.getElementById("cfgSigns")||{}).value||"+ -"}
  addBP(t,p);closeModal()}
/* ===== POLY MATH ===== */
function pTrim(p){var a=p.slice();while(a.length>1&&Math.abs(a[a.length-1])<1e-14)a.pop();return a.length?a:[0]}
function pAdd(a,b){var r=[];for(var i=0;i<Math.max(a.length,b.length);i++)r.push((a[i]||0)+(b[i]||0));return pTrim(r)}
@@ -574,6 +947,7 @@ def _load_visual_editor_html():
/* ===== PF (poly fraction) ===== */
function pfC(c){return{n:[c],d:[1]}}
function pfZ(a){var t=pTrim(a.n);return t.length===1&&Math.abs(t[0])<1e-14}
function pfAdd(a,b){return{n:pAdd(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)}}
function pfSub(a,b){return{n:pSub(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)}}
function pfMul(a,b){return{n:pMul(a.n,b.n),d:pMul(a.d,b.d)}}
function pfDiv(a,b){return{n:pMul(a.n,b.d),d:pMul(a.d,b.n)}}
@@ -597,6 +971,76 @@ def _load_visual_editor_html():
  var ks=Object.keys(co).map(Number);if(!ks.length)return[0];var mx=Math.max.apply(null,ks),r=[];
  for(var i=0;i<=mx;i++)r.push(co[i]||0);return pTrim(r)}
/* ===== SS PARSER (State Space) ===== */
function parseMat(s){
  s=(s||"").trim();
  var rows=s.split(";");
  return rows.map(function(r){return r.trim().split(/[\s,]+/).map(Number)})}
function ssToTF(As,Bs,Cs,Ds){
  /* For SISO: T(s) = C(sI-A)^-1 B + D computed symbolically via det/adj */
  var A=parseMat(As),B=parseMat(Bs),C=parseMat(Cs),D=parseMat(Ds);
  var n=A.length;
  if(n===0)return{n:[D[0]?D[0][0]:0],d:[1]};
  /* Build characteristic polynomial det(sI - A) using Fadeev-LeVerrier */
  /* For small n (1-4), this is efficient enough */
  var M=[];for(var i=0;i<n;i++){M.push([]);for(var j=0;j<n;j++)M[i].push(i===j?[A[i][j]*-1,1]:[-A[i][j]])} /* sI - A as poly entries */
  /* For n<=3, compute directly; for larger, use approximation */
  if(n===1){
    var a00=A[0][0],b0=B[0][0],c0=C[0][0],d0=D[0]?D[0][0]:0;
    return{n:pAdd([d0*(-a00)+c0*b0],[d0]),d:[-a00,1]}}
  if(n===2){
    var tr=A[0][0]+A[1][1],dt=A[0][0]*A[1][1]-A[0][1]*A[1][0];
    var den=[dt,-tr,1]; /* s^2 - tr*s + det */
    /* adj(sI-A) * B then C * that */
    var adj00=[-A[1][1],1],adj01=[A[0][1]],adj10=[A[1][0]],adj11=[-A[0][0],1];
    var v0=pAdd(pScl(adj00,B[0][0]),pScl(adj01,B[1][0]));
    var v1=pAdd(pScl(adj10,B[0][0]),pScl(adj11,B[1][0]));
    var num=pAdd(pScl(v0,C[0][0]),pScl(v1,C[0].length>1?C[0][1]:0));
    var d0v=D[0]?D[0][0]:0;
    if(Math.abs(d0v)>1e-14)num=pAdd(num,pMul([d0v],den));
    return pfReduce({n:num,d:den})}
  /* General case: numerical via companion form */
  /* Use Fadeev-LeVerrier for char poly, then numerical adj */
  var cp=[1];var Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(0)}
  for(var k=1;k<=n;k++){
    var prev=Mk;var AM=[];
    for(var i=0;i<n;i++){AM.push([]);for(var j=0;j<n;j++){var s2=0;for(var l=0;l<n;l++)s2+=A[i][l]*prev[l][j];AM[i].push(s2)}}
    if(k===1)for(var i=0;i<n;i++)for(var j=0;j<n;j++)AM[i][j]+=(i===j?1:0);
    var tk=0;for(var i=0;i<n;i++)tk+=AM[i][i];
    var ck=-tk/k;cp.unshift(ck);
    Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(AM[i][j]+(i===j?ck:0))}}
  /* cp is den coeffs [c0,c1,...,1] ascending */
  var den=cp;
  /* num: evaluate C*adj(sI-A)*B numerically at n+1 points, fit poly */
  var pts=[];for(var k=0;k<=n;k++){
    var sv=k*10+1;
    var sIA=[];for(var i=0;i<n;i++){sIA.push([]);for(var j=0;j<n;j++)sIA[i].push((i===j?sv:0)-A[i][j])}
    /* invert sIA by Gauss-Jordan */
    var aug=[];for(var i=0;i<n;i++){aug.push([]);for(var j=0;j<2*n;j++)aug[i].push(j<n?sIA[i][j]:(i===j-n?1:0))}
    for(var c2=0;c2<n;c2++){var mx2=c2;for(var r=c2+1;r<n;r++)if(Math.abs(aug[r][c2])>Math.abs(aug[mx2][c2]))mx2=r;
      var tmp=aug[c2];aug[c2]=aug[mx2];aug[mx2]=tmp;var piv=aug[c2][c2];if(Math.abs(piv)<1e-30)continue;
      for(var j=0;j<2*n;j++)aug[c2][j]/=piv;
      for(var r=0;r<n;r++){if(r===c2)continue;var f=aug[r][c2];for(var j=0;j<2*n;j++)aug[r][j]-=f*aug[c2][j]}}
    var inv=[];for(var i=0;i<n;i++){inv.push([]);for(var j=0;j<n;j++)inv[i].push(aug[i][j+n])}
    /* C * inv * B */
    var CiB=0;for(var i=0;i<(C[0]||[]).length;i++)for(var j=0;j<(B[0]||[]).length;j++){var v3=0;for(var l=0;l<n;l++)v3+=inv[i][l]*B[l][j];CiB+=C[0][i]*v3}
    var d0v=D[0]?D[0][0]:0;
    pts.push({s:sv,v:CiB+d0v*pEv(den,sv)})}
  /* Lagrange interpolation for numerator poly (degree <= n) */
  var numC=[];for(var i=0;i<=n;i++){
    var li=1,yi=pts[i].v;for(var j=0;j<=n;j++){if(j===i)continue;li*=(pts[i].s-pts[j].s)}
    numC.push(yi/li)}
  /* Build num poly from pts using Newton form... simpler: Vandermonde */
  var V=[];for(var i=0;i<=n;i++){V.push([]);for(var j=0;j<=n;j++)V[i].push(Math.pow(pts[i].s,j))}
  var rhs=pts.map(function(p){return p.v});
  /* Solve Vandermonde system */
  for(var c2=0;c2<=n;c2++){var mx2=c2;for(var r=c2+1;r<=n;r++)if(Math.abs(V[r][c2])>Math.abs(V[mx2][c2]))mx2=r;
    var tmp=V[c2];V[c2]=V[mx2];V[mx2]=tmp;var tmp2=rhs[c2];rhs[c2]=rhs[mx2];rhs[mx2]=tmp2;
    var piv=V[c2][c2];if(Math.abs(piv)<1e-30)continue;for(var j=c2;j<=n;j++)V[c2][j]/=piv;rhs[c2]/=piv;
    for(var r=0;r<=n;r++){if(r===c2)continue;var f=V[r][c2];for(var j=c2;j<=n;j++)V[r][j]-=f*V[c2][j];rhs[r]-=f*rhs[c2]}}
  var num=rhs.map(function(v){return Math.abs(v)<1e-10?0:v});
  return pfReduce({n:pTrim(num),d:pTrim(den)})}
/* ===== FORMAT ===== */
function fN(n){if(Math.abs(n-Math.round(n))<1e-8)return Math.round(n).toString();return n.toFixed(4).replace(/0+$/,"").replace(/\.$/,"")}
function fP(c){c=pTrim(c);var ts=[];for(var i=c.length-1;i>=0;i--){var v=c[i];if(Math.abs(v)<1e-10)continue;var t;
@@ -606,34 +1050,49 @@ def _load_visual_editor_html():
/* ===== BLOCK TF ===== */
function bTF(nd){var p=nd.params||{},t=nd.type;
  if(t==="tf"||t==="sensor")return{n:parseP(p.num||"1"),d:parseP(p.den||"1")};
  if(t==="tf"||t==="sensor"||t==="actuator")return{n:parseP(p.num||"1"),d:parseP(p.den||"1")};
  if(t==="gain")return pfC(parseFloat(p.k)||1);
  if(t==="int")return{n:[1],d:[0,1]};if(t==="der")return{n:[0,1],d:[1]};
  if(t==="pid"){var kp=parseFloat(p.kp)||0,ki=parseFloat(p.ki)||0,kd=parseFloat(p.kd)||0;
    if(ki===0&&kd===0)return pfC(kp||1);if(ki===0)return{n:[kp,kd],d:[1]};return{n:[ki,kp,kd],d:[0,1]}}
  return pfC(1)}
/* ===== SOLVER ===== */
/* ===== SOLVER (Gaussian elimination on signal-flow graph) ===== */
function solve(nodes,edges){
  if(!nodes.length)return{e:"Adicione blocos."};
  var inp=null,out=null;nodes.forEach(function(n){if(n.type==="input")inp=n;if(n.type==="output")out=n});
  if(!inp)return{e:"Adicione Entrada R(s)."};if(!out)return{e:"Adicione Saida Y(s)."};if(!edges.length)return{e:"Conecte os blocos."};
  var N=nodes.length,ix={};nodes.forEach(function(n,i){ix[n.id]=i});
  /* Build system of equations: for each node i, X_i = sum(contributions) */
  var A=[],b=[];for(var i=0;i<N;i++){A.push([]);for(var j=0;j<N;j++)A[i].push(pfC(0));b.push(pfC(0))}
  for(var i=0;i<N;i++){var nd=nodes[i];A[i][i]=pfC(1);
    if(nd.type==="input"){b[i]=pfC(1);continue}
    var inc=edges.filter(function(e){return e.dst===nd.id});
    if(nd.type==="sum"){var sg=((nd.params||{}).signs||"+ -").trim().split(/\s+/);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;var pi=parseInt((e.dstPort||"in0").replace("in",""))||0;
        var sign=sg[pi]==="-"?-1:1;A[i][si]=pfSub(A[i][si],pfC(sign))})}
    else{var tf=bTF(nd);inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;A[i][si]=pfSub(A[i][si],tf)})}}
    if(nd.type==="sum"){
      /* Summing junction: output = sum of signed inputs */
      var sg=((nd.params||{}).signs||"+ -").trim().split(/\s+/);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;
        var pi=parseInt((e.dstPort||"in0").replace("in",""))||0;
        var sign=sg[pi]==="-"?-1:1;
        A[i][si]=pfSub(A[i][si],pfC(sign))})
    } else {
      /* Regular block: output = TF * input */
      var tf=bTF(nd);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;
        A[i][si]=pfSub(A[i][si],tf)})
    }
  }
  /* Gaussian elimination with partial pivoting */
  for(var c=0;c<N;c++){var pv=-1;for(var r=c;r<N;r++)if(!pfZ(A[r][c])){pv=r;break}
    if(pv<0)return{e:"Sistema singular."};if(pv!==c){var t=A[c];A[c]=A[pv];A[pv]=t;var t2=b[c];b[c]=b[pv];b[pv]=t2}
    if(pv<0)return{e:"Sistema singular - verifique as conexoes."};
    if(pv!==c){var t=A[c];A[c]=A[pv];A[pv]=t;var t2=b[c];b[c]=b[pv];b[pv]=t2}
    for(var r=c+1;r<N;r++){if(pfZ(A[r][c]))continue;var f=pfDiv(A[r][c],A[c][c]);
      for(var j=c;j<N;j++)A[r][j]=pfSub(A[r][j],pfMul(f,A[c][j]));b[r]=pfSub(b[r],pfMul(f,b[c]))}}
  /* Back substitution */
  var x=[];for(var i=0;i<N;i++)x.push(pfC(0));
  for(var i=N-1;i>=0;i--){var s=b[i];for(var j=i+1;j<N;j++)s=pfSub(s,pfMul(A[i][j],x[j]));x[i]=pfDiv(s,A[i][i])}
  var oi=ix[out.id],tf={n:pTrim(x[oi].n),d:pTrim(x[oi].d)};
  /* Normalize */
  if(tf.d[tf.d.length-1]<0){tf.n=pScl(tf.n,-1);tf.d=pScl(tf.d,-1)}
  var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
  tf=pfReduce(tf);return{tf:tf}}
@@ -651,7 +1110,7 @@ def _load_visual_editor_html():
    if(cAbs(pr)<1e-30)continue;var dl=cDiv(v,pr);rs[i].r-=dl.r;rs[i].i-=dl.i;mx=Math.max(mx,cAbs(dl))}if(mx<1e-12)break}
  rs.forEach(function(r){if(Math.abs(r.i)<1e-8)r.i=0});return rs}
/* ===== STEP RESPONSE (Stehfest) ===== */
/* ===== STEP/FORCED RESPONSE (Stehfest) ===== */
function fact(n){var r=1;for(var i=2;i<=n;i++)r*=i;return r}
var SN=14,SV=[];(function(){for(var i=1;i<=SN;i++){var s=0,k0=Math.floor((i+1)/2),k1=Math.min(i,SN/2);
  for(var k=k0;k<=k1;k++)s+=Math.pow(k,SN/2)*fact(2*k)/(fact(SN/2-k)*fact(k)*fact(k-1)*fact(i-k)*fact(2*k-i));
@@ -660,8 +1119,6 @@ def _load_visual_editor_html():
  for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;
    for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv)*sv;if(Math.abs(dv)<1e-30)continue;s+=SV[j]*nv/dv}
    ys.push(s*ln2/t)}return{t:ts,y:ys}}
/* ===== FORCED RESPONSE (generalized) ===== */
function forceResp(tf,sig,tMax,nP){var ln2=Math.LN2,ts=[],ys=[],us=[];var om=2*Math.PI;
  for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;
    for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv);
@@ -682,7 +1139,7 @@ def _load_visual_editor_html():
    var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc);re.push(T.r);im.push(T.i)}
  return{re:re,im:im}}
/* ===== LGR (Root Locus) ===== */
/* ===== LGR ===== */
function lgr(tf,nK){var kMax=200,branches=[];var nt=pTrim(tf.n),dt=pTrim(tf.d);var np=dt.length-1;
  if(np<=0)return branches;
  for(var i=0;i<np;i++)branches.push({re:[],im:[]});var prev=null;
@@ -749,7 +1206,7 @@ def _load_visual_editor_html():
  if(Math.abs(yf)>1e-6)for(var i=y.length-1;i>=0;i--)if(Math.abs(y[i]-yf)>.02*Math.abs(yf)){ts2=i<y.length-1?t[i+1]:t[i];break}
  return{"Valor Final":fN(yf),"Sobressinal":fN(os)+"%","T. Subida":isNaN(tr)?"N/A":fN(tr)+"s","T. Acomod.":isNaN(ts2)?"N/A":fN(ts2)+"s","Pico":fN(ym)}}
/* ===== CANVAS CHART ===== */
/* ===== CANVAS CHARTS ===== */
function chart(id,xD,yD,xL,yL,col,logX){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=280,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
@@ -769,7 +1226,6 @@ def _load_visual_editor_html():
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}
/* ===== CHART 2 TRACES (input+output) ===== */
function chart2(id,xD,y1,y2,xL,yL){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=280,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
@@ -794,7 +1250,6 @@ def _load_visual_editor_html():
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}
/* ===== CHART XY (Nyquist) ===== */
function chartXY(id,xD,yD,xL,yL){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=380,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:45},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
@@ -918,12 +1373,15 @@ def _load_visual_editor_html():
  document.getElementById("manualSec").classList.toggle("vis",m==="manual");
  document.getElementById("btnCalcMain").innerHTML=m==="diag"?"&#9654; CALCULAR DIAGRAMA":"&#9654; CALCULAR T(s)"}
function setSubMode(m){curSubMode=m;
  ["subDirect","subClosed","subOpen","subSS"].forEach(function(id){document.getElementById(id).classList.toggle("active",id==="sub"+m.charAt(0).toUpperCase()+m.slice(1))});
  document.getElementById("subDirect").classList.toggle("active",m==="direct");
  document.getElementById("subClosed").classList.toggle("active",m==="closed");
  document.getElementById("subOpen").classList.toggle("active",m==="open");
  document.getElementById("subSS").classList.toggle("active",m==="ss");
  document.getElementById("manDirect").style.display=m==="direct"?"block":"none";
  document.getElementById("manClosed").style.display=m==="closed"?"block":"none";
  document.getElementById("manOpen").style.display=m==="open"?"block":"none"}
  document.getElementById("manOpen").style.display=m==="open"?"block":"none";
  document.getElementById("manSS").style.display=m==="ss"?"block":"none"}
function onCalc(){
  if(curMode==="manual"){
@@ -933,11 +1391,25 @@ def _load_visual_editor_html():
    } else if(curSubMode==="closed"){
      var gn=parseP(document.getElementById("manGN").value),gd=parseP(document.getElementById("manGD").value);
      var hn=parseP(document.getElementById("manHN").value),hd=parseP(document.getElementById("manHD").value);
      /* T(s) = G/(1+GH) = (Gn*Hd) / (Gd*Hd + Gn*Hn) */
      tf={n:pMul(gn,hd),d:pAdd(pMul(gd,hd),pMul(gn,hn))};
    } else {
    } else if(curSubMode==="open"){
      var gn=parseP(document.getElementById("manOGN").value),gd=parseP(document.getElementById("manOGD").value);
      var hn=parseP(document.getElementById("manOHN").value),hd=parseP(document.getElementById("manOHD").value);
      /* L(s) = G*H = (Gn*Hn) / (Gd*Hd) */
      tf={n:pMul(gn,hn),d:pMul(gd,hd)};
    } else if(curSubMode==="ss"){
      /* State Space */
      try{
        tf=ssToTF(
          document.getElementById("manA").value,
          document.getElementById("manB").value,
          document.getElementById("manC").value,
          document.getElementById("manD").value);
      }catch(e){
        var rd=document.getElementById("res"),rb=document.getElementById("rb");
        rd.classList.add("vis");rb.innerHTML='<div class="ebox">Erro no espaco de estados: '+esc(String(e))+'</div>';
        rd.scrollIntoView({behavior:"smooth"});return}
    }
    var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
    tf=pfReduce(tf);showRes(tf);return;
@@ -951,16 +1423,20 @@ def _load_visual_editor_html():
var cw=document.getElementById("cw"),cv=document.getElementById("cv"),wSvg=document.getElementById("wires");
function nxtId(){var m=0;model.nodes.forEach(function(n){var v=parseInt(n.id.replace("n",""))||0;if(v>m)m=v});return"n"+(m+1)}
function ptr(e){if(e.touches&&e.touches.length)return{x:e.touches[0].clientX,y:e.touches[0].clientY};if(e.changedTouches&&e.changedTouches.length)return{x:e.changedTouches[0].clientX,y:e.changedTouches[0].clientY};return{x:e.clientX,y:e.clientY}}
var BL={tf:"Funcao Transf.",gain:"Ganho",sum:"Somador",int:"Integrador",der:"Derivador",pid:"PID",sensor:"Sensor",input:"Entrada",output:"Saida",branch:"Ramificacao"};
function dPar(t){if(t==="tf")return{num:"1",den:"s+1"};if(t==="gain")return{k:"1"};if(t==="sum")return{signs:"+ -"};if(t==="pid")return{kp:"1",ki:"0",kd:"0"};if(t==="sensor")return{num:"1",den:"1"};if(t==="input")return{label:"R(s)"};if(t==="output")return{label:"Y(s)"};return{}}
var BL={tf:"Planta",gain:"Ganho",sum:"Somador",int:"Integrador",der:"Derivador",pid:"PID",sensor:"Sensor",actuator:"Atuador",input:"Entrada",output:"Saida",branch:"Ramificacao"};
function dPar(t){if(t==="tf")return{num:"1",den:"s+1"};if(t==="gain")return{k:"1"};if(t==="sum")return{signs:"+ -"};if(t==="pid")return{kp:"1",ki:"0",kd:"0"};if(t==="sensor"||t==="actuator")return{num:"1",den:"1"};if(t==="input")return{label:"R(s)"};if(t==="output")return{label:"Y(s)"};return{}}
function gPC(t,p){if(t==="input")return{i:[],o:[{id:"out0"}]};if(t==="output")return{i:[{id:"in0"}],o:[]};if(t==="branch")return{i:[{id:"in0"}],o:[{id:"out0"},{id:"out1"}]};
  if(t==="sum"){var sg=(p&&p.signs?p.signs:"+ -").trim().split(/\s+/);return{i:sg.map(function(s,i){return{id:"in"+i,sign:s}}),o:[{id:"out0"}]}}return{i:[{id:"in0"}],o:[{id:"out0"}]}}
function bTxt(n){var p=n.params||{};if(n.type==="tf")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
function bTxt(n){var p=n.params||{};if(n.type==="tf"||n.type==="actuator")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="gain")return"K="+(p.k||"1");if(n.type==="pid")return'<div class="block-tf-disp">Kp='+(p.kp||"0")+" Ki="+(p.ki||"0")+" Kd="+(p.kd||"0")+'</div>';
  if(n.type==="sensor")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="input")return p.label||"R(s)";if(n.type==="output")return p.label||"Y(s)";if(n.type==="sum")return"\u03a3";if(n.type==="int")return"1/s";if(n.type==="der")return"s";return""}
/* addB: quick add (toolbar) */
function addB(t){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:dPar(t)};
  if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
/* addBP: add with params (modal) */
function addBP(t,p){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:Object.assign(dPar(t),p)};
  if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
function delSel(){if(!selId)return;model.edges=model.edges.filter(function(e){return e.src!==selId&&e.dst!==selId});model.nodes=model.nodes.filter(function(n){return n.id!==selId});selId=null;conSt=null;render()}
function clrAll(){model={nodes:[],edges:[]};selId=null;conSt=null;render();document.getElementById("res").classList.remove("vis")}
function autoLay(){if(!model.nodes.length)return;var lv={},aj={};model.edges.forEach(function(e){if(!aj[e.src])aj[e.src]=[];aj[e.src].push(e.dst)});
@@ -1020,7 +1496,7 @@ def _load_visual_editor_html():
  var pa=document.getElementById("pA");if(!selId){pa.innerHTML='<span class="hint">Selecione um bloco.</span>';return}
  var nd=model.nodes.find(function(n){return n.id===selId});if(!nd){pa.innerHTML="";return}var p=nd.params||{};
  var h='<div style="font-size:11px;color:var(--txm);margin-bottom:6px">'+BL[nd.type]+' <b style="color:var(--tx)">'+nd.id+'</b></div>';
  if(nd.type==="tf"||nd.type==="sensor"){h+=pI("Num","num",p.num||"1");h+=pI("Den","den",p.den||"1")}
  if(nd.type==="tf"||nd.type==="sensor"||nd.type==="actuator"){h+=pI("Num","num",p.num||"1");h+=pI("Den","den",p.den||"1")}
  else if(nd.type==="gain")h+=pI("K","k",p.k||"1");
  else if(nd.type==="sum")h+=pI("Sinais","signs",p.signs||"+ -");
  else if(nd.type==="pid"){h+=pI("Kp","kp",p.kp||"1");h+=pI("Ki","ki",p.ki||"0");h+=pI("Kd","kd",p.kd||"0")}
@@ -1030,85 +1506,325 @@ def _load_visual_editor_html():
function pI(l,k,v){return'<div class="pg"><label>'+l+'</label><input data-key="'+k+'" value="'+esc(String(v))+'"></div>'}
document.querySelectorAll(".tb[data-add]").forEach(function(b){b.addEventListener("click",function(){addB(b.dataset.add)})});
document.getElementById("btnDel").addEventListener("click",delSel);document.getElementById("btnClear").addEventListener("click",clrAll);document.getElementById("btnAuto").addEventListener("click",autoLay);
document.addEventListener("keydown",function(e){if(e.target.tagName==="INPUT")return;if(e.key==="Delete"||e.key==="Backspace")delSel();if(e.key==="Escape"){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")})}});
document.addEventListener("keydown",function(e){if(e.target.tagName==="INPUT")return;if(e.key==="Delete"||e.key==="Backspace")delSel();if(e.key==="Escape"){conSt=null;closeModal();document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")})}});
render();new ResizeObserver(function(){rW()}).observe(cw);
</script></body></html>'''


# =====================================================
# APLICACAO PRINCIPAL
# =====================================================
# ══════════════════════════════════════════════════
# TELA INICIAL (SELECAO DE MODO)
# ══════════════════════════════════════════════════

def tela_inicial():
    st.markdown("""
    <style>
    .mode-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 2px solid #333654;
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        height: 100%;
    }
    .mode-card:hover {
        border-color: #5b6be0;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(91,107,224,0.2);
    }
    .mode-icon { font-size: 42px; margin-bottom: 12px; }
    .mode-title { font-size: 20px; font-weight: 700; margin-bottom: 8px; color: #e0e4f0; }
    .mode-desc { font-size: 14px; color: #8890b0; line-height: 1.6; }
    .welcome-title {
        text-align: center; font-size: 32px; font-weight: 700;
        margin-bottom: 8px; color: #e0e4f0;
    }
    .welcome-sub {
        text-align: center; font-size: 16px; color: #8890b0;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("Modelagem e Analise de Sistemas de Controle")
    st.markdown("")
    st.markdown("")
    st.markdown('<div class="welcome-title">Sistema de Modelagem e Analise de Controle</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Escolha o modo de trabalho para comecar</div>',
                unsafe_allow_html=True)

    inicializar_blocos()
    col1, col2, col3 = st.columns(3, gap="large")

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
    with col1:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">&#128203;</div>
            <div class="mode-title">Modo Lista</div>
            <div class="mode-desc">
                Adicione blocos a uma lista, defina conexoes
                (serie, paralelo, feedback) e obtenha a FT equivalente.
                Suporte a espaco de estados (A, B, C, D).
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Selecionar Modo Lista", use_container_width=True, type="primary"):
            st.session_state.modo_selecionado = 'lista'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">&#128200;</div>
            <div class="mode-title">Modo Classico</div>
            <div class="mode-desc">
                Insira Planta, Controlador e Sensor por funcao de
                transferencia. Analise em malha aberta ou fechada
                com ganho K ajustavel.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Selecionar Modo Classico", use_container_width=True, type="primary"):
            st.session_state.modo_selecionado = 'classico'
            st.rerun()

    with col3:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">&#127912;</div>
            <div class="mode-title">Modo Canvas</div>
            <div class="mode-desc">
                Editor visual de diagrama de blocos. Arraste, conecte
                e calcule a FT equivalente graficamente.
                Entrada manual e espaco de estados integrados.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Selecionar Modo Canvas", use_container_width=True, type="primary"):
            st.session_state.modo_selecionado = 'canvas'
            st.rerun()


# ══════════════════════════════════════════════════
# MODO LISTA
# ══════════════════════════════════════════════════

def modo_lista():
    st.title("Modo Lista - Blocos e Conexoes")

    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Adicionar Bloco")

        nome = st.text_input("Nome do bloco", value=f"B{len(st.session_state.blocos)+1}")
        tipo = st.selectbox("Tipo", list(BLOCK_TYPES.keys()))

        representacao = st.radio(
            "Representacao",
            ['Funcao de Transferencia', 'Espaco de Estados'],
            horizontal=True)

        if representacao == 'Funcao de Transferencia':
            numerador = st.text_input("Numerador", placeholder="ex: s+1")
            denominador = st.text_input("Denominador", placeholder="ex: s^2+2*s+3")
            A_str = B_str = C_str = D_str = ''
        else:
            st.markdown("**Matrizes do Espaco de Estados**")
            st.caption("Use `;` para separar linhas. Ex: `0 1; -2 -3`")
            A_str = st.text_input("Matriz A (nxn)", value="0 1; -2 -3")
            B_str = st.text_input("Matriz B (nxm)", value="0; 1")
            C_str = st.text_input("Matriz C (pxn)", value="1 0")
            D_str = st.text_input("Matriz D (pxm)", value="0")
            numerador = denominador = ''

        if st.button("Adicionar Bloco", type="primary", use_container_width=True):
            if not nome.strip():
                st.error("Informe um nome.")
            elif any(st.session_state.blocos['nome'] == nome):
                st.error(f"Bloco '{nome}' ja existe.")
            else:
                ok, msg = adicionar_bloco(
                    nome, tipo, representacao, numerador, denominador,
                    A_str, B_str, C_str, D_str)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # --- Area principal ---
    tab_blocos, tab_conexoes, tab_analise = st.tabs(
        ["Blocos", "Conexoes", "Analise"])

    with tab_blocos:
        st.subheader("Blocos Definidos")
        if st.session_state.blocos.empty:
            st.info("Nenhum bloco adicionado. Use a barra lateral para adicionar.")
        else:
            for idx, row in st.session_state.blocos.iterrows():
                with st.expander(f"{row['nome']} ({row['tipo']}) - {row['representacao']}", expanded=False):
                    tf_obj = row['tf']
                    num_str = _tf_to_str(tf_obj.num[0][0])
                    den_str = _tf_to_str(tf_obj.den[0][0])
                    st.latex(f"\\frac{{{num_str}}}{{{den_str}}}")

                    if row['representacao'] == 'Espaco de Estados' and row['A']:
                        st.markdown(f"**A:** `{row['A']}`  |  **B:** `{row['B']}`")
                        st.markdown(f"**C:** `{row['C']}`  |  **D:** `{row['D']}`")

                    polos = ctrl.poles(tf_obj)
                    zeros = ctrl.zeros(tf_obj)
                    st.markdown(f"**Polos:** {_format_complex_list(polos)}")
                    st.markdown(f"**Zeros:** {_format_complex_list(zeros)}")

                    if st.button(f"Remover {row['nome']}", key=f"rm_{idx}"):
                        remover_bloco(row['nome'])
                        st.rerun()

    with tab_conexoes:
        st.subheader("Definir Conexoes entre Blocos")
        if len(st.session_state.blocos) < 2:
            st.info("Adicione pelo menos 2 blocos para definir conexoes.")
        else:
            nomes_disponiveis = list(st.session_state.blocos['nome'])

            st.markdown("**Nova Conexao**")
            c1, c2 = st.columns(2)
            with c1:
                tipo_conexao = st.selectbox("Tipo de conexao", CONNECTION_TYPES)
            with c2:
                if tipo_conexao in ['Realimentacao Negativa', 'Realimentacao Positiva']:
                    st.caption("Bloco 1 = G(s) direto, Bloco 2 = H(s) feedback")

            blocos_sel = st.multiselect(
                "Blocos (na ordem desejada)", nomes_disponiveis,
                max_selections=10 if tipo_conexao in ['Serie', 'Paralelo'] else 2)

            if st.button("Adicionar Conexao", type="primary"):
                if len(blocos_sel) < 2:
                    st.error("Selecione pelo menos 2 blocos.")
                else:
                    st.session_state.conexoes.append({
                        'tipo': tipo_conexao,
                        'blocos': blocos_sel,
                    })
                    st.success(f"Conexao '{tipo_conexao}' adicionada: {' -> '.join(blocos_sel)}")
                    st.rerun()

            if st.session_state.conexoes:
                st.markdown("---")
                st.markdown("**Conexoes Definidas**")
                for i, con in enumerate(st.session_state.conexoes):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        simbolo = {'Serie': ' --> ', 'Paralelo': ' || ',
                                   'Realimentacao Negativa': ' -fb-> ',
                                   'Realimentacao Positiva': ' +fb-> '}
                        st.markdown(
                            f"**{i+1}.** {con['tipo']}: "
                            f"`{simbolo.get(con['tipo'], ' -> ').join(con['blocos'])}`")
                    with col_b:
                        if st.button("X", key=f"rmcon_{i}"):
                            st.session_state.conexoes.pop(i)
                            st.rerun()

    with tab_analise:
        st.subheader("Analise do Sistema")
        if st.session_state.blocos.empty:
            st.info("Adicione blocos primeiro.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                tipo_malha = st.selectbox("Tipo de analise", ["Malha Aberta", "Malha Fechada"])
                usar_ganho = st.checkbox("Ganho K ajustavel")
                K = st.slider("K", 0.1, 100.0, 1.0, 0.1) if usar_ganho else 1.0
            with col2:
                analise_opcoes = ANALYSIS_OPTIONS[
                    "malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
                analises = st.multiselect("Analises", analise_opcoes, default=[analise_opcoes[0]])
                entrada = st.selectbox("Sinal de entrada", INPUT_SIGNALS)

            if st.button("Calcular e Analisar", type="primary", use_container_width=True):
                try:
                    sistema = simplificar_diagrama(
                        st.session_state.blocos, st.session_state.conexoes)

                    if usar_ganho and K != 1.0:
                        sistema = TransferFunction([K], [1]) * sistema

                    if tipo_malha == "Malha Fechada" and not any(
                            c['tipo'].startswith('Realimentacao')
                            for c in st.session_state.conexoes):
                        sistema = ctrl.feedback(sistema, TransferFunction([1], [1]))

                    sistema = ctrl.minreal(sistema, verbose=False)

                    st.markdown("### Funcao de Transferencia Equivalente")
                    num_str = _tf_to_str(sistema.num[0][0])
                    den_str = _tf_to_str(sistema.den[0][0])
                    st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")

                    executar_analises(sistema, analises, entrada, tipo_malha)

                except Exception as e:
                    st.error(f"Erro: {e}")


# ══════════════════════════════════════════════════
# MODO CLASSICO
# ══════════════════════════════════════════════════

def modo_classico():
    st.title("Modo Classico - Funcao de Transferencia")

    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Adicionar Blocos")

        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Atuador'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("Adicionar"):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            if sucesso:
                st.success(mensagem)
        if st.button("Adicionar", type="primary", use_container_width=True):
            ok, msg = adicionar_bloco(nome, tipo, 'Funcao de Transferencia',
                                      numerador, denominador)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(mensagem)
                st.error(msg)

        if not st.session_state.blocos.empty:
            st.header("Excluir Blocos")
            st.markdown("---")
            st.header("Remover Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("Excluir"):
                mensagem = remover_bloco(excluir)
                st.success(mensagem)
                remover_bloco(excluir)
                st.rerun()

        st.markdown("---")
        st.header("Configuracoes")
        if st.button("Habilitar Calculo de Erro" if not st.session_state.calculo_erro_habilitado else "Desabilitar Calculo de Erro"):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
        if st.button(
            "Habilitar Calculo de Erro"
            if not st.session_state.calculo_erro_habilitado
            else "Desabilitar Calculo de Erro"
        ):
            st.session_state.calculo_erro_habilitado = (
                not st.session_state.calculo_erro_habilitado)
            st.rerun()

    # Calculo de erro
    if st.session_state.calculo_erro_habilitado:
        st.subheader("Calculo de Erro Estacionario")
        col1, col2 = st.columns(2)
@@ -1117,57 +1833,37 @@ def main():
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
        if st.button("Calcular Erro Estacionario"):
            try:
                G, _ = converter_para_tf(num_erro, den_erro)
                tipo, Kp, Kv, Ka = constantes_de_erro(G)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tipo", tipo)
                c2.metric("Kp", formatar_numero(Kp))
                c3.metric("Kv", formatar_numero(Kv))
                c4.metric("Ka", formatar_numero(Ka))
            except Exception as e:
                st.error(f"Erro: {e}")

    # Painel de analise
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Tipo de Sistema")
        tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
        st.subheader("Configuracao")
        tipo_malha = st.selectbox("Tipo de Sistema", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustavel", value=False)

        if usar_ganho:
            K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1)
            st.info(f"Ganho K: {K:.2f}")
        else:
            K = 1.0
        K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1) if usar_ganho else 1.0

        st.subheader("Analises")
        analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
        analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
        chave = "malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"
        analise_opcoes = ANALYSIS_OPTIONS[chave]
        analises = st.multiselect("Escolha:", analise_opcoes, default=[analise_opcoes[0]])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("Resultados")

        if st.button("Executar Simulacao", use_container_width=True):
        if st.button("Executar Simulacao", use_container_width=True, type="primary"):
            try:
                df = st.session_state.blocos
                if df.empty:
@@ -1186,56 +1882,133 @@ def main():

                if tipo_malha == "Malha Aberta":
                    sistema = ganho_tf * planta
                    st.info(f"Sistema em Malha Aberta com K = {K:.2f}")
                    if controlador is not None:
                        sistema = controlador * sistema
                    st.info(f"Malha Aberta | K = {K:.2f}")
                else:
                    planta_com_ganho = ganho_tf * planta
                    sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                    st.info(f"Sistema em Malha Fechada com K = {K:.2f}")
                    st.info(f"Malha Fechada | K = {K:.2f}")

                for analise in analises:
                    st.markdown(f"### {analise}")
                # Mostrar FT equivalente
                num_str = _tf_to_str(sistema.num[0][0])
                den_str = _tf_to_str(sistema.den[0][0])
                st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")

                    if analise == 'Resposta no tempo':
                        fig, t_out, y = plot_resposta_temporal(sistema, entrada)
                        st.plotly_chart(fig, use_container_width=True)
                executar_analises(sistema, analises, entrada, tipo_malha)

                    elif analise == 'Desempenho':
                        desempenho = calcular_desempenho(sistema)
                        for chave, valor in desempenho.items():
                            st.markdown(f"**{chave}:** {valor}")
            except Exception as e:
                st.error(f"Erro durante a simulacao: {e}")

                    elif analise == 'Diagrama De Bode Magnitude':
                        fig = plot_bode(sistema, 'magnitude')
                        st.plotly_chart(fig, use_container_width=True)

                    elif analise == 'Diagrama De Bode Fase':
                        fig = plot_bode(sistema, 'fase')
                        st.plotly_chart(fig, use_container_width=True)
# ══════════════════════════════════════════════════
# MODO CANVAS
# ══════════════════════════════════════════════════

                    elif analise == 'Diagrama de Polos e Zeros':
                        fig = plot_polos_zeros(sistema)
                        st.plotly_chart(fig, use_container_width=True)
def modo_canvas():
    st.title("Modo Canvas - Editor Visual")

                    elif analise == 'LGR':
                        fig = plot_lgr(sistema)
                        st.plotly_chart(fig, use_container_width=True)
    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

                    elif analise == 'Nyquist':
                        fig, polos_spd, voltas, Z = plot_nyquist(sistema)
                        st.markdown(f"**Polos SPD (P):** {polos_spd}")
                        st.markdown(f"**Voltas (N):** {voltas}")
                        st.markdown(f"**Z = {Z} -> {'Estavel' if Z == 0 else 'Instavel'}**")
                        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("### Como usar")
        st.markdown("""
        **Diagrama de Blocos:**
        1. Clique **+ Adicionar Bloco** ou use os botoes rapidos
        2. Arraste para posicionar
        3. Conecte: porta azul (saida) -> porta verde (entrada)
        4. Edite parametros no painel lateral
        5. Clique **CALCULAR** para ver resultados
        **Entrada Manual:**
        1. Clique em "Entrada Manual"
        2. Escolha: T(s) Direta, Malha Fechada, Malha Aberta ou Espaco de Estados
        3. Preencha os campos
        4. Clique **CALCULAR T(s)**
        """)

    html_content = _load_visual_editor_html()
    components.html(html_content, height=1200, scrolling=True)


# ══════════════════════════════════════════════════
# FUNCOES AUXILIARES DE FORMATACAO
# ══════════════════════════════════════════════════

def _tf_to_str(coeffs):
    """Converte coeficientes para string LaTeX de polinomio."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_val = float(c)
        if abs(c_val) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c_val:.4g}")
        elif power == 1:
            if abs(c_val - 1) < 1e-10:
                terms.append("s")
            elif abs(c_val + 1) < 1e-10:
                terms.append("-s")
            else:
                terms.append(f"{c_val:.4g}s")
        else:
            if abs(c_val - 1) < 1e-10:
                terms.append(f"s^{{{power}}}")
            elif abs(c_val + 1) < 1e-10:
                terms.append(f"-s^{{{power}}}")
            else:
                terms.append(f"{c_val:.4g}s^{{{power}}}")
    if not terms:
        return "0"
    result = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"
    return result


def _format_complex_list(arr):
    """Formata lista de numeros complexos para exibicao."""
    if len(arr) == 0:
        return "Nenhum"
    parts = []
    for v in arr:
        if np.isreal(v):
            parts.append(f"{np.real(v):.4f}")
        else:
            parts.append(f"{np.real(v):.4f} {'+' if np.imag(v) >= 0 else '-'} {abs(np.imag(v)):.4f}j")
    return ", ".join(parts)

            except Exception as e:
                st.error(f"Erro durante a simulacao: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dica")
    st.sidebar.info("Experimente o Editor Visual para construir sistemas graficamente!")
# ══════════════════════════════════════════════════
# APLICACAO PRINCIPAL
# ══════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Modelagem de Sistemas de Controle",
        page_icon="&#9881;",
        layout="wide",
    )
    inicializar_estado()

if __name__ == "__main__":
    main()
    if st.session_state.modo_selecionado is None:
        tela_inicial()
    elif st.session_state.modo_selecionado == 'lista':
        modo_lista()
    elif st.session_state.modo_selecionado == 'classico':
        modo_classico()
    elif st.session_state.modo_selecionado == 'canvas':
        modo_canvas()


if __name__ == "__main__":
    main()
