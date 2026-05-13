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
from pathlib import Path

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
# EDITOR VISUAL - HTML EXTERNO
# Mantem o HTML separado do codigo Python
# =====================================================

def _load_visual_editor_html():
    html_path = Path(__file__).with_name("visual_editor.html")
    return html_path.read_text(encoding="utf-8")


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

