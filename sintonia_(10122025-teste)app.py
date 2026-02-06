# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Editor Visual Xcos - VERSÃO COMPLETAMENTE FUNCIONAL
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
import time

# =====================================================
# CONFIGURAÇÃO
# =====================================================

st.set_page_config(page_title="Editor Xcos - Sistemas de Controle", layout="wide")

# CSS
st.markdown("""
<style>
    .main > div {padding-top: 1rem;}
    .stButton > button {width: 100%; font-weight: bold;}
    h1 {color: #667eea;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    if np.isinf(valor): return '∞'
    elif np.isnan(valor): return '-'
    else: return f"{valor:.3f}"

# =====================================================
# FUNÇÕES DE TRANSFERÊNCIA
# =====================================================

def converter_para_tf(numerador_str, denominador_str):
    """Converte strings para função de transferência"""
    try:
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
        
        return TransferFunction(num_coeffs, den_coeffs)
    except Exception as e:
        raise Exception(f"Erro ao converter: {str(e)}")

# =====================================================
# ANÁLISE DE SISTEMAS
# =====================================================

def calcular_desempenho(tf):
    """Calcula métricas de desempenho"""
    try:
        polos = ctrl.poles(tf)
        gm, pm, wg, wp = margin(tf)
        gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf
        
        resultado = {
            'Margem de ganho': f"{formatar_numero(gm)} ({'∞' if gm == np.inf else f'{formatar_numero(gm_db)} dB'})",
            'Margem de fase': f"{formatar_numero(pm)}°",
        }
        
        ordem = len(tf.den[0][0]) - 1
        
        if ordem == 1:
            tau = -1 / polos[0].real if polos[0].real != 0 else float('inf')
            resultado.update({
                'Tipo': '1ª Ordem',
                'Const. tempo (τ)': f"{formatar_numero(tau)} s",
                'Temp. acomodação (Ts)': f"{formatar_numero(4 * tau)} s",
            })
        elif ordem == 2:
            wn = np.sqrt(np.prod(np.abs(polos))).real
            zeta = -np.real(polos[0]) / wn if wn > 0 else 0
            Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
            Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
            
            resultado.update({
                'Tipo': '2ª Ordem',
                'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
                'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
                'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
                'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s"
            })
        else:
            resultado['Tipo'] = f'{ordem}ª Ordem'
        
        return resultado
    except:
        return {'Erro': 'Não foi possível calcular'}

def estimar_tempo_final(tf):
    """Estima tempo para simulação"""
    try:
        polos = ctrl.poles(tf)
        if len(polos) == 0 or any(np.real(p) > 1e-6 for p in polos):
            return 20.0
        partes_reais = [np.real(p) for p in polos if np.real(p) < -1e-6]
        if not partes_reais:
            return 100.0
        sigma = max(partes_reais)
        return np.clip(6 / abs(sigma), 10, 500)
    except:
        return 50.0

# =====================================================
# FUNÇÕES DE PLOTAGEM
# =====================================================

def plot_resposta_temporal(sistema, entrada='Degrau'):
    """Plota resposta temporal"""
    tempo_final = estimar_tempo_final(sistema)
    t = np.linspace(0, tempo_final, 1000)
    
    if entrada == 'Degrau':
        u = np.ones_like(t)
        t_out, y = step_response(sistema, t)
    elif entrada == 'Rampa':
        u = t
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    elif entrada == 'Senoidal':
        u = np.sin(2*np.pi*t)
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    else:  # Impulso
        u = np.concatenate([[1], np.zeros(len(t)-1)])
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=u[:len(t_out)], mode='lines', 
                             line=dict(dash='dash', color='blue', width=2), name='Entrada'))
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines', 
                             line=dict(color='red', width=3), name='Saída'))
    
    fig.update_layout(
        title=f'Resposta ao {entrada}',
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_bode(sistema):
    """Diagrama de Bode"""
    sys = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Magnitude', 'Fase'), vertical_spacing=0.15)
    
    fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='red', width=3)), row=2, col=1)
    
    fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Fase (°)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    return fig

def plot_polos_zeros(tf):
    """Diagrama de Polos e Zeros"""
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)
    
    fig = go.Figure()
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
                                marker=dict(symbol='circle', size=14, color='blue', 
                                          line=dict(width=2, color='white')), name='Zeros'))
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
                                marker=dict(symbol='x', size=16, color='red', 
                                          line=dict(width=3)), name='Polos'))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Diagrama de Polos e Zeros',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        height=500
    )
    
    return fig

def plot_lgr(sistema):
    """Lugar Geométrico das Raízes"""
    rlist, klist = root_locus(sistema, plot=False)
    
    fig = go.Figure()
    
    for r in rlist.T:
        fig.add_trace(go.Scatter(x=np.real(r), y=np.imag(r), mode='lines',
                                line=dict(color='blue', width=2), showlegend=False))
    
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
                                marker=dict(symbol='circle', size=12, color='green'), name='Zeros'))
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
                                marker=dict(symbol='x', size=14, color='red'), name='Polos'))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(title='Lugar Geométrico das Raízes (LGR)', 
                     xaxis_title='Parte Real', yaxis_title='Parte Imaginária', height=500)
    
    return fig

# =====================================================
# PROCESSAMENTO DO DIAGRAMA
# =====================================================

def processar_diagrama(blocos_lista, conexoes_lista):
    """Processa diagrama e retorna sistema equivalente"""
    try:
        if not blocos_lista:
            return None, "❌ Nenhum bloco no diagrama"
        
        # Criar funções de transferência
        tfs = {}
        
        for bloco in blocos_lista:
            bloco_id = bloco['id']
            tipo = bloco['tipo']
            config = bloco['config']
            
            if tipo == 'Transferência':
                tf = converter_para_tf(config['numerador'], config['denominador'])
                tfs[bloco_id] = tf
                
            elif tipo == 'Ganho':
                K = float(config.get('valor', 1))
                tfs[bloco_id] = TransferFunction([K], [1])
                
            elif tipo == 'Integrador':
                tfs[bloco_id] = TransferFunction([1], [1, 0])
                
            elif tipo == 'Somador':
                tfs[bloco_id] = TransferFunction([1], [1])
        
        # Processar conexões (simplificado: multiplicação em série)
        if len(blocos_lista) == 1:
            sistema_final = tfs[blocos_lista[0]['id']]
            msg = "✅ Sistema com 1 bloco"
        else:
            # Ordenar blocos pelas conexões
            if conexoes_lista:
                # Criar grafo
                grafo = {}
                for conn in conexoes_lista:
                    origem = conn['origem']
                    if origem not in grafo:
                        grafo[origem] = []
                    grafo[origem].append(conn['destino'])
                
                # Encontrar blocos iniciais
                todos_ids = {b['id'] for b in blocos_lista}
                destinos = {c['destino'] for c in conexoes_lista}
                blocos_iniciais = todos_ids - destinos
                
                # DFS para ordenar
                ordem = []
                visitados = set()
                
                def dfs(bid):
                    if bid in visitados:
                        return
                    visitados.add(bid)
                    ordem.append(bid)
                    if bid in grafo:
                        for prox in grafo[bid]:
                            dfs(prox)
                
                for inicial in blocos_iniciais:
                    dfs(inicial)
                
                # Multiplicar em série
                sistema_final = tfs[ordem[0]]
                for bid in ordem[1:]:
                    sistema_final = sistema_final * tfs[bid]
                
                msg = f"✅ Sistema com {len(ordem)} blocos conectados"
            else:
                # Sem conexões: multiplicar todos
                sistema_final = tfs[blocos_lista[0]['id']]
                for i in range(1, len(blocos_lista)):
                    sistema_final = sistema_final * tfs[blocos_lista[i]['id']]
                
                msg = f"✅ Sistema com {len(blocos_lista)} blocos (série simples)"
        
        return sistema_final, msg
        
    except Exception as e:
        return None, f"❌ Erro: {str(e)}"

# =====================================================
# INICIALIZAÇÃO
# =====================================================

def inicializar():
    """Inicializa session state"""
    if 'blocos_xcos' not in st.session_state:
        st.session_state.blocos_xcos = []
    if 'conexoes_xcos' not in st.session_state:
        st.session_state.conexoes_xcos = []
    if 'sistema_processado' not in st.session_state:
        st.session_state.sistema_processado = None

# =====================================================
# INTERFACE DO EDITOR XCOS
# =====================================================

def interface_editor_xcos():
    """Interface principal do editor Xcos"""
    
    st.title("🎨 Editor Visual Xcos - Sistemas de Controle")
    
    # Tabs
    tab1, tab2 = st.tabs(["🎨 Editor de Blocos", "📊 Análise do Sistema"])
    
    with tab1:
        st.markdown("### 🔧 Construa seu Diagrama de Blocos")
        
        # Painel de controle
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("➕ Adicionar G(s)", use_container_width=True):
                st.session_state.modo_adicao = 'transferencia'
        
        with col2:
            if st.button("📊 Adicionar Ganho K", use_container_width=True):
                st.session_state.modo_adicao = 'ganho'
        
        with col3:
            if st.button("∫ Adicionar Integrador", use_container_width=True):
                st.session_state.modo_adicao = 'integrador'
        
        with col4:
            if st.button("🗑️ Limpar Tudo", use_container_width=True):
                st.session_state.blocos_xcos = []
                st.session_state.conexoes_xcos = []
                st.session_state.sistema_processado = None
                st.rerun()
        
        # Formulário de adição baseado no modo
        if 'modo_adicao' in st.session_state:
            st.markdown("---")
            
            if st.session_state.modo_adicao == 'transferencia':
                st.markdown("#### ➕ Nova Função de Transferência G(s)")
                col_a, col_b, col_c = st.columns([2, 2, 1])
                
                with col_a:
                    num = st.text_input("Numerador", placeholder="Ex: 1, s, 2*s+1", key="num_input")
                with col_b:
                    den = st.text_input("Denominador", placeholder="Ex: s+1, s^2+2*s+1", key="den_input")
                with col_c:
                    st.markdown("##")
                    if st.button("✅ Adicionar", type="primary", use_container_width=True):
                        if num and den:
                            bloco_id = len(st.session_state.blocos_xcos) + 1
                            st.session_state.blocos_xcos.append({
                                'id': bloco_id,
                                'tipo': 'Transferência',
                                'config': {
                                    'nome': f'G{bloco_id}',
                                    'numerador': num,
                                    'denominador': den,
                                    'tf': f"{num}/{den}"
                                }
                            })
                            del st.session_state.modo_adicao
                            st.rerun()
            
            elif st.session_state.modo_adicao == 'ganho':
                st.markdown("#### 📊 Novo Ganho K")
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    ganho = st.number_input("Valor do Ganho", value=1.0, step=0.1, key="ganho_input")
                with col_b:
                    st.markdown("##")
                    if st.button("✅ Adicionar", type="primary", use_container_width=True):
                        bloco_id = len(st.session_state.blocos_xcos) + 1
                        st.session_state.blocos_xcos.append({
                            'id': bloco_id,
                            'tipo': 'Ganho',
                            'config': {
                                'nome': f'K{bloco_id}',
                                'valor': str(ganho),
                                'tf': str(ganho)
                            }
                        })
                        del st.session_state.modo_adicao
                        st.rerun()
            
            elif st.session_state.modo_adicao == 'integrador':
                st.markdown("#### ∫ Novo Integrador (1/s)")
                if st.button("✅ Adicionar Integrador", type="primary", use_container_width=True):
                    bloco_id = len(st.session_state.blocos_xcos) + 1
                    st.session_state.blocos_xcos.append({
                        'id': bloco_id,
                        'tipo': 'Integrador',
                        'config': {
                            'nome': f'Int{bloco_id}',
                            'tf': '1/s'
                        }
                    })
                    del st.session_state.modo_adicao
                    st.rerun()
        
        # Mostrar blocos existentes
        st.markdown("---")
        st.markdown("### 📦 Blocos no Diagrama")
        
        if st.session_state.blocos_xcos:
            for i, bloco in enumerate(st.session_state.blocos_xcos):
                col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
                
                with col1:
                    st.markdown(f"**#{bloco['id']}**")
                with col2:
                    st.markdown(f"**{bloco['tipo']}**")
                with col3:
                    tf_display = bloco['config'].get('tf', '-')
                    st.code(tf_display, language="text")
                with col4:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.blocos_xcos.pop(i)
                        st.rerun()
            
            # Configurar conexões
            st.markdown("---")
            st.markdown("### 🔗 Conexões")
            
            st.info("💡 Os blocos serão conectados em SÉRIE na ordem em que foram adicionados")
            
            # Mostrar conexões atuais
            if len(st.session_state.blocos_xcos) > 1:
                st.markdown("**Conexões automáticas (série):**")
                for i in range(len(st.session_state.blocos_xcos) - 1):
                    bloco1 = st.session_state.blocos_xcos[i]
                    bloco2 = st.session_state.blocos_xcos[i+1]
                    st.markdown(f"• Bloco #{bloco1['id']} ({bloco1['config']['nome']}) → Bloco #{bloco2['id']} ({bloco2['config']['nome']})")
        else:
            st.info("👈 Nenhum bloco adicionado ainda. Use os botões acima para começar!")
    
    with tab2:
        st.markdown("### 📊 Análise do Sistema")
        
        if st.button("⚡ PROCESSAR E ANALISAR SISTEMA", type="primary", use_container_width=True, key="processar"):
            if st.session_state.blocos_xcos:
                # Criar conexões automáticas em série
                conexoes = []
                for i in range(len(st.session_state.blocos_xcos) - 1):
                    conexoes.append({
                        'origem': st.session_state.blocos_xcos[i]['id'],
                        'destino': st.session_state.blocos_xcos[i+1]['id']
                    })
                
                st.session_state.conexoes_xcos = conexoes
                
                # Processar
                sistema, msg = processar_diagrama(st.session_state.blocos_xcos, 
                                                  st.session_state.conexoes_xcos)
                
                if sistema:
                    st.session_state.sistema_processado = sistema
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("⚠️ Adicione pelo menos um bloco antes de processar!")
        
        # Mostrar análises se o sistema foi processado
        if st.session_state.sistema_processado:
            sistema = st.session_state.sistema_processado
            
            st.markdown("---")
            st.markdown("#### 📐 Função de Transferência Resultante")
            st.code(f"G(s) = {sistema}", language="text")
            
            # Verificar estabilidade
            polos = ctrl.poles(sistema)
            estavel = all(np.real(p) < 0 for p in polos)
            
            if estavel:
                st.success("✅ **Sistema ESTÁVEL** (todos os polos no semiplano esquerdo)")
            else:
                st.error("❌ **Sistema INSTÁVEL** (polos no semiplano direito)")
            
            # Métricas de desempenho
            st.markdown("---")
            st.markdown("#### 📈 Métricas de Desempenho")
            
            desempenho = calcular_desempenho(sistema)
            
            cols = st.columns(3)
            items = list(desempenho.items())
            for i, (chave, valor) in enumerate(items):
                with cols[i % 3]:
                    st.metric(chave, valor)
            
            # Gráficos
            st.markdown("---")
            st.markdown("#### 📊 Visualizações")
            
            tab_resp, tab_bode, tab_pz, tab_lgr = st.tabs([
                "📈 Resposta Temporal",
                "📊 Bode",
                "🎯 Polos e Zeros",
                "🔄 LGR"
            ])
            
            with tab_resp:
                entrada = st.selectbox("Sinal de Entrada", 
                                      ['Degrau', 'Rampa', 'Senoidal', 'Impulso'])
                fig = plot_resposta_temporal(sistema, entrada)
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas
                if entrada == 'Degrau':
                    t = np.linspace(0, estimar_tempo_final(sistema), 1000)
                    _, y = step_response(sistema, t)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Valor Final", f"{y[-1]:.3f}")
                    with col2:
                        st.metric("Máximo", f"{np.max(y):.3f}")
                    with col3:
                        overshoot = ((np.max(y) - y[-1]) / y[-1] * 100) if y[-1] != 0 else 0
                        st.metric("Overshoot", f"{overshoot:.1f}%")
            
            with tab_bode:
                fig = plot_bode(sistema)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_pz:
                fig = plot_polos_zeros(sistema)
                st.plotly_chart(fig, use_container_width=True)
                
                # Informações sobre polos e zeros
                polos = ctrl.poles(sistema)
                zeros = ctrl.zeros(sistema)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Polos:**")
                    for i, p in enumerate(polos, 1):
                        st.text(f"p{i} = {p:.4f}")
                
                with col2:
                    st.markdown("**Zeros:**")
                    if len(zeros) > 0:
                        for i, z in enumerate(zeros, 1):
                            st.text(f"z{i} = {z:.4f}")
                    else:
                        st.text("Nenhum zero")
            
            with tab_lgr:
                fig = plot_lgr(sistema)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 O LGR mostra como os polos variam com o ganho K de 0 a ∞")

# =====================================================
# MAIN
# =====================================================

def main():
    inicializar()
    interface_editor_xcos()

if __name__ == "__main__":
    main()
