#!/usr/bin/env python3
"""
XCOS SIMULATOR - Streamlit Version
Interface web para design e análise de sistemas de controle
"""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid

# =====================================================================
# CLASSES E ESTRUTURAS DE DADOS
# =====================================================================

class BlockType(Enum):
    """Tipos de blocos disponíveis"""
    # Entrada
    STEP = "step"
    RAMP = "ramp"
    SINE = "sine"
    PULSE = "pulse"
    
    # Dinâmica
    TF = "tf"
    INTEGRATOR = "integrator"
    DERIVATIVE = "derivative"
    GAIN = "gain"
    STATE_SPACE = "state_space"
    DELAY = "delay"
    
    # Operações
    SUM = "sum"
    PRODUCT = "product"
    DIVIDE = "divide"
    
    # Saída
    SCOPE = "scope"
    PLOT = "plot"
    SINK = "sink"
    
    # Controladores
    PID = "pid"
    LEAD = "lead"
    LAG = "lag"


@dataclass
class Block:
    """Representa um bloco"""
    id: str
    type: str
    params: Dict
    label: str
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'params': self.params,
            'label': self.label
        }
    
    @staticmethod
    def from_dict(data):
        return Block(
            id=data['id'],
            type=data['type'],
            params=data['params'],
            label=data.get('label', '')
        )


@dataclass
class Connection:
    """Representa conexão entre blocos"""
    from_block_id: str
    to_block_id: str
    
    def to_dict(self):
        return {
            'from_block_id': self.from_block_id,
            'to_block_id': self.to_block_id
        }
    
    @staticmethod
    def from_dict(data):
        return Connection(
            from_block_id=data['from_block_id'],
            to_block_id=data['to_block_id']
        )


# =====================================================================
# CONFIGURAÇÃO STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="XCOS Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }
    .block-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        cursor: pointer;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s;
    }
    .block-box:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0066cc;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# INICIALIZAÇÃO DO ESTADO
# =====================================================================

def init_session():
    """Inicializa variáveis de sessão"""
    if 'blocks' not in st.session_state:
        st.session_state.blocks = {}
    if 'connections' not in st.session_state:
        st.session_state.connections = []
    if 'block_counter' not in st.session_state:
        st.session_state.block_counter = 0
    if 'selected_block' not in st.session_state:
        st.session_state.selected_block = None

init_session()

# =====================================================================
# DEFINIÇÕES DE BLOCOS
# =====================================================================

BLOCK_INFO = {
    'step': {'label': 'Step', 'category': 'Entrada', 'icon': '📊'},
    'ramp': {'label': 'Ramp', 'category': 'Entrada', 'icon': '📈'},
    'sine': {'label': 'Sine', 'category': 'Entrada', 'icon': '〰️'},
    'pulse': {'label': 'Pulse', 'category': 'Entrada', 'icon': '⬜'},
    'tf': {'label': 'TF', 'category': 'Dinâmica', 'icon': '⚙️'},
    'integrator': {'label': '∫', 'category': 'Dinâmica', 'icon': '∫'},
    'derivative': {'label': 'd/dt', 'category': 'Dinâmica', 'icon': 'd'},
    'gain': {'label': 'K', 'category': 'Dinâmica', 'icon': '✕'},
    'state_space': {'label': 'SS', 'category': 'Dinâmica', 'icon': '▦'},
    'delay': {'label': 'Delay', 'category': 'Dinâmica', 'icon': '⏱'},
    'sum': {'label': 'Σ', 'category': 'Operação', 'icon': '➕'},
    'product': {'label': '×', 'category': 'Operação', 'icon': '✕'},
    'divide': {'label': '÷', 'category': 'Operação', 'icon': '➗'},
    'scope': {'label': 'Scope', 'category': 'Saída', 'icon': '📺'},
    'plot': {'label': 'Plot', 'category': 'Saída', 'icon': '📊'},
    'sink': {'label': 'Sink', 'category': 'Saída', 'icon': '💧'},
    'pid': {'label': 'PID', 'category': 'Controlador', 'icon': '🎛'},
    'lead': {'label': 'Lead', 'category': 'Controlador', 'icon': '➡'},
    'lag': {'label': 'Lag', 'category': 'Controlador', 'icon': '⬅'},
}

DEFAULT_PARAMS = {
    'step': {'amplitude': 1.0, 'delay': 0},
    'ramp': {'slope': 1.0, 'start_time': 0},
    'sine': {'amplitude': 1.0, 'frequency': 1.0, 'phase': 0},
    'pulse': {'amplitude': 1.0, 'period': 2.0, 'duty': 0.5},
    'tf': {'numerator': [1], 'denominator': [1, 1]},
    'integrator': {'initial_value': 0},
    'derivative': {},
    'gain': {'K': 1.0},
    'state_space': {'A': [[1]], 'B': [[1]], 'C': [[1]], 'D': [[0]]},
    'delay': {'tau': 0.1},
    'sum': {'gains': [1, -1]},
    'product': {},
    'divide': {},
    'scope': {'buffer_size': 10000},
    'plot': {},
    'sink': {},
    'pid': {'Kp': 1.0, 'Ki': 0, 'Kd': 0},
    'lead': {'K': 1.0, 'z': 1.0, 'p': 2.0},
    'lag': {'K': 1.0, 'z': 0.5, 'p': 0.1},
}

# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

def add_block(block_type: str) -> str:
    """Adiciona um novo bloco"""
    st.session_state.block_counter += 1
    block_id = f"{block_type}_{st.session_state.block_counter}"
    
    info = BLOCK_INFO[block_type]
    block = Block(
        id=block_id,
        type=block_type,
        params=DEFAULT_PARAMS[block_type].copy(),
        label=info['label']
    )
    
    st.session_state.blocks[block_id] = block
    return block_id

def remove_block(block_id: str):
    """Remove um bloco"""
    if block_id in st.session_state.blocks:
        del st.session_state.blocks[block_id]
        # Remover conexões relacionadas
        st.session_state.connections = [
            c for c in st.session_state.connections
            if c.from_block_id != block_id and c.to_block_id != block_id
        ]

def add_connection(from_id: str, to_id: str):
    """Adiciona conexão entre blocos"""
    if from_id != to_id and from_id in st.session_state.blocks and to_id in st.session_state.blocks:
        # Remover conexões existentes para o destino
        st.session_state.connections = [
            c for c in st.session_state.connections
            if c.to_block_id != to_id
        ]
        st.session_state.connections.append(Connection(from_id, to_id))

def build_transfer_function() -> Optional[signal.TransferFunction]:
    """Constrói função de transferência a partir dos blocos"""
    # Encontrar blocos TF
    tf_blocks = [b for b in st.session_state.blocks.values() if b.type == 'tf']
    
    if not tf_blocks:
        return None
    
    # Usar primeiro TF
    tf_block = tf_blocks[0]
    num = tf_block.params.get('numerator', [1])
    den = tf_block.params.get('denominator', [1, 1])
    
    # Aplicar ganhos
    gain_blocks = [b for b in st.session_state.blocks.values() if b.type == 'gain']
    for gb in gain_blocks:
        K = gb.params.get('K', 1.0)
        num = [n * K for n in num]
    
    return signal.TransferFunction(num, den)

def simulate_step_response(sys: signal.TransferFunction, t_final: float = 10) -> Tuple:
    """Simula resposta ao degrau"""
    t = np.linspace(0, t_final, 1000)
    t, y = signal.step(sys, T=t)
    return t, y

def calculate_metrics(t, y) -> Dict:
    """Calcula métricas da resposta"""
    y_final = y[-1]
    y_max = np.max(y)
    y_min = np.min(y)
    
    # Overshoot
    if y_final != 0:
        overshoot = ((y_max - y_final) / abs(y_final)) * 100
    else:
        overshoot = 0
    
    # Tempo de acomodação (2%)
    tolerance = 0.02 * abs(y_final) if y_final != 0 else 0.02
    idx_settle = np.where(np.abs(y - y_final) <= tolerance)[0]
    settling_time = t[idx_settle[0]] if len(idx_settle) > 0 else None
    
    # Tempo de pico
    idx_peak = np.argmax(np.abs(y - y_final))
    peak_time = t[idx_peak]
    
    return {
        'steady_state': float(y_final),
        'peak': float(y_max),
        'overshoot_percent': float(overshoot),
        'peak_time': float(peak_time),
        'settling_time': float(settling_time) if settling_time else None,
    }

# =====================================================================
# INTERFACE PRINCIPAL
# =====================================================================

st.title("⚡ XCOS Simulator - Simulador de Sistemas de Controle")
st.markdown("Interface web para design e análise de sistemas de controle lineares")

# Sidebar - Blocos disponíveis
st.sidebar.header("📦 Blocos Disponíveis")

categories = {}
for block_type, info in BLOCK_INFO.items():
    cat = info['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append((block_type, info))

for category in sorted(categories.keys()):
    st.sidebar.subheader(f"🔹 {category}")
    cols = st.sidebar.columns(2)
    
    for idx, (block_type, info) in enumerate(categories[category]):
        with cols[idx % 2]:
            if st.button(f"{info['icon']} {info['label']}", key=f"btn_{block_type}", use_container_width=True):
                block_id = add_block(block_type)
                st.success(f"✓ {info['label']} adicionado!")
                st.rerun()

st.sidebar.divider()

# Botões principais
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("💾 Salvar", use_container_width=True):
        st.session_state.save_requested = True

with col2:
    if st.button("📂 Carregar", use_container_width=True):
        st.session_state.load_requested = True

with col3:
    if st.button("🗑️ Limpar", use_container_width=True):
        st.session_state.blocks.clear()
        st.session_state.connections.clear()
        st.rerun()

# =====================================================================
# ABAS PRINCIPAIS
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔧 Diagrama", "📊 Análise", "💾 Projeto", "ℹ️ Info"])

# =====================================================================
# ABA 1: DIAGRAMA
# =====================================================================

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📐 Diagrama em Blocos")
        
        if st.session_state.blocks:
            # Visualizar blocos
            st.write("**Blocos no Diagrama:**")
            
            for block_id, block in st.session_state.blocks.items():
                with st.expander(f"📦 {block.label} ({block_id})", expanded=False):
                    st.write(f"**Tipo:** {block.type}")
                    
                    # Editar parâmetros
                    st.write("**Parâmetros:**")
                    params = block.params.copy()
                    
                    for param_name, param_value in params.items():
                        if isinstance(param_value, (int, float)):
                            new_val = st.number_input(
                                param_name,
                                value=float(param_value),
                                step=0.1,
                                key=f"{block_id}_{param_name}"
                            )
                            block.params[param_name] = new_val
                        elif isinstance(param_value, list):
                            new_val_str = st.text_input(
                                f"{param_name} (separado por espaço)",
                                value=str(param_value),
                                key=f"{block_id}_{param_name}"
                            )
                            try:
                                block.params[param_name] = [float(x) for x in new_val_str.strip('[]').split(',')]
                            except:
                                st.error(f"Formato inválido para {param_name}")
                    
                    # Botões de ação
                    col_del, col_dup = st.columns(2)
                    with col_del:
                        if st.button("🗑️ Deletar", key=f"del_{block_id}", use_container_width=True):
                            remove_block(block_id)
                            st.rerun()
                    with col_dup:
                        if st.button("📋 Duplicar", key=f"dup_{block_id}", use_container_width=True):
                            new_id = add_block(block.type)
                            st.session_state.blocks[new_id].params = block.params.copy()
                            st.rerun()
            
            st.divider()
            
            # Conexões
            st.write("**Conexões:**")
            
            col_from, col_to, col_add = st.columns([2, 2, 1])
            
            with col_from:
                from_block = st.selectbox(
                    "De:",
                    options=list(st.session_state.blocks.keys()),
                    format_func=lambda x: f"{st.session_state.blocks[x].label} ({x})"
                )
            
            with col_to:
                to_blocks = [b for b in st.session_state.blocks.keys() if b != from_block]
                to_block = st.selectbox(
                    "Para:",
                    options=to_blocks,
                    format_func=lambda x: f"{st.session_state.blocks[x].label} ({x})"
                )
            
            with col_add:
                if st.button("➕ Conectar", use_container_width=True):
                    add_connection(from_block, to_block)
                    st.success("✓ Conexão criada!")
                    st.rerun()
            
            # Listar conexões
            if st.session_state.connections:
                st.write("**Conexões Existentes:**")
                for idx, conn in enumerate(st.session_state.connections):
                    from_label = st.session_state.blocks[conn.from_block_id].label
                    to_label = st.session_state.blocks[conn.to_block_id].label
                    
                    col_info, col_remove = st.columns([4, 1])
                    with col_info:
                        st.write(f"{idx+1}. {from_label} → {to_label}")
                    with col_remove:
                        if st.button("✕", key=f"remove_conn_{idx}", use_container_width=True):
                            st.session_state.connections.pop(idx)
                            st.rerun()
        else:
            st.info("👈 Adicione blocos usando os botões no painel esquerdo")
    
    with col_right:
        st.subheader("📋 Resumo")
        
        st.metric("Total de Blocos", len(st.session_state.blocks))
        st.metric("Conexões", len(st.session_state.connections))
        
        # Listar tipos
        if st.session_state.blocks:
            st.write("**Distribuição de Blocos:**")
            types = {}
            for block in st.session_state.blocks.values():
                types[block.type] = types.get(block.type, 0) + 1
            
            for btype, count in sorted(types.items()):
                info = BLOCK_INFO[btype]
                st.write(f"{info['icon']} {info['label']}: {count}")

# =====================================================================
# ABA 2: ANÁLISE
# =====================================================================

with tab2:
    if st.session_state.blocks:
        st.subheader("📊 Análise do Sistema")
        
        sys = build_transfer_function()
        
        if sys:
            # Parâmetros de simulação
            col1, col2 = st.columns(2)
            with col1:
                t_final = st.slider("Tempo final (s)", 1.0, 30.0, 10.0)
            with col2:
                t_points = st.slider("Número de pontos", 100, 5000, 1000)
            
            # Simular
            if st.button("▶️ Simular Resposta ao Degrau", use_container_width=True):
                try:
                    t = np.linspace(0, t_final, t_points)
                    t, y = signal.step(sys, T=t)
                    
                    # Calcular métricas
                    metrics = calculate_metrics(t, y)
                    
                    # Plotar
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Resposta ao Degrau:**")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=t, y=y,
                            mode='lines',
                            name='y(t)',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_hline(y=metrics['steady_state'], 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text="Valor final")
                        fig.update_layout(
                            title="Resposta ao Degrau Unitário",
                            xaxis_title="Tempo (s)",
                            yaxis_title="Amplitude",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Métricas:**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Valor Final", f"{metrics['steady_state']:.6f}")
                            st.metric("Pico", f"{metrics['peak']:.6f}")
                        with col_b:
                            st.metric("Overshoot", f"{metrics['overshoot_percent']:.2f}%")
                            st.metric("Tempo Pico", f"{metrics['peak_time']:.4f}s")
                        
                        if metrics['settling_time']:
                            st.metric("Tempo Acomodação", f"{metrics['settling_time']:.4f}s")
                    
                    # Polos e Zeros
                    st.write("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Polos e Zeros:**")
                        
                        poles = np.roots(sys.den)
                        zeros = np.roots(sys.num) if len(sys.num) > 0 else np.array([])
                        is_stable = all(p.real < 0 for p in poles)
                        
                        st.write(f"✓ **Estável:** {'SIM' if is_stable else 'NÃO'}")
                        st.write(f"**Polos:** {[f'{p:.4f}' for p in poles]}")
                        st.write(f"**Zeros:** {[f'{z:.4f}' for z in zeros]}")
                        
                        fig = go.Figure()
                        
                        # Polos
                        fig.add_trace(go.Scatter(
                            x=poles.real, y=poles.imag,
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='x'),
                            name='Polos'
                        ))
                        
                        # Zeros
                        if len(zeros) > 0:
                            fig.add_trace(go.Scatter(
                                x=zeros.real, y=zeros.imag,
                                mode='markers',
                                marker=dict(size=12, color='blue', symbol='circle'),
                                name='Zeros'
                            ))
                        
                        # Eixo de estabilidade
                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        fig.update_layout(
                            title="Mapa de Polos e Zeros",
                            xaxis_title="Parte Real",
                            yaxis_title="Parte Imaginária",
                            hovermode='closest',
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Diagrama de Bode:**")
                        
                        w = np.logspace(-2, 2, 500)
                        w_rad, mag, phase = signal.bode(sys, w)
                        
                        fig = go.Figure()
                        
                        # Magnitude
                        fig.add_trace(go.Scatter(
                            x=np.log10(w_rad), y=mag,
                            mode='lines',
                            name='Magnitude (dB)',
                            line=dict(color='blue'),
                            yaxis='y1'
                        ))
                        
                        # Fase
                        fig.add_trace(go.Scatter(
                            x=np.log10(w_rad), y=phase,
                            mode='lines',
                            name='Fase (°)',
                            line=dict(color='red'),
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title="Diagrama de Bode",
                            xaxis_title="log10(ω) [rad/s]",
                            yaxis=dict(title="Magnitude (dB)", side='left'),
                            yaxis2=dict(title="Fase (°)", overlaying='y', side='right'),
                            hovermode='x unified',
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Erro na simulação: {str(e)}")
        else:
            st.warning("⚠️ Nenhum bloco Transfer Function encontrado. Adicione um bloco TF para simular.")
    else:
        st.info("👈 Adicione blocos para realizar análise")

# =====================================================================
# ABA 3: PROJETO
# =====================================================================

with tab3:
    col_export, col_import = st.columns(2)
    
    with col_export:
        st.subheader("💾 Exportar Projeto")
        
        if st.session_state.blocks or st.session_state.connections:
            project_data = {
                'blocks': [b.to_dict() for b in st.session_state.blocks.values()],
                'connections': [c.to_dict() for c in st.session_state.connections]
            }
            
            json_str = json.dumps(project_data, indent=2)
            
            st.download_button(
                label="📥 Baixar Projeto (JSON)",
                data=json_str,
                file_name="xcos_diagram.json",
                mime="application/json"
            )
            
            st.text_area("Visualizar JSON:", value=json_str, height=300)
        else:
            st.info("Nenhum diagrama para exportar")
    
    with col_import:
        st.subheader("📤 Importar Projeto")
        
        uploaded_file = st.file_uploader("Carregar arquivo JSON", type=['json'])
        
        if uploaded_file:
            try:
                project_data = json.load(uploaded_file)
                
                # Limpar estado atual
                st.session_state.blocks.clear()
                st.session_state.connections.clear()
                
                # Carregar blocos
                for block_data in project_data.get('blocks', []):
                    block = Block.from_dict(block_data)
                    st.session_state.blocks[block.id] = block
                
                # Carregar conexões
                for conn_data in project_data.get('connections', []):
                    conn = Connection.from_dict(conn_data)
                    st.session_state.connections.append(conn)
                
                st.success("✓ Projeto carregado com sucesso!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

# =====================================================================
# ABA 4: INFORMAÇÕES
# =====================================================================

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Sobre XCOS Simulator")
        
        st.markdown("""
        **XCOS Simulator** é uma ferramenta web para design e análise de sistemas de controle.
        
        ### Características:
        - 🔧 Interface visual intuitiva
        - 📦 20+ tipos de blocos pré-configurados
        - 📊 Análise completa (Bode, Nyquist, Polos/Zeros)
        - 💾 Salvar/carregar projetos em JSON
        - ⚡ Simulação rápida
        
        ### Tipos de Blocos:
        - **Entradas:** Step, Ramp, Sine, Pulse
        - **Dinâmica:** TF, Integrador, Derivador, Ganho, State Space, Delay
        - **Operações:** Soma, Multiplicação, Divisão
        - **Saída:** Scope, Plot, Sink
        - **Controladores:** PID, Lead, Lag
        """)
    
    with col2:
        st.subheader("🚀 Guia Rápido")
        
        st.markdown("""
        ### Como usar:
        
        1️⃣ **Adicionar Blocos**
           - Clique nos botões no painel esquerdo
           - Blocos aparecem na aba Diagrama
        
        2️⃣ **Configurar Parâmetros**
           - Expanda cada bloco
           - Edite os parâmetros
        
        3️⃣ **Conectar Blocos**
           - Use os seletores "De" e "Para"
           - Clique "➕ Conectar"
        
        4️⃣ **Simular**
           - Vá para "Análise"
           - Clique "▶️ Simular"
           - Visualize resultados
        
        5️⃣ **Salvar Projeto**
           - Vá para "Projeto"
           - Clique "📥 Baixar"
        """)
    
    st.divider()
    
    st.subheader("⚙️ Referência de Blocos")
    
    for category in sorted(categories.keys()):
        with st.expander(f"🔹 {category}", expanded=False):
            for block_type, info in categories[category]:
                with st.expander(f"{info['icon']} {info['label']}", expanded=False):
                    default_params = DEFAULT_PARAMS[block_type]
                    
                    st.write(f"**Tipo:** `{block_type}`")
                    st.write("**Parâmetros padrão:**")
                    
                    for param_name, param_value in default_params.items():
                        st.write(f"- `{param_name}`: {param_value}")

# =====================================================================
# FOOTER
# =====================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>⚡ XCOS Simulator v1.0 | Desenvolvido em Python com Streamlit | 2025</p>
    <p>Para sistemas de controle lineares | Educação & Engenharia</p>
</div>
""", unsafe_allow_html=True)
