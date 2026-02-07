import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection
import control as ct
from scipy import signal
import json

# Configuração da página
st.set_page_config(page_title="Sistema de Controle Interativo", layout="wide", initial_sidebar_state="expanded")

# CSS customizado para melhorar a interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .block-info {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'blocks' not in st.session_state:
    st.session_state.blocks = []
if 'connections' not in st.session_state:
    st.session_state.connections = []
if 'block_counter' not in st.session_state:
    st.session_state.block_counter = 0
if 'selected_block' not in st.session_state:
    st.session_state.selected_block = None

# Funções auxiliares
def add_block(block_type, x, y, label="", tf_num=[1], tf_den=[1, 1]):
    """Adiciona um bloco ao diagrama"""
    block = {
        'id': st.session_state.block_counter,
        'type': block_type,
        'x': x,
        'y': y,
        'label': label if label else f"{block_type}_{st.session_state.block_counter}",
        'tf_num': tf_num,
        'tf_den': tf_den,
        'width': 1.5 if block_type == 'transfer' else 0.8,
        'height': 1.0 if block_type == 'transfer' else 0.8
    }
    st.session_state.blocks.append(block)
    st.session_state.block_counter += 1
    return block

def add_connection(from_block_id, to_block_id):
    """Adiciona uma conexão entre blocos"""
    connection = {
        'from': from_block_id,
        'to': to_block_id
    }
    if connection not in st.session_state.connections:
        st.session_state.connections.append(connection)

def delete_block(block_id):
    """Remove um bloco e suas conexões"""
    st.session_state.blocks = [b for b in st.session_state.blocks if b['id'] != block_id]
    st.session_state.connections = [c for c in st.session_state.connections 
                                   if c['from'] != block_id and c['to'] != block_id]

def draw_block_diagram():
    """Desenha o diagrama de blocos"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F5F5F5')
    
    # Desenhar conexões primeiro
    for conn in st.session_state.connections:
        from_block = next((b for b in st.session_state.blocks if b['id'] == conn['from']), None)
        to_block = next((b for b in st.session_state.blocks if b['id'] == conn['to']), None)
        
        if from_block and to_block:
            x1 = from_block['x'] + from_block['width']/2
            y1 = from_block['y']
            x2 = to_block['x'] - to_block['width']/2
            y2 = to_block['y']
            
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2, color='#1976D2', zorder=1)
            ax.add_patch(arrow)
    
    # Desenhar blocos
    for block in st.session_state.blocks:
        x, y = block['x'], block['y']
        
        if block['type'] == 'transfer':
            # Bloco de transferência (retângulo)
            rect = FancyBboxPatch((x - block['width']/2, y - block['height']/2),
                                 block['width'], block['height'],
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#0D47A1', facecolor='#BBDEFB',
                                 linewidth=2.5, zorder=2)
            ax.add_patch(rect)
            
            # Texto da função de transferência
            num_str = 'num: ' + str(block['tf_num'])
            den_str = 'den: ' + str(block['tf_den'])
            ax.text(x, y + 0.15, block['label'], ha='center', va='center',
                   fontsize=10, fontweight='bold', zorder=3)
            ax.text(x, y - 0.15, f"{num_str}\n{den_str}", ha='center', va='center',
                   fontsize=7, zorder=3)
            
        elif block['type'] == 'sum':
            # Somador (círculo)
            circle = Circle((x, y), 0.4, edgecolor='#1B5E20', 
                          facecolor='#C8E6C9', linewidth=2.5, zorder=2)
            ax.add_patch(circle)
            
            # Sinais + e -
            ax.plot([x-0.4, x+0.4], [y, y], 'k-', linewidth=2, zorder=3)
            ax.plot([x, x], [y-0.4, y+0.4], 'k-', linewidth=2, zorder=3)
            ax.text(x, y - 0.65, block['label'], ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=3)
            
        elif block['type'] == 'junction':
            # Junção (ponto)
            circle = Circle((x, y), 0.15, edgecolor='#E65100', 
                          facecolor='#FF6F00', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y - 0.5, block['label'], ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=3)
            
        elif block['type'] == 'gain':
            # Ganho (triângulo)
            triangle = plt.Polygon([(x-0.4, y-0.4), (x-0.4, y+0.4), (x+0.4, y)],
                                 edgecolor='#4A148C', facecolor='#E1BEE7',
                                 linewidth=2.5, zorder=2)
            ax.add_patch(triangle)
            ax.text(x, y, block['label'], ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=3)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title('Diagrama de Blocos do Sistema de Controle', 
                fontsize=14, fontweight='bold', pad=20)
    
    return fig

def calculate_system_transfer_function():
    """Calcula a função de transferência total do sistema"""
    try:
        if len(st.session_state.blocks) == 0:
            return None
        
        # Exemplo simplificado - em um sistema real, você precisaria de um algoritmo
        # mais sofisticado para calcular a FT baseado nas conexões
        
        # Para demonstração, vamos assumir blocos em série
        transfer_blocks = [b for b in st.session_state.blocks if b['type'] == 'transfer']
        
        if not transfer_blocks:
            return None
        
        # Multiplicar funções de transferência em série
        num_total = transfer_blocks[0]['tf_num']
        den_total = transfer_blocks[0]['tf_den']
        
        for block in transfer_blocks[1:]:
            num_total = np.convolve(num_total, block['tf_num'])
            den_total = np.convolve(den_total, block['tf_den'])
        
        sys = ct.TransferFunction(num_total, den_total)
        return sys
    except Exception as e:
        st.error(f"Erro ao calcular função de transferência: {e}")
        return None

def plot_bode():
    """Plota o diagrama de Bode"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Calcular resposta em frequência
    omega = np.logspace(-2, 3, 1000)
    mag, phase, omega = ct.bode(sys, omega, plot=False)
    
    # Magnitude
    ax1.semilogx(omega, 20 * np.log10(mag), 'b-', linewidth=2)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Diagrama de Bode', fontsize=13, fontweight='bold')
    
    # Fase
    ax2.semilogx(omega, phase * 180/np.pi, 'r-', linewidth=2)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlabel('Frequência (rad/s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Fase (graus)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_step_response():
    """Plota a resposta ao degrau"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t, y = ct.step_response(sys)
    
    ax.plot(t, y, 'b-', linewidth=2, label='Resposta ao Degrau')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Tempo (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title('Resposta ao Degrau Unitário', fontsize=13, fontweight='bold')
    ax.legend()
    
    # Calcular e mostrar características
    overshoot = (np.max(y) - y[-1]) / y[-1] * 100 if y[-1] != 0 else 0
    settling_time = t[-1]
    
    ax.text(0.02, 0.98, f'Overshoot: {overshoot:.2f}%\nTempo de acomodação: {settling_time:.2f}s',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_impulse_response():
    """Plota a resposta ao impulso"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t, y = ct.impulse_response(sys)
    
    ax.plot(t, y, 'g-', linewidth=2, label='Resposta ao Impulso')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Tempo (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title('Resposta ao Impulso', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_root_locus():
    """Plota o lugar das raízes"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ct.root_locus(sys, ax=ax)
    ax.set_title('Lugar das Raízes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_nyquist():
    """Plota o diagrama de Nyquist"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ct.nyquist_plot(sys, ax=ax)
    ax.set_title('Diagrama de Nyquist', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_pole_zero():
    """Plota polos e zeros"""
    sys = calculate_system_transfer_function()
    if sys is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    poles = ct.poles(sys)
    zeros = ct.zeros(sys)
    
    # Plotar zeros
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'go', markersize=12, 
               markerfacecolor='white', markeredgewidth=2, label='Zeros')
    
    # Plotar polos
    if len(poles) > 0:
        ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, 
               markeredgewidth=2, label='Polos')
    
    # Círculo unitário
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.3, label='Círculo Unitário')
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Parte Real', fontsize=11, fontweight='bold')
    ax.set_ylabel('Parte Imaginária', fontsize=11, fontweight='bold')
    ax.set_title('Mapa de Polos e Zeros', fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

# Interface principal
st.markdown('<p class="main-header">🎛️ Sistema de Controle Interativo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Crie, conecte e analise diagramas de blocos de sistemas de controle</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Ferramentas")

st.sidebar.subheader("➕ Adicionar Blocos")

col1, col2 = st.sidebar.columns(2)

with col1:
    block_x = st.number_input("Posição X", min_value=0.0, max_value=14.0, value=2.0, step=0.5)
with col2:
    block_y = st.number_input("Posição Y", min_value=0.0, max_value=9.0, value=5.0, step=0.5)

block_type = st.sidebar.selectbox(
    "Tipo de Bloco",
    ["transfer", "sum", "junction", "gain"],
    format_func=lambda x: {
        "transfer": "📦 Função de Transferência",
        "sum": "➕ Somador",
        "junction": "⚫ Junção",
        "gain": "📐 Ganho"
    }[x]
)

block_label = st.sidebar.text_input("Nome do Bloco", value=f"{block_type}_{st.session_state.block_counter}")

if block_type == "transfer":
    st.sidebar.markdown("**Função de Transferência:**")
    
    num_input = st.sidebar.text_input("Numerador (separado por vírgula)", value="1")
    den_input = st.sidebar.text_input("Denominador (separado por vírgula)", value="1,1")
    
    try:
        tf_num = [float(x.strip()) for x in num_input.split(',')]
        tf_den = [float(x.strip()) for x in den_input.split(',')]
    except:
        tf_num = [1]
        tf_den = [1, 1]
        st.sidebar.error("Formato inválido! Use números separados por vírgula.")
else:
    tf_num = [1]
    tf_den = [1]

if st.sidebar.button("➕ Adicionar Bloco", use_container_width=True):
    add_block(block_type, block_x, block_y, block_label, tf_num, tf_den)
    st.rerun()

st.sidebar.markdown("---")

# Conexões
st.sidebar.subheader("🔗 Conectar Blocos")

if len(st.session_state.blocks) >= 2:
    block_options = {f"{b['label']} (ID: {b['id']})": b['id'] for b in st.session_state.blocks}
    
    from_block_name = st.sidebar.selectbox("Bloco de Origem", list(block_options.keys()), key="from")
    to_block_name = st.sidebar.selectbox("Bloco de Destino", list(block_options.keys()), key="to")
    
    if st.sidebar.button("🔗 Conectar", use_container_width=True):
        add_connection(block_options[from_block_name], block_options[to_block_name])
        st.rerun()
else:
    st.sidebar.info("Adicione pelo menos 2 blocos para criar conexões")

st.sidebar.markdown("---")

# Gerenciamento de blocos
st.sidebar.subheader("🗑️ Gerenciar Blocos")

if st.session_state.blocks:
    block_to_delete = st.sidebar.selectbox(
        "Selecionar bloco para deletar",
        [f"{b['label']} (ID: {b['id']})" for b in st.session_state.blocks]
    )
    
    if st.sidebar.button("🗑️ Deletar Bloco", use_container_width=True):
        block_id = int(block_to_delete.split("ID: ")[1].rstrip(")"))
        delete_block(block_id)
        st.rerun()

if st.sidebar.button("🔄 Limpar Tudo", use_container_width=True):
    st.session_state.blocks = []
    st.session_state.connections = []
    st.session_state.block_counter = 0
    st.rerun()

st.sidebar.markdown("---")

# Exemplos pré-configurados
st.sidebar.subheader("📋 Exemplos")

if st.sidebar.button("🎯 Sistema Malha Fechada", use_container_width=True):
    st.session_state.blocks = []
    st.session_state.connections = []
    st.session_state.block_counter = 0
    
    # Criar sistema de malha fechada básico
    add_block('sum', 2, 5, 'Erro', [1], [1])
    add_block('transfer', 5, 5, 'Controlador', [10], [1])
    add_block('transfer', 9, 5, 'Planta', [1], [1, 2, 1])
    add_block('junction', 11, 5, 'Saída', [1], [1])
    add_block('transfer', 9, 3, 'Sensor', [1], [0.1, 1])
    
    add_connection(0, 1)
    add_connection(1, 2)
    add_connection(2, 3)
    add_connection(3, 4)
    add_connection(4, 0)
    
    st.rerun()

if st.sidebar.button("⚙️ Sistema em Cascata", use_container_width=True):
    st.session_state.blocks = []
    st.session_state.connections = []
    st.session_state.block_counter = 0
    
    add_block('transfer', 2, 5, 'G1', [1], [1, 1])
    add_block('transfer', 5, 5, 'G2', [2], [1, 0.5])
    add_block('transfer', 8, 5, 'G3', [5], [1, 2, 1])
    add_block('junction', 10.5, 5, 'Saída', [1], [1])
    
    add_connection(0, 1)
    add_connection(1, 2)
    add_connection(2, 3)
    
    st.rerun()

# Área principal com abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Diagrama", "📈 Análise de Frequência", "⏱️ Análise Temporal", "🎯 Estabilidade"])

with tab1:
    st.subheader("Diagrama de Blocos")
    
    if st.session_state.blocks:
        fig_diagram = draw_block_diagram()
        st.pyplot(fig_diagram)
        plt.close()
        
        # Informações dos blocos
        with st.expander("ℹ️ Informações dos Blocos"):
            for block in st.session_state.blocks:
                st.markdown(f"""
                <div class="block-info">
                <b>{block['label']}</b> (ID: {block['id']}) - Tipo: {block['type']}<br>
                Posição: ({block['x']:.1f}, {block['y']:.1f})<br>
                {f"FT: {block['tf_num']} / {block['tf_den']}" if block['type'] == 'transfer' else ""}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Adicione blocos usando a barra lateral para começar!")

with tab2:
    st.subheader("Análise de Frequência")
    
    if st.session_state.blocks:
        analysis_type = st.radio(
            "Selecione o tipo de análise:",
            ["Diagrama de Bode", "Diagrama de Nyquist"],
            horizontal=True
        )
        
        if analysis_type == "Diagrama de Bode":
            with st.spinner("Calculando diagrama de Bode..."):
                fig_bode = plot_bode()
                if fig_bode:
                    st.pyplot(fig_bode)
                    plt.close()
                else:
                    st.warning("Adicione blocos de transferência para análise")
        
        else:
            with st.spinner("Calculando diagrama de Nyquist..."):
                fig_nyquist = plot_nyquist()
                if fig_nyquist:
                    st.pyplot(fig_nyquist)
                    plt.close()
                else:
                    st.warning("Adicione blocos de transferência para análise")
    else:
        st.info("Adicione blocos ao diagrama para realizar análises de frequência")

with tab3:
    st.subheader("Análise Temporal")
    
    if st.session_state.blocks:
        response_type = st.radio(
            "Selecione o tipo de resposta:",
            ["Resposta ao Degrau", "Resposta ao Impulso"],
            horizontal=True
        )
        
        if response_type == "Resposta ao Degrau":
            with st.spinner("Calculando resposta ao degrau..."):
                fig_step = plot_step_response()
                if fig_step:
                    st.pyplot(fig_step)
                    plt.close()
                else:
                    st.warning("Adicione blocos de transferência para análise")
        
        else:
            with st.spinner("Calculando resposta ao impulso..."):
                fig_impulse = plot_impulse_response()
                if fig_impulse:
                    st.pyplot(fig_impulse)
                    plt.close()
                else:
                    st.warning("Adicione blocos de transferência para análise")
    else:
        st.info("Adicione blocos ao diagrama para realizar análises temporais")

with tab4:
    st.subheader("Análise de Estabilidade")
    
    if st.session_state.blocks:
        stability_type = st.radio(
            "Selecione o tipo de análise:",
            ["Lugar das Raízes", "Mapa de Polos e Zeros"],
            horizontal=True
        )
        
        if stability_type == "Lugar das Raízes":
            with st.spinner("Calculando lugar das raízes..."):
                fig_rlocus = plot_root_locus()
                if fig_rlocus:
                    st.pyplot(fig_rlocus)
                    plt.close()
                else:
                    st.warning("Adicione blocos de transferência para análise")
        
        else:
            with st.spinner("Calculando polos e zeros..."):
                fig_pz = plot_pole_zero()
                if fig_pz:
                    st.pyplot(fig_pz)
                    plt.close()
                    
                    # Análise de estabilidade
                    sys = calculate_system_transfer_function()
                    if sys:
                        poles = ct.poles(sys)
                        is_stable = all(np.real(p) < 0 for p in poles)
                        
                        if is_stable:
                            st.success("✅ Sistema ESTÁVEL: Todos os polos têm parte real negativa")
                        else:
                            st.error("❌ Sistema INSTÁVEL: Existem polos com parte real positiva ou nula")
                        
                        st.write("**Polos do sistema:**")
                        for i, pole in enumerate(poles):
                            st.write(f"Polo {i+1}: {pole:.4f}")
                else:
                    st.warning("Adicione blocos de transferência para análise")
    else:
        st.info("Adicione blocos ao diagrama para realizar análises de estabilidade")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Sistema de Controle Interativo</b> - Desenvolvido com Streamlit</p>
    <p>📚 Funcionalidades: Diagrama de Blocos, Bode, Nyquist, Resposta Temporal, Lugar das Raízes, Polos e Zeros</p>
</div>
""", unsafe_allow_html=True)
