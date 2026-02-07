import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.integrate import odeint
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import pandas as pd

# ==================== CONFIGURAÇÃO ====================

st.set_page_config(
    page_title="XCOS Simulador Profissional",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Personalizado
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 2px solid #334155;
    }
    
    .block-item {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 4px solid;
        background-color: rgba(51, 65, 85, 0.5);
        cursor: move;
        transition: all 0.3s ease;
    }
    
    .block-item:hover {
        background-color: rgba(51, 65, 85, 0.8);
        transform: translateX(4px);
    }
    
    .canvas-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 20px;
        min-height: 600px;
    }
    
    .block-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #475569;
        border-radius: 8px;
        padding: 12px;
        margin: 8px;
        cursor: grab;
        transition: all 0.3s ease;
    }
    
    .block-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.3);
    }
    
    .block-card.selected {
        border-color: #fbbf24;
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CLASSES E ESTRUTURAS ====================

@dataclass
class BlockDefinition:
    """Definição de um bloco"""
    id: int
    type: str  # controller_p, controller_pid, plant_1st, plant_2nd, feedback, input, output, sum
    name: str
    x: float
    y: float
    width: float = 120
    height: float = 80
    params: dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {
                'kp': 1.0,
                'ki': 0.5,
                'kd': 0.5,
                'tau': 1.0,
                'gain': 1.0,
                'wn': 2.0,
                'zeta': 0.7,
            }

@dataclass
class Connection:
    """Conexão entre blocos"""
    id: int
    from_block_id: int
    to_block_id: int
    from_port: str = "output"
    to_port: str = "input"

class TransferFunction:
    """Classe para representar funções de transferência"""
    
    def __init__(self, num, den, name=""):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)
        self.name = name
        try:
            self.tf = signal.TransferFunction(self.num, self.den)
            self.polos = self.tf.poles
            self.zeros = self.tf.zeros
        except:
            self.tf = None
            self.polos = np.array([])
            self.zeros = np.array([])
    
    def __mul__(self, other):
        """Multiplicação em série"""
        num = np.convolve(self.num, other.num)
        den = np.convolve(self.den, other.den)
        return TransferFunction(num, den, f"({self.name})×({other.name})")
    
    def ganho_dc(self):
        """Ganho em DC"""
        if len(self.den) == 0 or self.den[0] == 0:
            return 0
        return self.num[0] / self.den[0]
    
    def resposta_degrau(self, t):
        """Resposta ao degrau"""
        try:
            t_sim, y = signal.step(self.tf, T=t)
            return t_sim, y
        except:
            return t, np.zeros_like(t)
    
    def diagrama_bode(self, w=None):
        """Diagrama de Bode"""
        if w is None:
            w = np.logspace(-2, 2, 500)
        try:
            w_real, mag, fase = signal.bode(self.tf, w)
            return w_real, mag, fase
        except:
            return [], [], []
    
    def is_stable(self):
        """Verifica estabilidade"""
        if len(self.polos) == 0:
            return True
        return np.all(np.real(self.polos) < 0)

# ==================== FUNÇÕES AUXILIARES ====================

def criar_controlador(tipo: str, params: dict) -> TransferFunction:
    """Cria controlador baseado no tipo"""
    
    if tipo == 'controller_p':
        kp = params.get('kp', 1.0)
        return TransferFunction([kp], [1], f"P({kp})")
    
    elif tipo == 'controller_pid':
        kp = params.get('kp', 1.0)
        ki = params.get('ki', 0.5)
        kd = params.get('kd', 0.5)
        num = np.convolve([kd, kp, ki], [1])
        return TransferFunction(num, [1, 0], "PID")
    
    else:
        return TransferFunction([1], [1], "Unity")

def criar_planta(tipo: str, params: dict) -> TransferFunction:
    """Cria planta baseado no tipo"""
    
    if tipo == 'plant_1st':
        k = params.get('gain', 1.0)
        tau = params.get('tau', 1.0)
        return TransferFunction([k], [tau, 1], f"G₁({k}/({tau}s+1))")
    
    elif tipo == 'plant_2nd':
        k = params.get('gain', 1.0)
        wn = params.get('wn', 2.0)
        zeta = params.get('zeta', 0.7)
        num = [k * wn**2]
        den = [1, 2*zeta*wn, wn**2]
        return TransferFunction(num, den, "G₂")
    
    else:
        return TransferFunction([1], [1], "Unity")

def construir_sistema(blocos: List[BlockDefinition], conexoes: List[Connection]) -> Optional[TransferFunction]:
    """Constrói função de transferência a partir dos blocos"""
    
    if not blocos:
        return None
    
    # Mapear blocos para TFs
    tfs_blocos = {}
    
    for bloco in blocos:
        if bloco.type.startswith('controller'):
            tfs_blocos[bloco.id] = criar_controlador(bloco.type, bloco.params)
        elif bloco.type.startswith('plant'):
            tfs_blocos[bloco.id] = criar_planta(bloco.type, bloco.params)
        elif bloco.type == 'feedback':
            k = bloco.params.get('gain', 1.0)
            tfs_blocos[bloco.id] = TransferFunction([k], [1], "B(s)")
    
    # Encontrar controladores e plantas
    controllers = [b for b in blocos if b.type.startswith('controller')]
    plants = [b for b in blocos if b.type.startswith('plant')]
    feedbacks = [b for b in blocos if b.type == 'feedback']
    
    if not controllers or not plants:
        return None
    
    # Construir C*G
    C = tfs_blocos.get(controllers[0].id)
    G = tfs_blocos.get(plants[0].id)
    
    if not C or not G:
        return None
    
    CG = C * G
    
    # Se há realimentação, calcular malha fechada
    if feedbacks:
        B = tfs_blocos.get(feedbacks[0].id)
        if B:
            BCG = B * CG
            # H(s) = C(s)*G(s) / [1 + B(s)*C(s)*G(s)]
            num_mf = CG.num
            try:
                den_mf = np.polyadd(np.array([1]), BCG.num)
            except:
                den_mf = np.array([1]) + BCG.num
            return TransferFunction(num_mf, den_mf, "H_MF")
    
    return CG

# ==================== INICIALIZAR SESSION STATE ====================

if 'blocos' not in st.session_state:
    st.session_state.blocos = []

if 'conexoes' not in st.session_state:
    st.session_state.conexoes = []

if 'bloco_selecionado' not in st.session_state:
    st.session_state.bloco_selecionado = None

if 'resultado_simulacao' not in st.session_state:
    st.session_state.resultado_simulacao = None

if 'id_contador' not in st.session_state:
    st.session_state.id_contador = 1000

# ==================== BIBLIOTECA DE BLOCOS ====================

blocos_disponiveis = [
    {'type': 'input', 'nome': '📥 Entrada R(s)', 'cor': '#3B82F6', 'descricao': 'Sinal de referência'},
    {'type': 'sum', 'nome': '➕ Somador', 'cor': '#8B5CF6', 'descricao': 'Somador/Subtrator'},
    {'type': 'controller_p', 'nome': '🎛️ P (Proporcional)', 'cor': '#10B981', 'descricao': 'Controlador P'},
    {'type': 'controller_pid', 'nome': '🎛️ PID', 'cor': '#10B981', 'descricao': 'Controlador PID'},
    {'type': 'plant_1st', 'nome': '🔧 Planta 1ª Ordem', 'cor': '#F59E0B', 'descricao': 'Sistema 1ª ordem'},
    {'type': 'plant_2nd', 'nome': '🔧 Planta 2ª Ordem', 'cor': '#F59E0B', 'descricao': 'Sistema 2ª ordem'},
    {'type': 'feedback', 'nome': '🔄 Realimentação B(s)', 'cor': '#EF4444', 'descricao': 'Sensor'},
    {'type': 'output', 'nome': '📤 Saída Y(s)', 'cor': '#06B6D4', 'descricao': 'Sinal de saída'},
    {'type': 'display', 'nome': '📊 Display', 'cor': '#64748B', 'descricao': 'Visualização'},
]

# ==================== LAYOUT PRINCIPAL ====================

# Header
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #60a5fa; margin-bottom: 10px;'>⚙️ XCOS Simulador Profissional</h1>
    <p style='color: #94a3b8; font-size: 14px;'>Simulação de Sistemas de Controle Clássico em Malha Aberta e Fechada</p>
</div>
""", unsafe_allow_html=True)

# Layout de colunas
col_library, col_canvas = st.columns([1, 3], gap="medium")

# ==================== PAINEL LATERAL - BIBLIOTECA ====================

with col_library:
    st.markdown("### 📦 Biblioteca de Blocos")
    st.markdown("---")
    
    # Botão para adicionar blocos
    for bloco_tipo in blocos_disponiveis:
        col_btn = st.columns([1], gap="small")[0]
        
        if col_btn.button(
            f"{bloco_tipo['nome']}",
            key=f"add_{bloco_tipo['type']}_{st.session_state.id_contador}",
            use_container_width=True,
            help=bloco_tipo['descricao']
        ):
            novo_bloco = BlockDefinition(
                id=st.session_state.id_contador,
                type=bloco_tipo['type'],
                name=bloco_tipo['nome'],
                x=np.random.uniform(100, 600),
                y=np.random.uniform(100, 400),
            )
            st.session_state.blocos.append(novo_bloco)
            st.session_state.id_contador += 1
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🎮 Controles")
    
    col1, col2 = st.columns(2)
    
    if col1.button("▶️ Simular", use_container_width=True, help="Executar simulação"):
        if st.session_state.blocos:
            # Construir sistema
            H = construir_sistema(st.session_state.blocos, st.session_state.conexoes)
            
            if H:
                st.session_state.resultado_simulacao = {
                    'sistema': H,
                    'estavel': H.is_stable(),
                    'polos': H.polos,
                    'zeros': H.zeros,
                    'ganho_dc': H.ganho_dc(),
                }
                st.success("✅ Simulação executada com sucesso!")
                st.rerun()
            else:
                st.error("❌ Erro ao construir o sistema. Verifique as conexões.")
        else:
            st.warning("⚠️ Adicione blocos primeiro!")
    
    if col2.button("🗑️ Limpar", use_container_width=True, help="Remover tudo"):
        st.session_state.blocos = []
        st.session_state.conexoes = []
        st.session_state.bloco_selecionado = None
        st.session_state.resultado_simulacao = None
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Informações do Sistema")
    
    info_col = st.container()
    
    with info_col:
        st.metric("Blocos", len(st.session_state.blocos))
        st.metric("Conexões", len(st.session_state.conexoes))
        
        if st.session_state.resultado_simulacao:
            resultado = st.session_state.resultado_simulacao
            
            if resultado['estavel']:
                st.success("✅ Sistema Estável")
            else:
                st.error("❌ Sistema Instável")
            
            st.write(f"**Ganho DC:** {resultado['ganho_dc']:.4f}")
            
            if len(resultado['polos']) > 0:
                st.write(f"**Polos:** {len(resultado['polos'])}")
                for i, polo in enumerate(resultado['polos']):
                    st.write(f"  p{i+1} = {polo:.4f}")

# ==================== CANVAS PRINCIPAL ====================

with col_canvas:
    st.markdown("### 🎨 Canvas de Diagrama")
    
    # Container para o diagrama
    canvas_container = st.container()
    
    with canvas_container:
        if len(st.session_state.blocos) == 0:
            st.info("👈 Adicione blocos da biblioteca para começar!")
        else:
            # Criar visualização dos blocos
            col_blocos = st.columns(min(3, len(st.session_state.blocos)))
            
            for idx, bloco in enumerate(st.session_state.blocos):
                with col_blocos[idx % 3]:
                    # Encontrar cor do bloco
                    cor_bloco = next((b['cor'] for b in blocos_disponiveis if b['type'] == bloco.type), '#6B7280')
                    
                    # Card do bloco
                    bloco_selecionado = st.session_state.bloco_selecionado == bloco.id
                    
                    css_classe = "block-card selected" if bloco_selecionado else "block-card"
                    
                    st.markdown(f"""
                    <div style='
                        background-color: {cor_bloco};
                        border: 3px solid {"#fbbf24" if bloco_selecionado else "#475569"};
                        border-radius: 8px;
                        padding: 12px;
                        margin: 8px 0;
                        cursor: pointer;
                        opacity: 0.9;
                    '>
                        <div style='color: white; font-weight: bold; margin-bottom: 8px;'>
                            {bloco.name}
                        </div>
                        <div style='color: rgba(255,255,255,0.8); font-size: 12px; margin-bottom: 8px;'>
                            ID: {bloco.id}
                        </div>
                        <div style='color: rgba(255,255,255,0.7); font-size: 11px;'>
                            Tipo: {bloco.type}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Botão para selecionar
                    if st.button("Selecionar", key=f"select_{bloco.id}", use_container_width=True):
                        st.session_state.bloco_selecionado = bloco.id
                        st.rerun()
                    
                    # Botão para deletar
                    if st.button("🗑️ Deletar", key=f"delete_{bloco.id}", use_container_width=True):
                        st.session_state.blocos = [b for b in st.session_state.blocos if b.id != bloco.id]
                        st.session_state.conexoes = [c for c in st.session_state.conexoes 
                                                     if c.from_block_id != bloco.id and c.to_block_id != bloco.id]
                        st.rerun()

# ==================== PAINEL DE PROPRIEDADES ====================

if st.session_state.bloco_selecionado:
    bloco = next((b for b in st.session_state.blocos if b.id == st.session_state.bloco_selecionado), None)
    
    if bloco:
        st.markdown("---")
        st.markdown(f"### ⚙️ Propriedades de {bloco.name}")
        
        # Abas para diferentes tipos de configuração
        tab1, tab2 = st.tabs(["Parâmetros", "Conexões"])
        
        with tab1:
            st.write(f"**ID:** {bloco.id}")
            st.write(f"**Tipo:** {bloco.type}")
            
            # Parâmetros específicos por tipo
            if bloco.type == 'controller_p':
                bloco.params['kp'] = st.slider("Kp (Ganho Proporcional)", 0.1, 10.0, bloco.params['kp'], 0.1)
                st.write(f"C(s) = {bloco.params['kp']}")
            
            elif bloco.type == 'controller_pid':
                bloco.params['kp'] = st.slider("Kp", 0.1, 10.0, bloco.params['kp'], 0.1)
                bloco.params['ki'] = st.slider("Ki", 0.0, 10.0, bloco.params['ki'], 0.1)
                bloco.params['kd'] = st.slider("Kd", 0.0, 10.0, bloco.params['kd'], 0.1)
                st.write(f"C(s) = {bloco.params['kp']} + {bloco.params['ki']}/s + {bloco.params['kd']}s")
            
            elif bloco.type == 'plant_1st':
                bloco.params['gain'] = st.slider("Ganho (K)", 0.1, 10.0, bloco.params['gain'], 0.1)
                bloco.params['tau'] = st.slider("Constante de Tempo (τ)", 0.1, 5.0, bloco.params['tau'], 0.1)
                st.write(f"G(s) = {bloco.params['gain']} / ({bloco.params['tau']}s + 1)")
            
            elif bloco.type == 'plant_2nd':
                bloco.params['gain'] = st.slider("Ganho (K)", 0.1, 10.0, bloco.params['gain'], 0.1)
                bloco.params['wn'] = st.slider("Frequência Natural (ωn)", 0.1, 10.0, bloco.params['wn'], 0.1)
                bloco.params['zeta'] = st.slider("Amortecimento (ζ)", 0.0, 2.0, bloco.params['zeta'], 0.1)
                st.write(f"G(s) = {bloco.params['gain']*bloco.params['wn']**2} / (s² + {2*bloco.params['zeta']*bloco.params['wn']:.2f}s + {bloco.params['wn']**2:.2f})")
            
            elif bloco.type == 'feedback':
                bloco.params['gain'] = st.slider("Ganho", 0.1, 5.0, bloco.params['gain'], 0.1)
                st.write(f"B(s) = {bloco.params['gain']}")
        
        with tab2:
            st.write("Conexões disponíveis para este bloco")
            
            # Botão para conectar
            if st.button("➕ Adicionar Conexão", use_container_width=True):
                # Criar diálogo para selecionar destino
                blocos_destino = [b for b in st.session_state.blocos if b.id != bloco.id]
                
                if blocos_destino:
                    bloco_destino = st.selectbox(
                        "Conectar para:",
                        blocos_destino,
                        format_func=lambda x: x.name,
                        key="select_conexao"
                    )
                    
                    if st.button("Confirmar Conexão"):
                        nova_conexao = Connection(
                            id=len(st.session_state.conexoes),
                            from_block_id=bloco.id,
                            to_block_id=bloco_destino.id
                        )
                        st.session_state.conexoes.append(nova_conexao)
                        st.success(f"✅ Conexão criada: {bloco.name} → {bloco_destino.name}")
                        st.rerun()

# ==================== RESULTADOS DA SIMULAÇÃO ====================

if st.session_state.resultado_simulacao:
    st.markdown("---")
    st.markdown("### 📊 Resultados da Simulação")
    
    resultado = st.session_state.resultado_simulacao
    sistema = resultado['sistema']
    
    # Abas de resultados
    tab_degrau, tab_bode, tab_polos, tab_info = st.tabs([
        "📈 Resposta ao Degrau",
        "📊 Diagrama de Bode",
        "🎯 Polos e Zeros",
        "ℹ️ Informações"
    ])
    
    # Tab 1: Resposta ao Degrau
    with tab_degrau:
        t = np.linspace(0, 10, 1000)
        t_sim, y_sim = sistema.resposta_degrau(t)
        
        fig_degrau = go.Figure()
        
        fig_degrau.add_trace(go.Scatter(
            x=t_sim,
            y=y_sim,
            mode='lines',
            name='y(t)',
            line=dict(color='#3B82F6', width=3)
        ))
        
        fig_degrau.add_hline(
            y=sistema.ganho_dc(),
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Ganho DC"
        )
        
        fig_degrau.update_layout(
            title="Resposta ao Degrau Unitário",
            xaxis_title="Tempo (s)",
            yaxis_title="Amplitude",
            hovermode='x unified',
            plot_bgcolor='#0f172a',
            paper_bgcolor='#1e293b',
            font=dict(color='#e2e8f0'),
            height=400
        )
        
        st.plotly_chart(fig_degrau, use_container_width=True)
    
    # Tab 2: Bode
    with tab_bode:
        w = np.logspace(-2, 2, 500)
        w_bode, mag_bode, fase_bode = sistema.diagrama_bode(w)
        
        fig_bode = go.Figure()
        
        # Magnitude
        fig_bode.add_trace(go.Scatter(
            x=w_bode,
            y=mag_bode,
            mode='lines',
            name='Magnitude',
            line=dict(color='#10B981', width=2),
            yaxis='y'
        ))
        
        # Fase (eixo Y secundário)
        fig_bode.add_trace(go.Scatter(
            x=w_bode,
            y=fase_bode,
            mode='lines',
            name='Fase',
            line=dict(color='#F59E0B', width=2),
            yaxis='y2'
        ))
        
        fig_bode.update_xaxes(type='log', title='Frequência (rad/s)')
        
        fig_bode.update_yaxes(title='Magnitude (dB)', secondary_y=False)
        fig_bode.update_yaxes(title='Fase (graus)', secondary_y=True)
        
        fig_bode.update_layout(
            title="Diagrama de Bode",
            hovermode='x unified',
            plot_bgcolor='#0f172a',
            paper_bgcolor='#1e293b',
            font=dict(color='#e2e8f0'),
            height=500
        )
        
        st.plotly_chart(fig_bode, use_container_width=True)
    
    # Tab 3: Polos e Zeros
    with tab_polos:
        polos = resultado['polos']
        zeros = resultado['zeros']
        
        fig_pz = go.Figure()
        
        # Plotar polos
        if len(polos) > 0:
            fig_pz.add_trace(go.Scatter(
                x=np.real(polos),
                y=np.imag(polos),
                mode='markers',
                marker=dict(size=12, color='#EF4444', symbol='x', line=dict(width=2)),
                name='Polos'
            ))
        
        # Plotar zeros
        if len(zeros) > 0:
            fig_pz.add_trace(go.Scatter(
                x=np.real(zeros),
                y=np.imag(zeros),
                mode='markers',
                marker=dict(size=12, color='#3B82F6', symbol='circle', line=dict(width=2)),
                name='Zeros'
            ))
        
        # Eixos
        fig_pz.add_hline(y=0, line_dash="dash", line_color='rgba(255,255,255,0.3)')
        fig_pz.add_vline(x=0, line_dash="dash", line_color='rgba(255,255,255,0.3)')
        
        # Região estável (sombreada)
        fig_pz.add_vrect(
            x0=-10, x1=0,
            fillcolor='green', opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Região Estável", annotation_position="left"
        )
        
        fig_pz.update_xaxes(title='Eixo Real', zeroline=True)
        fig_pz.update_yaxes(title='Eixo Imaginário', zeroline=True)
        
        fig_pz.update_layout(
            title="Diagrama de Polos e Zeros",
            plot_bgcolor='#0f172a',
            paper_bgcolor='#1e293b',
            font=dict(color='#e2e8f0'),
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_pz, use_container_width=True)
    
    # Tab 4: Informações
    with tab_info:
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("#### Características")
            st.metric("Ganho DC", f"{resultado['ganho_dc']:.4f}")
            st.metric("Número de Polos", len(polos))
            st.metric("Número de Zeros", len(zeros))
            
            if resultado['estavel']:
                st.success("✅ **Sistema Estável**")
            else:
                st.error("❌ **Sistema Instável**")
        
        with info_col2:
            st.markdown("#### Função de Transferência")
            st.write(f"**Numerador:** {sistema.num}")
            st.write(f"**Denominador:** {sistema.den}")
            
            st.markdown("#### Polos")
            for i, polo in enumerate(polos):
                st.write(f"p{i+1} = {polo}")
            
            st.markdown("#### Zeros")
            if len(zeros) > 0:
                for i, zero in enumerate(zeros):
                    st.write(f"z{i+1} = {zero}")
            else:
                st.write("Nenhum zero")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 12px; margin-top: 20px;'>
    <p>⚙️ XCOS Simulador Profissional v2.0 | Simulação de Sistemas de Controle Clássico</p>
    <p>Desenvolvido com Streamlit | Python + NumPy + SciPy</p>
</div>
""", unsafe_allow_html=True)
