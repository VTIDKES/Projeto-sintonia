import streamlit as st
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import graphviz

# Configuração da Página
st.set_page_config(page_title="Sistema de Controle - Web Designer", layout="wide", page_icon="⚙️")

# ==========================================
# Lógica de Controle
# ==========================================
def parse_poly(poly_str):
    """Converte string '1, 2, 1' para lista de floats [1.0, 2.0, 1.0]"""
    try:
        # Remove espaços e colchetes se houver
        clean_str = poly_str.replace('[', '').replace(']', '').strip()
        if not clean_str:
            return [1.0]
        return [float(x) for x in clean_str.split(',')]
    except:
        return [1.0]

def get_transfer_function(num_str, den_str):
    num = parse_poly(num_str)
    den = parse_poly(den_str)
    return signal.TransferFunction(num, den)

# ==========================================
# Gerenciamento de Estado (Session State)
# ==========================================
if 'blocks' not in st.session_state:
    st.session_state.blocks = [
        {'id': 1, 'type': 'input', 'label': 'R(s)', 'data': {}},
        {'id': 2, 'type': 'transfer', 'label': 'G(s)', 'data': {'num': '1', 'den': '1, 1, 1'}},
        {'id': 3, 'type': 'output', 'label': 'Y(s)', 'data': {}}
    ]

if 'connections' not in st.session_state:
    st.session_state.connections = [
        {'from': 1, 'to': 2},
        {'from': 2, 'to': 3}
    ]

if 'next_id' not in st.session_state:
    st.session_state.next_id = 4

# ==========================================
# Interface - Barra Lateral
# ==========================================
with st.sidebar:
    st.title("🛠️ Configuração")
    
    # --- Adicionar Blocos ---
    st.subheader("Adicionar Elemento")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Função Transf."):
            st.session_state.blocks.append({
                'id': st.session_state.next_id,
                'type': 'transfer',
                'label': f'G{st.session_state.next_id}(s)',
                'data': {'num': '1', 'den': '1, 1'}
            })
            st.session_state.next_id += 1
            st.rerun()
    with col2:
        if st.button("➕ Somador"):
            st.session_state.blocks.append({
                'id': st.session_state.next_id,
                'type': 'sum',
                'label': 'Sum',
                'data': {}
            })
            st.session_state.next_id += 1
            st.rerun()

    st.markdown("---")
    
    # --- Gerenciar Conexões ---
    st.subheader("Conexões")
    
    # Criar lista de opções para dropdown
    block_options = {b['id']: f"{b['id']}: {b['label']}" for b in st.session_state.blocks}
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        source_id = st.selectbox("De:", options=list(block_options.keys()), format_func=lambda x: block_options[x], key="src_sel")
    with c_col2:
        target_id = st.selectbox("Para:", options=list(block_options.keys()), format_func=lambda x: block_options[x], key="tgt_sel")
        
    if st.button("🔗 Conectar"):
        # Evitar duplicatas
        if not any(c['from'] == source_id and c['to'] == target_id for c in st.session_state.connections):
            st.session_state.connections.append({'from': source_id, 'to': target_id})
            st.rerun()
            
    if st.button("❌ Limpar Conexões"):
        st.session_state.connections = []
        st.rerun()

    st.markdown("---")

    # --- Editar Blocos ---
    st.subheader("Editar Bloco Selecionado")
    selected_block_id = st.selectbox("Selecione para editar:", options=list(block_options.keys()), format_func=lambda x: block_options[x])
    
    # Encontrar o bloco selecionado
    current_block = next((b for b in st.session_state.blocks if b['id'] == selected_block_id), None)
    
    if current_block:
        new_label = st.text_input("Nome/Rótulo", value=current_block['label'])
        current_block['label'] = new_label
        
        if current_block['type'] == 'transfer':
            num_val = st.text_input("Numerador (ex: 1)", value=current_block['data'].get('num', '1'))
            den_val = st.text_input("Denominador (ex: 1, 2, 1)", value=current_block['data'].get('den', '1, 1'))
            current_block['data']['num'] = num_val
            current_block['data']['den'] = den_val
        
        if st.button("🗑️ Deletar Bloco", type="primary"):
            st.session_state.blocks = [b for b in st.session_state.blocks if b['id'] != selected_block_id]
            st.session_state.connections = [c for c in st.session_state.connections if c['from'] != selected_block_id and c['to'] != selected_block_id]
            st.rerun()

# ==========================================
# Interface - Área Principal
# ==========================================
st.title("Sistema de Controle Dinâmico")

tab1, tab2 = st.tabs(["📊 Diagrama de Blocos", "📈 Análise do Sistema"])

with tab1:
    st.markdown("### Visualização do Sistema")
    
    # Usar Graphviz para desenhar o diagrama
    if st.session_state.blocks:
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR')
        
        for block in st.session_state.blocks:
            shape = 'box'
            color = 'lightblue'
            style = 'filled'
            
            if block['type'] == 'sum':
                shape = 'circle'
                color = 'lightgreen'
            elif block['type'] in ['input', 'output']:
                shape = 'ellipse'
                color = 'white'
                style = ''
                
            label = block['label']
            if block['type'] == 'transfer':
                label = f"{block['label']}\n{block['data']['num']}\n----------\n{block['data']['den']}"
                
            graph.node(str(block['id']), label, shape=shape, style=style, fillcolor=color)
            
        for conn in st.session_state.connections:
            graph.edge(str(conn['from']), str(conn['to']))
            
        st.graphviz_chart(graph)
    else:
        st.info("Adicione blocos na barra lateral para começar.")

with tab2:
    st.markdown("### Análise de Resposta")
    st.caption("A análise é realizada no bloco de Função de Transferência selecionado abaixo.")
    
    # Filtrar apenas blocos de transferência
    tf_blocks = [b for b in st.session_state.blocks if b['type'] == 'transfer']
    
    if not tf_blocks:
        st.warning("Adicione pelo menos um bloco de Função de Transferência para analisar.")
    else:
        # Selecionar qual bloco analisar
        analyze_id = st.selectbox("Qual bloco analisar?", options=[b['id'] for b in tf_blocks], format_func=lambda x: next(b['label'] for b in tf_blocks if b['id'] == x))
        target_block = next(b for b in tf_blocks if b['id'] == analyze_id)
        
        try:
            sys = get_transfer_function(target_block['data']['num'], target_block['data']['den'])
            
            # Criar gráficos com Matplotlib
            fig = plt.figure(figsize=(10, 12))
            plt.style.use('dark_background')
            
            # 1. Diagrama de Bode
            ax1 = plt.subplot(3, 1, 1)
            w, mag, phase = sys.bode()
            ax1.semilogx(w, mag, color='cyan')
            ax1.set_title(f'Diagrama de Bode - {target_block["label"]}')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.grid(True, which='both', linestyle='--', alpha=0.3)
            
            # 2. Resposta ao Degrau
            ax2 = plt.subplot(3, 1, 2)
            t, y = signal.step(sys)
            ax2.plot(t, y, color='magenta')
            ax2.set_title('Resposta ao Degrau')
            ax2.set_xlabel('Tempo (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            # 3. Mapa Polos e Zeros
            ax3 = plt.subplot(3, 1, 3)
            # Extrair polos e zeros
            zeros = sys.zeros
            poles = sys.poles
            
            ax3.scatter(np.real(zeros), np.imag(zeros), s=100, marker='o', facecolors='none', edgecolors='cyan', label='Zeros')
            ax3.scatter(np.real(poles), np.imag(poles), s=100, marker='x', color='magenta', label='Polos')
            ax3.axhline(0, color='white', alpha=0.3)
            ax3.axvline(0, color='white', alpha=0.3)
            ax3.set_title('Mapa de Polos e Zeros')
            ax3.set_xlabel('Real')
            ax3.set_ylabel('Imaginário')
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro ao calcular o sistema: {str(e)}. Verifique os coeficientes.")
