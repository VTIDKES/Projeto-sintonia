import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import control as ct
import json
from scipy import signal

# Configuração da página
st.set_page_config(page_title="Editor de Blocos Xcos - Sistemas de Controle", layout="wide")

# Inicialização do estado da sessão
if 'diagrama_blocos' not in st.session_state:
    st.session_state.diagrama_blocos = {
        'blocos': [],
        'conexoes': []
    }

if 'modo_editor' not in st.session_state:
    st.session_state.modo_editor = 'visual'

if 'sistema_calculado' not in st.session_state:
    st.session_state.sistema_calculado = None

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def criar_diagrama_blocos_html():
    """Cria o editor visual HTML/JavaScript estilo Xcos"""
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
            }
            
            #toolbar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }
            
            .btn-toolbar {
                background: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .btn-toolbar:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }
            
            .btn-toolbar:active {
                transform: translateY(0);
            }
            
            #canvas {
                width: 100%;
                height: 600px;
                background: linear-gradient(90deg, #f0f0f0 1px, transparent 1px),
                           linear-gradient(#f0f0f0 1px, transparent 1px);
                background-size: 20px 20px;
                position: relative;
                overflow: hidden;
                cursor: crosshair;
            }
            
            .bloco {
                position: absolute;
                background: white;
                border: 3px solid #667eea;
                border-radius: 10px;
                padding: 15px;
                cursor: move;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
                min-width: 120px;
                text-align: center;
                transition: all 0.2s;
                user-select: none;
            }
            
            .bloco:hover {
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
                transform: scale(1.02);
            }
            
            .bloco.selecionado {
                border-color: #ff6b6b;
                box-shadow: 0 0 0 3px rgba(255,107,107,0.3);
            }
            
            .bloco-titulo {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .bloco-tipo {
                font-size: 11px;
                color: #666;
                margin-bottom: 5px;
            }
            
            .bloco-formula {
                font-size: 13px;
                color: #333;
                font-family: 'Courier New', monospace;
                background: #f8f9fa;
                padding: 5px;
                border-radius: 4px;
                margin-top: 5px;
            }
            
            .porta {
                position: absolute;
                width: 12px;
                height: 12px;
                background: #667eea;
                border: 2px solid white;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.2s;
                z-index: 10;
            }
            
            .porta:hover {
                transform: scale(1.5);
                background: #ff6b6b;
            }
            
            .porta-entrada {
                left: -6px;
                top: 50%;
                transform: translateY(-50%);
            }
            
            .porta-saida {
                right: -6px;
                top: 50%;
                transform: translateY(-50%);
            }
            
            .conexao {
                stroke: #667eea;
                stroke-width: 3;
                fill: none;
                pointer-events: stroke;
                cursor: pointer;
            }
            
            .conexao:hover {
                stroke: #ff6b6b;
                stroke-width: 4;
            }
            
            .btn-remover {
                position: absolute;
                top: -10px;
                right: -10px;
                background: #ff6b6b;
                color: white;
                border: none;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                cursor: pointer;
                font-size: 16px;
                line-height: 1;
                display: none;
            }
            
            .bloco.selecionado .btn-remover {
                display: block;
            }
            
            #info-panel {
                position: fixed;
                bottom: 10px;
                right: 10px;
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                max-width: 300px;
            }
            
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin: 5px;
            }
            
            .status-conectando {
                background: #ffd93d;
                color: #333;
            }
            
            .status-normal {
                background: #6bcf7f;
                color: white;
            }
        </style>
    </head>
    <body>
        <div id="toolbar">
            <button class="btn-toolbar" onclick="adicionarBloco('transferencia')">
                ➕ Função Transferência
            </button>
            <button class="btn-toolbar" onclick="adicionarBloco('somador')">
                ⊕ Somador
            </button>
            <button class="btn-toolbar" onclick="adicionarBloco('ganho')">
                📊 Ganho
            </button>
            <button class="btn-toolbar" onclick="adicionarBloco('integrador')">
                ∫ Integrador
            </button>
            <button class="btn-toolbar" onclick="adicionarBloco('derivador')">
                d/dt Derivador
            </button>
            <button class="btn-toolbar" onclick="adicionarBloco('atraso')">
                ⏱️ Atraso
            </button>
            <button class="btn-toolbar" onclick="limparTudo()" style="background: #ff6b6b; color: white;">
                🗑️ Limpar Tudo
            </button>
        </div>
        
        <svg id="svg-conexoes" style="position: absolute; width: 100%; height: 600px; pointer-events: none; z-index: 1;">
        </svg>
        
        <div id="canvas"></div>
        
        <div id="info-panel">
            <div style="font-weight: bold; margin-bottom: 10px; color: #667eea;">ℹ️ Informações</div>
            <div id="status">
                <span class="status-badge status-normal">Normal</span>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
                <div>Blocos: <strong id="count-blocos">0</strong></div>
                <div>Conexões: <strong id="count-conexoes">0</strong></div>
            </div>
        </div>
        
        <script>
            let blocos = [];
            let conexoes = [];
            let blocoSelecionado = null;
            let portaConectando = null;
            let draggingBloco = null;
            let offsetX = 0;
            let offsetY = 0;
            let blocoIdCounter = 0;
            
            function adicionarBloco(tipo) {
                const canvas = document.getElementById('canvas');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + blocoIdCounter;
                
                const blocoData = {
                    id: blocoIdCounter,
                    tipo: tipo,
                    x: 100 + Math.random() * 300,
                    y: 100 + Math.random() * 200,
                    parametros: {}
                };
                
                let titulo = '';
                let formula = '';
                
                switch(tipo) {
                    case 'transferencia':
                        titulo = 'G(s)';
                        formula = '1/(s+1)';
                        blocoData.parametros = {num: [1], den: [1, 1]};
                        break;
                    case 'somador':
                        titulo = 'Σ';
                        formula = 'u1 + u2';
                        blocoData.parametros = {sinais: ['+', '+']};
                        break;
                    case 'ganho':
                        titulo = 'K';
                        formula = 'K = 1.0';
                        blocoData.parametros = {K: 1.0};
                        break;
                    case 'integrador':
                        titulo = '∫';
                        formula = '1/s';
                        blocoData.parametros = {num: [1], den: [1, 0]};
                        break;
                    case 'derivador':
                        titulo = 'd/dt';
                        formula = 's';
                        blocoData.parametros = {num: [1, 0], den: [1]};
                        break;
                    case 'atraso':
                        titulo = 'Atraso';
                        formula = '1/(τs+1)';
                        blocoData.parametros = {tau: 1.0};
                        break;
                }
                
                bloco.innerHTML = `
                    <div class="bloco-titulo">${titulo}</div>
                    <div class="bloco-tipo">${tipo}</div>
                    <div class="bloco-formula">${formula}</div>
                    <div class="porta porta-entrada" onclick="conectarPorta(event, ${blocoIdCounter}, 'entrada')"></div>
                    <div class="porta porta-saida" onclick="conectarPorta(event, ${blocoIdCounter}, 'saida')"></div>
                    <button class="btn-remover" onclick="removerBloco(${blocoIdCounter})">×</button>
                `;
                
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                
                bloco.addEventListener('mousedown', startDrag);
                bloco.addEventListener('click', (e) => {
                    if (e.target.className === 'bloco' || e.target.className.includes('bloco-')) {
                        selecionarBloco(blocoIdCounter);
                    }
                });
                
                canvas.appendChild(bloco);
                blocos.push(blocoData);
                blocoIdCounter++;
                
                atualizarContadores();
                enviarDadosParent();
            }
            
            function startDrag(e) {
                if (e.target.classList.contains('porta') || e.target.classList.contains('btn-remover')) {
                    return;
                }
                
                draggingBloco = e.currentTarget;
                const rect = draggingBloco.getBoundingClientRect();
                offsetX = e.clientX - rect.left;
                offsetY = e.clientY - rect.top;
                
                document.addEventListener('mousemove', drag);
                document.addEventListener('mouseup', stopDrag);
            }
            
            function drag(e) {
                if (!draggingBloco) return;
                
                const canvas = document.getElementById('canvas');
                const rect = canvas.getBoundingClientRect();
                
                let x = e.clientX - rect.left - offsetX;
                let y = e.clientY - rect.top - offsetY;
                
                x = Math.max(0, Math.min(x, rect.width - draggingBloco.offsetWidth));
                y = Math.max(0, Math.min(y, rect.height - draggingBloco.offsetHeight));
                
                draggingBloco.style.left = x + 'px';
                draggingBloco.style.top = y + 'px';
                
                const blocoId = parseInt(draggingBloco.id.split('-')[1]);
                const bloco = blocos.find(b => b.id === blocoId);
                if (bloco) {
                    bloco.x = x;
                    bloco.y = y;
                }
                
                redesenharConexoes();
            }
            
            function stopDrag() {
                draggingBloco = null;
                document.removeEventListener('mousemove', drag);
                document.removeEventListener('mouseup', stopDrag);
                enviarDadosParent();
            }
            
            function selecionarBloco(id) {
                document.querySelectorAll('.bloco').forEach(b => {
                    b.classList.remove('selecionado');
                });
                
                const bloco = document.getElementById('bloco-' + id);
                if (bloco) {
                    bloco.classList.add('selecionado');
                    blocoSelecionado = id;
                }
            }
            
            function conectarPorta(e, blocoId, tipo) {
                e.stopPropagation();
                
                if (!portaConectando) {
                    portaConectando = {blocoId: blocoId, tipo: tipo};
                    document.getElementById('status').innerHTML = 
                        '<span class="status-badge status-conectando">Conectando...</span>';
                } else {
                    if (portaConectando.tipo === 'saida' && tipo === 'entrada') {
                        conexoes.push({
                            de: portaConectando.blocoId,
                            para: blocoId
                        });
                        redesenharConexoes();
                        enviarDadosParent();
                    } else if (portaConectando.tipo === 'entrada' && tipo === 'saida') {
                        conexoes.push({
                            de: blocoId,
                            para: portaConectando.blocoId
                        });
                        redesenharConexoes();
                        enviarDadosParent();
                    }
                    
                    portaConectando = null;
                    document.getElementById('status').innerHTML = 
                        '<span class="status-badge status-normal">Normal</span>';
                    atualizarContadores();
                }
            }
            
            function redesenharConexoes() {
                const svg = document.getElementById('svg-conexoes');
                svg.innerHTML = '';
                
                conexoes.forEach((conexao, index) => {
                    const blocoOrigem = document.getElementById('bloco-' + conexao.de);
                    const blocoDestino = document.getElementById('bloco-' + conexao.para);
                    
                    if (!blocoOrigem || !blocoDestino) return;
                    
                    const rectOrigem = blocoOrigem.getBoundingClientRect();
                    const rectDestino = blocoDestino.getBoundingClientRect();
                    const canvas = document.getElementById('canvas').getBoundingClientRect();
                    
                    const x1 = rectOrigem.right - canvas.left;
                    const y1 = rectOrigem.top + rectOrigem.height/2 - canvas.top;
                    const x2 = rectDestino.left - canvas.left;
                    const y2 = rectDestino.top + rectDestino.height/2 - canvas.top;
                    
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    const dx = (x2 - x1) / 2;
                    const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
                    
                    path.setAttribute('d', d);
                    path.setAttribute('class', 'conexao');
                    path.onclick = () => removerConexao(index);
                    
                    svg.appendChild(path);
                    
                    // Seta
                    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    const angle = Math.atan2(y2 - y1, x2 - x1);
                    const arrowSize = 8;
                    const arrowPoints = [
                        [x2, y2],
                        [x2 - arrowSize * Math.cos(angle - Math.PI/6), y2 - arrowSize * Math.sin(angle - Math.PI/6)],
                        [x2 - arrowSize * Math.cos(angle + Math.PI/6), y2 - arrowSize * Math.sin(angle + Math.PI/6)]
                    ];
                    arrow.setAttribute('points', arrowPoints.map(p => p.join(',')).join(' '));
                    arrow.setAttribute('fill', '#667eea');
                    svg.appendChild(arrow);
                });
            }
            
            function removerBloco(id) {
                const bloco = document.getElementById('bloco-' + id);
                if (bloco) {
                    bloco.remove();
                }
                
                blocos = blocos.filter(b => b.id !== id);
                conexoes = conexoes.filter(c => c.de !== id && c.para !== id);
                
                redesenharConexoes();
                atualizarContadores();
                enviarDadosParent();
            }
            
            function removerConexao(index) {
                if (confirm('Remover esta conexão?')) {
                    conexoes.splice(index, 1);
                    redesenharConexoes();
                    atualizarContadores();
                    enviarDadosParent();
                }
            }
            
            function limparTudo() {
                if (confirm('Limpar todo o diagrama?')) {
                    document.getElementById('canvas').innerHTML = '';
                    blocos = [];
                    conexoes = [];
                    blocoIdCounter = 0;
                    redesenharConexoes();
                    atualizarContadores();
                    enviarDadosParent();
                }
            }
            
            function atualizarContadores() {
                document.getElementById('count-blocos').textContent = blocos.length;
                document.getElementById('count-conexoes').textContent = conexoes.length;
            }
            
            function enviarDadosParent() {
                const dados = {
                    blocos: blocos,
                    conexoes: conexoes
                };
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: dados
                }, '*');
            }
            
            // Redesenhar conexões quando a janela é redimensionada
            window.addEventListener('resize', redesenharConexoes);
            
            // Inicialização
            atualizarContadores();
        </script>
    </body>
    </html>
    """
    
    return html_code

def processar_diagrama_blocos():
    """Processa o diagrama de blocos e calcula o sistema"""
    try:
        diagrama = st.session_state.diagrama_blocos
        
        if not diagrama['blocos']:
            return None, "⚠️ Nenhum bloco no diagrama"
        
        # Criar dicionário de funções de transferência
        sistemas = {}
        
        for bloco in diagrama['blocos']:
            bloco_id = bloco['id']
            tipo = bloco['tipo']
            params = bloco['parametros']
            
            if tipo == 'transferencia':
                tf = ct.TransferFunction(params['num'], params['den'])
            elif tipo == 'ganho':
                tf = ct.TransferFunction([params['K']], [1])
            elif tipo == 'integrador':
                tf = ct.TransferFunction([1], [1, 0])
            elif tipo == 'derivador':
                tf = ct.TransferFunction([1, 0], [1])
            elif tipo == 'atraso':
                tf = ct.TransferFunction([1], [params['tau'], 1])
            elif tipo == 'somador':
                # Somador será tratado na conexão
                tf = ct.TransferFunction([1], [1])
            else:
                tf = ct.TransferFunction([1], [1])
            
            sistemas[bloco_id] = tf
        
        # Processar conexões em série
        if not diagrama['conexoes']:
            # Se não há conexões, usar o primeiro bloco
            sistema_final = list(sistemas.values())[0]
        else:
            # Encontrar bloco inicial (sem entrada)
            blocos_com_entrada = set([c['para'] for c in diagrama['conexoes']])
            blocos_iniciais = [b['id'] for b in diagrama['blocos'] if b['id'] not in blocos_com_entrada]
            
            if not blocos_iniciais:
                blocos_iniciais = [diagrama['blocos'][0]['id']]
            
            # Conectar em série seguindo as conexões
            sistema_final = sistemas[blocos_iniciais[0]]
            visitados = {blocos_iniciais[0]}
            
            mudou = True
            while mudou:
                mudou = False
                for conexao in diagrama['conexoes']:
                    if conexao['de'] in visitados and conexao['para'] not in visitados:
                        sistema_final = ct.series(sistema_final, sistemas[conexao['para']])
                        visitados.add(conexao['para'])
                        mudou = True
        
        st.session_state.sistema_calculado = sistema_final
        return sistema_final, "✅ Sistema calculado com sucesso!"
        
    except Exception as e:
        return None, f"❌ Erro ao processar diagrama: {str(e)}"

def calcular_desempenho(sistema):
    """Calcula métricas de desempenho do sistema"""
    try:
        info = ct.step_info(sistema)
        
        desempenho = {
            "Tempo de Subida": f"{info['RiseTime']:.3f} s",
            "Tempo de Pico": f"{info['PeakTime']:.3f} s",
            "Tempo de Acomodação": f"{info['SettlingTime']:.3f} s",
            "Overshoot": f"{info['Overshoot']:.2f} %",
            "Valor Final": f"{info['SteadyStateValue']:.3f}",
            "Estabilidade": "Estável" if np.all(np.real(ct.pole(sistema)) < 0) else "Instável"
        }
        
        return desempenho
    except:
        return {"Status": "Não foi possível calcular métricas"}

def plot_resposta_temporal(sistema, tipo='Degrau'):
    """Plota resposta temporal usando Plotly"""
    t = np.linspace(0, 10, 1000)
    
    if tipo == 'Degrau':
        t_out, y_out = ct.step_response(sistema, t)
        titulo = "Resposta ao Degrau"
    elif tipo == 'Impulso':
        t_out, y_out = ct.impulse_response(sistema, t)
        titulo = "Resposta ao Impulso"
    else:
        t_out, y_out = ct.step_response(sistema/ct.TransferFunction([1], [1, 0]), t)
        titulo = "Resposta à Rampa"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_out, y=y_out, mode='lines', name='Resposta',
                            line=dict(color='#667eea', width=3)))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig, t_out, y_out

def plot_bode(sistema, plot_type='both'):
    """Plota diagrama de Bode usando Plotly"""
    w = np.logspace(-2, 3, 1000)
    mag, phase, omega = ct.bode(sistema, w, plot=False)
    
    mag_db = 20 * np.log10(mag)
    phase_deg = np.rad2deg(phase)
    
    if plot_type == 'both':
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Magnitude", "Fase"),
                           vertical_spacing=0.1)
        
        fig.add_trace(go.Scatter(x=omega, y=mag_db, mode='lines',
                                name='Magnitude', line=dict(color='#667eea', width=2)),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=omega, y=phase_deg, mode='lines',
                                name='Fase', line=dict(color='#764ba2', width=2)),
                     row=2, col=1)
        
        fig.update_xaxes(type="log", title_text="Frequência (rad/s)", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (graus)", row=2, col=1)
        
    fig.update_layout(height=600, template="plotly_white", showlegend=False)
    
    return fig

def plot_polos_zeros(sistema):
    """Plota diagrama de polos e zeros usando Plotly"""
    polos = ct.pole(sistema)
    zeros = ct.zero(sistema)
    
    fig = go.Figure()
    
    # Polos
    fig.add_trace(go.Scatter(
        x=np.real(polos), y=np.imag(polos),
        mode='markers',
        name='Polos',
        marker=dict(symbol='x', size=15, color='red', line=dict(width=2))
    ))
    
    # Zeros
    fig.add_trace(go.Scatter(
        x=np.real(zeros), y=np.imag(zeros),
        mode='markers',
        name='Zeros',
        marker=dict(symbol='circle', size=12, color='blue', line=dict(width=2))
    ))
    
    # Eixos
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Diagrama de Polos e Zeros",
        xaxis_title="Parte Real",
        yaxis_title="Parte Imaginária",
        template="plotly_white",
        hovermode='closest',
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig

# =====================================================
# INTERFACE PRINCIPAL
# =====================================================

st.title("🎨 Editor de Blocos Xcos - Sistemas de Controle")
st.markdown("*Editor visual interativo para análise de sistemas de controle*")

# Sidebar
st.sidebar.header("⚙️ Configurações")

modo = st.sidebar.radio(
    "Modo de Editor:",
    ["visual", "texto"],
    format_func=lambda x: "🎨 Editor Visual (Xcos)" if x == "visual" else "📝 Editor de Texto"
)

st.session_state.modo_editor = modo

# =====================================================
# MODO EDITOR VISUAL
# =====================================================
if st.session_state.modo_editor == 'visual':
    st.subheader("🎨 Editor Visual de Diagrama de Blocos")
    st.info("💡 **Modo Xcos ativado!** Arraste blocos, conecte portas e construa seu sistema visualmente.")
    
    html_editor = criar_diagrama_blocos_html()
    diagrama_data = components.html(html_editor, height=700, scrolling=False)
    
    # Atualizar estado se houver dados
    if diagrama_data:
        st.session_state.diagrama_blocos = diagrama_data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⚡ Processar Diagrama", type="primary", use_container_width=True):
            sistema, msg = processar_diagrama_blocos()
            if sistema:
                st.success(msg)
                st.subheader("📊 Análise do Sistema")
                
                desempenho = calcular_desempenho(sistema)
                st.markdown("**Métricas de Desempenho:**")
                for chave, valor in desempenho.items():
                    st.markdown(f"- **{chave}:** {valor}")
                
                tab1, tab2, tab3 = st.tabs(["📈 Resposta Temporal", "📊 Bode", "🎯 Polos e Zeros"])
                
                with tab1:
                    fig, t, y = plot_resposta_temporal(sistema, 'Degrau')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = plot_bode(sistema, 'both')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    fig = plot_polos_zeros(sistema)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(msg)
    
    with col2:
        if st.button("💾 Exportar Diagrama", use_container_width=True):
            diagrama_json = json.dumps(st.session_state.diagrama_blocos, indent=2)
            st.download_button(
                label="📥 Baixar JSON",
                data=diagrama_json,
                file_name="diagrama_blocos.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        if st.button("📖 Ajuda", use_container_width=True):
            st.info("""
            **Como usar o Editor Visual:**
            
            1. **Adicionar Blocos:** Clique nos botões da barra superior
            2. **Mover Blocos:** Arraste os blocos pela área
            3. **Conectar:** Clique na porta de saída (⚫), depois na entrada
            4. **Selecionar:** Clique no bloco
            5. **Remover:** Selecione e clique em "🗑️ Remover"
            
            **Tipos de Blocos:**
            - ➕ **Função Transferência:** G(s) = num/den
            - ⊕ **Somador:** Soma/subtrai sinais
            - 📊 **Ganho:** Multiplicador K
            - ∫ **Integrador:** 1/s
            - d/dt **Derivador:** s
            - ⏱️ **Atraso:** 1/(τs+1)
            """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estatísticas")
    st.sidebar.metric("Blocos", len(st.session_state.diagrama_blocos.get('blocos', [])))
    st.sidebar.metric("Conexões", len(st.session_state.diagrama_blocos.get('conexoes', [])))

# =====================================================
# MODO EDITOR DE TEXTO
# =====================================================
else:
    st.subheader("📝 Editor de Texto")
    st.info("Configure o sistema usando entrada de texto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerador**")
        num_str = st.text_input("Coeficientes (separados por vírgula):", "1")
        
    with col2:
        st.markdown("**Denominador**")
        den_str = st.text_input("Coeficientes (separados por vírgula):", "1,1")
    
    if st.button("Calcular Sistema"):
        try:
            num = [float(x.strip()) for x in num_str.split(',')]
            den = [float(x.strip()) for x in den_str.split(',')]
            sistema = ct.TransferFunction(num, den)
            st.session_state.sistema_calculado = sistema
            
            st.success("✅ Sistema calculado!")
            
            desempenho = calcular_desempenho(sistema)
            st.markdown("**Métricas de Desempenho:**")
            for chave, valor in desempenho.items():
                st.markdown(f"- **{chave}:** {valor}")
            
            tab1, tab2, tab3 = st.tabs(["📈 Resposta Temporal", "📊 Bode", "🎯 Polos e Zeros"])
            
            with tab1:
                fig, t, y = plot_resposta_temporal(sistema, 'Degrau')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = plot_bode(sistema, 'both')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = plot_polos_zeros(sistema)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erro: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**💡 Dica:** Use o modo visual para construir sistemas complexos de forma intuitiva!")
