def criar_diagrama_blocos_html():
    """Cria o editor visual de diagrama de blocos"""
    blocos_data = json.dumps(st.session_state.diagrama_blocos['blocos'])
    conexoes_data = json.dumps(st.session_state.diagrama_blocos['conexoes'])
    contador = st.session_state.bloco_contador
    
    # Usar substituição direta
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }
        #canvas-container {
            width: 100%;
            height: 600px;
            background: linear-gradient(#f0f0f0 1px, transparent 1px),
                        linear-gradient(90deg, #f0f0f0 1px, transparent 1px);
            background-size: 20px 20px;
            position: relative;
            border: 2px solid #ddd;
            cursor: crosshair;
        }
        .bloco {
            position: absolute;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            cursor: move;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            min-width: 120px;
            text-align: center;
            user-select: none;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        .bloco:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .bloco.selecionado {
            border: 3px solid #ffd700;
            box-shadow: 0 0 20px rgba(255,215,0,0.5);
        }
        .bloco-tipo {
            font-size: 10px;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        .bloco-nome {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .bloco-tf {
            font-size: 11px;
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.2);
            padding: 5px;
            border-radius: 4px;
            margin-top: 5px;
        }
        .porta {
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border: 2px solid white;
            border-radius: 50%;
            position: absolute;
            cursor: pointer;
            transition: all 0.2s;
        }
        .porta:hover {
            background: #45a049;
            transform: scale(1.3);
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
        .toolbar {
            background: #333;
            color: white;
            padding: 10px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #5568d3;
        }
        .btn-danger {
            background: #e74c3c;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        #info-panel {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            max-width: 250px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <button class="btn" onclick="adicionarBlocoTransferencia()">➕ Função Transferência</button>
        <button class="btn" onclick="adicionarBlocoSomador()">⊕ Somador</button>
        <button class="btn" onclick="adicionarBlocoGanho()">📊 Ganho</button>
        <button class="btn" onclick="adicionarBlocoIntegrador()">∫ Integrador</button>
        <button class="btn" onclick="adicionarBlocoAtraso()">⏱️ Atraso</button>
        <button class="btn" onclick="adicionarBlocoFeedback()">🔄 Feedback</button>
        <button class="btn btn-danger" onclick="removerSelecionado()">🗑️ Remover</button>
        <button class="btn btn-danger" onclick="limparDiagrama()">🔄 Limpar</button>
    </div>
    
    <div id="canvas-container">
        <svg id="conexoes-svg"></svg>
        <div id="info-panel">
            <strong>📊 Instruções:</strong>
            <div>• Clique nos botões para adicionar blocos</div>
            <div>• Arraste blocos para mover</div>
            <div>• Clique nas portas verdes para conectar</div>
            <div>• Clique no bloco para selecionar</div>
        </div>
    </div>

    <script>
        let blocos = BLOCOS_DATA;
        let conexoes = CONEXOES_DATA;
        let blocoIdCounter = CONTADOR;
        let blocoSelecionado = null;
        let portaSelecionada = null;
        let arrastandoBloco = null;
        let offsetX = 0, offsetY = 0;

        function adicionarBloco(tipo, config) {
            const container = document.getElementById("canvas-container");
            const bloco = document.createElement("div");
            bloco.className = "bloco";
            bloco.id = "bloco-" + blocoIdCounter;
            
            const blocoData = {
                id: blocoIdCounter,
                tipo: tipo,
                x: 100 + Math.random() * 200,
                y: 100 + Math.random() * 200,
                config: config
            };
            
            blocos.push(blocoData);
            
            bloco.style.left = blocoData.x + "px";
            bloco.style.top = blocoData.y + "px";
            
            let nomeDisplay = config.nome || tipo;
            let tfDisplay = config.tf || "";
            
            bloco.innerHTML = `
                <div class="bloco-tipo">${tipo}</div>
                <div class="bloco-nome">${nomeDisplay}</div>
                ${tfDisplay ? '<div class="bloco-tf">' + tfDisplay + "</div>" : ""}
                <div class="porta porta-entrada" data-bloco="${blocoIdCounter}" data-tipo="entrada"></div>
                <div class="porta porta-saida" data-bloco="${blocoIdCounter}" data-tipo="saida"></div>
            `;
            
            container.appendChild(bloco);
            
            bloco.addEventListener("mousedown", iniciarArrastar);
            bloco.addEventListener("click", selecionarBloco);
            
            const portas = bloco.querySelectorAll(".porta");
            portas.forEach(porta => {
                porta.addEventListener("click", clickPorta);
            });
            
            blocoIdCounter++;
            salvarEstado();
        }

        function adicionarBlocoTransferencia() {
            const num = prompt("Numerador (ex: 1, s+1):", "1");
            if (num === null) return;
            const den = prompt("Denominador (ex: s+1, s^2+2*s+1):", "s+1");
            if (den === null) return;
            adicionarBloco("Transferência", {
                nome: "G" + blocoIdCounter,
                numerador: num,
                denominador: den,
                tf: num + " / " + den
            });
        }

        function adicionarBlocoSomador() {
            adicionarBloco("Somador", {nome: "Σ" + blocoIdCounter});
        }

        function adicionarBlocoGanho() {
            const ganho = prompt("Valor do ganho K:", "1");
            if (ganho === null) return;
            adicionarBloco("Ganho", {nome: "K=" + ganho, valor: ganho, tf: ganho});
        }

        function adicionarBlocoIntegrador() {
            adicionarBloco("Integrador", {nome: "∫", tf: "1/s"});
        }

        function adicionarBlocoAtraso() {
            const atraso = prompt("Tempo de atraso τ:", "1");
            if (atraso === null) return;
            adicionarBloco("Atraso", {nome: "τ=" + atraso, tf: "e^{ -" + atraso + "s}"});
        }

        function adicionarBlocoFeedback() {
            adicionarBloco("Feedback", {nome: "H" + blocoIdCounter});
        }

        function iniciarArrastar(e) {
            if (e.target.classList.contains("porta")) return;
            e.stopPropagation();
            arrastandoBloco = e.currentTarget;
            const rect = arrastandoBloco.getBoundingClientRect();
            const container = document.getElementById("canvas-container").getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            
            document.addEventListener("mousemove", arrastar);
            document.addEventListener("mouseup", pararArrastar);
        }

        function arrastar(e) {
            if (arrastandoBloco) {
                const container = document.getElementById("canvas-container").getBoundingClientRect();
                let x = e.clientX - container.left - offsetX;
                let y = e.clientY - container.top - offsetY;
                
                x = Math.max(0, Math.min(x, container.width - arrastandoBloco.offsetWidth));
                y = Math.max(0, Math.min(y, container.height - arrastandoBloco.offsetHeight));
                
                arrastandoBloco.style.left = x + "px";
                arrastandoBloco.style.top = y + "px";
                
                const blocoId = parseInt(arrastandoBloco.id.split("-")[1]);
                const bloco = blocos.find(b => b.id === blocoId);
                if (bloco) {
                    bloco.x = x;
                    bloco.y = y;
                }
                
                redesenharConexoes();
            }
        }

        function pararArrastar() {
            arrastandoBloco = null;
            document.removeEventListener("mousemove", arrastar);
            document.removeEventListener("mouseup", pararArrastar);
            salvarEstado();
        }

        function selecionarBloco(e) {
            if (e.target.classList.contains("porta")) return;
            e.stopPropagation();
            
            document.querySelectorAll(".bloco").forEach(b => b.classList.remove("selecionado"));
            e.currentTarget.classList.add("selecionado");
            blocoSelecionado = parseInt(e.currentTarget.id.split("-")[1]);
        }

        function clickPorta(e) {
            e.stopPropagation();
            const blocoId = parseInt(e.target.dataset.bloco);
            const tipoPorta = e.target.dataset.tipo;
            
            if (!portaSelecionada) {
                portaSelecionada = {blocoId: blocoId, tipo: tipoPorta};
                e.target.style.background = "#FFC107";
            } else {
                if (portaSelecionada.tipo === "saida" && tipoPorta === "entrada") {
                    conexoes.push({
                        origem: portaSelecionada.blocoId,
                        destino: blocoId
                    });
                    redesenharConexoes();
                } else if (portaSelecionada.tipo === "entrada" && tipoPorta === "saida") {
                    conexoes.push({
                        origem: blocoId,
                        destino: portaSelecionada.blocoId
                    });
                    redesenharConexoes();
                } else {
                    alert("Conecte saída → entrada");
                }
                
                document.querySelectorAll(".porta").forEach(p => p.style.background = "#4CAF50");
                portaSelecionada = null;
                salvarEstado();
            }
        }

        function redesenharConexoes() {
            const svg = document.getElementById("conexoes-svg");
            svg.innerHTML = "";
            
            conexoes.forEach(conexao => {
                const blocoOrigem = document.getElementById("bloco-" + conexao.origem);
                const blocoDestino = document.getElementById("bloco-" + conexao.destino);
                
                if (blocoOrigem && blocoDestino) {
                    const rectOrigem = blocoOrigem.getBoundingClientRect();
                    const rectDestino = blocoDestino.getBoundingClientRect();
                    const container = document.getElementById("canvas-container").getBoundingClientRect();
                    
                    const x1 = rectOrigem.right - container.left;
                    const y1 = rectOrigem.top + rectOrigem.height/2 - container.top;
                    const x2 = rectDestino.left - container.left;
                    const y2 = rectDestino.top + rectDestino.height/2 - container.top;
                    
                    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    const dx = x2 - x1;
                    const curva = Math.abs(dx) / 2;
                    const d = "M " + x1 + " " + y1 + " C " + (x1 + curva) + " " + y1 + ", " + (x2 - curva) + " " + y2 + ", " + x2 + " " + y2;
                    
                    path.setAttribute("d", d);
                    path.setAttribute("stroke", "#667eea");
                    path.setAttribute("stroke-width", "3");
                    path.setAttribute("fill", "none");
                    path.setAttribute("marker-end", "url(#arrowhead)");
                    
                    svg.appendChild(path);
                }
            });
            
            const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
            marker.setAttribute("id", "arrowhead");
            marker.setAttribute("markerWidth", "10");
            marker.setAttribute("markerHeight", "10");
            marker.setAttribute("refX", "9");
            marker.setAttribute("refY", "3");
            marker.setAttribute("orient", "auto");
            const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            polygon.setAttribute("points", "0 0, 10 3, 0 6");
            polygon.setAttribute("fill", "#667eea");
            marker.appendChild(polygon);
            defs.appendChild(marker);
            svg.appendChild(defs);
        }

        function removerSelecionado() {
            if (blocoSelecionado !== null) {
                const blocoEl = document.getElementById("bloco-" + blocoSelecionado);
                if (blocoEl) {
                    blocoEl.remove();
                    blocos = blocos.filter(b => b.id !== blocoSelecionado);
                    conexoes = conexoes.filter(c => c.origem !== blocoSelecionado && c.destino !== blocoSelecionado);
                    redesenharConexoes();
                    blocoSelecionado = null;
                    salvarEstado();
                }
            } else {
                alert("Selecione um bloco primeiro!");
            }
        }

        function limparDiagrama() {
            if (confirm("Deseja limpar todo o diagrama?")) {
                blocos = [];
                conexoes = [];
                blocoSelecionado = null;
                document.getElementById("canvas-container").innerHTML = '<svg id="conexoes-svg"></svg><div id="info-panel"><strong>📊 Instruções:</strong><div>• Clique nos botões para adicionar</div><div>• Arraste blocos para mover</div><div>• Clique nas portas verdes para conectar</div><div>• Clique no bloco para selecionar</div></div>';
                salvarEstado();
            }
        }

        function salvarEstado() {
            window.parent.postMessage({
                type: "salvar_diagrama",
                blocos: blocos,
                conexoes: conexoes,
                contador: blocoIdCounter
            }, "*");
        }

        // Inicializar blocos existentes
        blocos.forEach(blocoData => {
            const container = document.getElementById("canvas-container");
            const bloco = document.createElement("div");
            bloco.className = "bloco";
            bloco.id = "bloco-" + blocoData.id;
            bloco.style.left = blocoData.x + "px";
            bloco.style.top = blocoData.y + "px";
            
            let nomeDisplay = blocoData.config.nome || blocoData.tipo;
            let tfDisplay = blocoData.config.tf || "";
            
            bloco.innerHTML = `
                <div class="bloco-tipo">${blocoData.tipo}</div>
                <div class="bloco-nome">${nomeDisplay}</div>
                ${tfDisplay ? '<div class="bloco-tf">' + tfDisplay + "</div>" : ""}
                <div class="porta porta-entrada" data-bloco="${blocoData.id}" data-tipo="entrada"></div>
                <div class="porta porta-saida" data-bloco="${blocoData.id}" data-tipo="saida"></div>
            `;
            
            container.appendChild(bloco);
            bloco.addEventListener("mousedown", iniciarArrastar);
            bloco.addEventListener("click", selecionarBloco);
            
            const portas = bloco.querySelectorAll(".porta");
            portas.forEach(porta => {
                porta.addEventListener("click", clickPorta);
            });
        });

        redesenharConexoes();
        
        // Adicionar evento de mensagem para comunicação com Streamlit
        window.addEventListener("message", function(event) {
            if (event.data.type === "load_diagram") {
                // Receber dados do Streamlit
                blocos = event.data.blocos;
                conexoes = event.data.conexoes;
                blocoIdCounter = event.data.contador;
                
                // Redesenhar
                document.querySelectorAll(".bloco").forEach(b => b.remove());
                redesenharConexoes();
            }
        });
    </script>
</body>
</html>'''
    
    # Substituir os marcadores de posição
    html_code = html_template.replace("BLOCOS_DATA", blocos_data)\
                             .replace("CONEXOES_DATA", conexoes_data)\
                             .replace("CONTADOR", str(contador))
    
    return html_code
