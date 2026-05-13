# -*- coding: utf-8 -*-
"""Modo de consulta didatica para teoria de sistemas de controle."""

from pathlib import Path
import html
import json

import streamlit as st
import streamlit.components.v1 as components


GUIDE_PATH = Path(__file__).parent / "notas_controle" / "guia_sintonia.md"


TOPICS = {
    "Visao geral": "Como Usar Este Guia",
    "Sinais e sistemas": "Sinais e Sistemas",
    "Laplace": "Transformada de Laplace",
    "Funcao de transferencia": "Funcao de Transferencia",
    "Diagrama de blocos": "Diagramas de Blocos",
    "1 ordem": "Sistemas de 1 Ordem",
    "2 ordem": "Sistemas de 2 Ordem",
    "Realimentacao": "Realimentacao e Erro Estacionario",
    "Estabilidade": "Estabilidade",
    "Bode": "Bode e Resposta em Frequencia",
    "Nyquist": "Nyquist",
    "LGR": "Lugar Geometrico das Raizes",
    "Espaco de estados": "Espaco de Estados",
    "Checklist": "Checklist de Analise no App",
    "Exemplos": "Exemplos Rapidos",
}


def _load_guide_markdown():
    return GUIDE_PATH.read_text(encoding="utf-8")


def _extract_section(markdown, heading):
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start < 0:
        return markdown

    next_heading = markdown.find("\n## ", start + len(marker))
    if next_heading < 0:
        return markdown[start:].strip()
    return markdown[start:next_heading].strip()


def _render_topic_cards():
    st.markdown(
        """
        <style>
        .guide-hero {
            background: linear-gradient(135deg, #12172a, #1b243a);
            border: 1px solid #303858;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 18px;
        }
        .guide-hero h1 {
            color: #e0e4f0;
            font-size: 30px;
            margin: 0 0 8px;
        }
        .guide-hero p {
            color: #a8b0d0;
            margin: 0;
            line-height: 1.6;
        }
        .guide-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 22px;
        }
        .guide-chip {
            border: 1px solid #333654;
            background: #1a1d2e;
            border-radius: 999px;
            color: #cbd5e1;
            font-size: 12px;
            padding: 6px 10px;
        }
        </style>
        <div class="guide-hero">
            <h1>Guia de Estudos</h1>
            <p>Notas de consulta para usar junto com o Sintonia: conceitos,
            formulas, interpretacao dos graficos e exemplos prontos para testar.</p>
        </div>
        <div class="guide-chip-row">
            <span class="guide-chip">Laplace</span>
            <span class="guide-chip">Polos e zeros</span>
            <span class="guide-chip">Bode</span>
            <span class="guide-chip">Nyquist</span>
            <span class="guide-chip">Realimentacao</span>
            <span class="guide-chip">Espaco de estados</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _markdown_to_simple_html(markdown):
    lines = markdown.splitlines()
    out = []
    in_list = False
    in_code = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            close_list()
            if in_code:
                out.append("</code></pre>")
            else:
                out.append("<pre><code>")
            in_code = not in_code
            continue

        if in_code:
            out.append(html.escape(line))
            continue

        if not stripped:
            close_list()
            continue

        if stripped.startswith("# "):
            close_list()
            out.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_list()
            out.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            close_list()
            out.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped.startswith("|"):
            close_list()
            out.append(f"<p class='guide-table-line'>{html.escape(stripped)}</p>")
        else:
            close_list()
            out.append(f"<p>{html.escape(stripped)}</p>")

    close_list()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def render_guia_janela(label="Guia"):
    """Render a movable guide window inside a Streamlit HTML component."""
    markdown = _load_guide_markdown()
    sections = {
        topic: _markdown_to_simple_html(_extract_section(markdown, heading))
        for topic, heading in TOPICS.items()
    }
    payload = json.dumps(sections, ensure_ascii=False)
    first_topic = next(iter(TOPICS.keys()))

    components.html(
        f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
        <meta charset="UTF-8">
        <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 520px;
            background: transparent;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #e0e4f0;
            overflow: hidden;
        }}
        .guide-tab {{
            position: absolute;
            left: 12px;
            top: 12px;
            z-index: 2;
            border: 1px solid #333654;
            background: #1a1d2e;
            color: #fbbf24;
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 8px 22px rgba(0,0,0,.28);
        }}
        .guide-window {{
            position: absolute;
            left: 16px;
            top: 58px;
            width: min(560px, calc(100% - 32px));
            height: 430px;
            display: none;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid #333654;
            border-radius: 14px;
            background: #111422;
            box-shadow: 0 20px 60px rgba(0,0,0,.45);
            z-index: 5;
        }}
        .guide-window.open {{ display: flex; }}
        .guide-titlebar {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: linear-gradient(135deg, #1a1d2e, #252840);
            border-bottom: 1px solid #333654;
            cursor: move;
            user-select: none;
        }}
        .guide-title {{
            font-weight: 800;
            color: #fbbf24;
            white-space: nowrap;
        }}
        .guide-titlebar select {{
            flex: 1;
            min-width: 0;
            border: 1px solid #333654;
            border-radius: 8px;
            background: #252840;
            color: #e0e4f0;
            padding: 8px 10px;
            outline: none;
        }}
        .guide-close {{
            border: 1px solid #8a3050;
            background: #3a1520;
            color: #ff8fa3;
            border-radius: 8px;
            width: 34px;
            height: 34px;
            cursor: pointer;
            font-size: 18px;
            line-height: 1;
        }}
        .guide-body {{
            padding: 16px 18px 22px;
            overflow: auto;
            line-height: 1.62;
            font-size: 14px;
        }}
        .guide-body h1 {{
            font-size: 22px;
            margin: 0 0 12px;
            color: #e0e4f0;
        }}
        .guide-body h2 {{
            font-size: 18px;
            margin: 0 0 12px;
            color: #34d399;
        }}
        .guide-body h3 {{
            font-size: 15px;
            margin: 16px 0 8px;
            color: #60a5fa;
        }}
        .guide-body p {{ margin: 0 0 10px; color: #cbd5e1; }}
        .guide-body ul {{ margin: 0 0 12px 20px; padding: 0; color: #cbd5e1; }}
        .guide-body li {{ margin: 4px 0; }}
        .guide-body pre {{
            background: #0e1117;
            border: 1px solid #333654;
            border-radius: 8px;
            padding: 10px 12px;
            overflow: auto;
        }}
        .guide-table-line {{
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 12px;
            color: #a8b0d0 !important;
            white-space: pre-wrap;
        }}
        </style>
        </head>
        <body>
        <button class="guide-tab" id="guideTab">{html.escape(label)}</button>
        <section class="guide-window" id="guideWindow">
            <div class="guide-titlebar" id="guideTitlebar">
                <span class="guide-title">Guia de Estudos</span>
                <select id="guideSelect"></select>
                <button class="guide-close" id="guideClose" title="Fechar">&times;</button>
            </div>
            <div class="guide-body" id="guideBody"></div>
        </section>
        <script>
        const sections = {payload};
        const topics = Object.keys(sections);
        const win = document.getElementById("guideWindow");
        const tab = document.getElementById("guideTab");
        const bar = document.getElementById("guideTitlebar");
        const closeBtn = document.getElementById("guideClose");
        const select = document.getElementById("guideSelect");
        const body = document.getElementById("guideBody");

        topics.forEach((topic) => {{
            const opt = document.createElement("option");
            opt.value = topic;
            opt.textContent = topic;
            select.appendChild(opt);
        }});

        function renderTopic(topic) {{
            body.innerHTML = sections[topic] || "";
            body.scrollTop = 0;
        }}
        renderTopic({json.dumps(first_topic)});

        tab.addEventListener("click", () => win.classList.add("open"));
        closeBtn.addEventListener("click", () => win.classList.remove("open"));
        select.addEventListener("change", () => renderTopic(select.value));

        let dragging = false, startX = 0, startY = 0, startLeft = 0, startTop = 0;
        bar.addEventListener("pointerdown", (ev) => {{
            if (ev.target === select || ev.target === closeBtn) return;
            dragging = true;
            startX = ev.clientX;
            startY = ev.clientY;
            startLeft = win.offsetLeft;
            startTop = win.offsetTop;
            bar.setPointerCapture(ev.pointerId);
        }});
        bar.addEventListener("pointermove", (ev) => {{
            if (!dragging) return;
            const maxLeft = Math.max(0, document.body.clientWidth - win.offsetWidth);
            const maxTop = Math.max(0, document.body.clientHeight - win.offsetHeight);
            const nextLeft = Math.min(maxLeft, Math.max(0, startLeft + ev.clientX - startX));
            const nextTop = Math.min(maxTop, Math.max(0, startTop + ev.clientY - startY));
            win.style.left = nextLeft + "px";
            win.style.top = nextTop + "px";
        }});
        bar.addEventListener("pointerup", () => dragging = false);
        bar.addEventListener("pointercancel", () => dragging = false);
        </script>
        </body>
        </html>
        """,
        height=540,
        scrolling=False,
    )


def modo_guia_estudos():
    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Topicos")
        topic = st.radio(
            "Escolha uma nota",
            list(TOPICS.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("Use o guia como apoio enquanto monta ou interpreta um sistema no app.")

    _render_topic_cards()

    markdown = _load_guide_markdown()
    section = _extract_section(markdown, TOPICS[topic])
    st.markdown(section)

    with st.expander("Ver guia completo"):
        st.markdown(markdown)
