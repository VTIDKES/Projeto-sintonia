# -*- coding: utf-8 -*-
"""Janela movel de consulta para as notas de Sistemas de Controle."""

from pathlib import Path
import html
import json
import re

import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).parent
GUIDE_PATH = BASE_DIR / "notas_controle" / "guia_sintonia.md"
ORIGINALS_DIR = BASE_DIR / "notas_controle" / "originais"

NOTE_FILES = {
    "1. Sinais e sistemas": ORIGINALS_DIR / "01_sinais_e_sistemas.md",
    "2. Transformada de Laplace": ORIGINALS_DIR / "02_laplace.md",
    "3. Dinamica de 1 ordem": ORIGINALS_DIR / "03_dinamica_ordem1.md",
    "4. Dinamica de 2 ordem": ORIGINALS_DIR / "04_dinamica_ordem2.md",
    "5. Realimentacao": ORIGINALS_DIR / "05_realimentacao.md",
    "6. Estabilidade": ORIGINALS_DIR / "06_estabilidade.md",
    "7. Bode e frequencia": ORIGINALS_DIR / "07_frequencia_bode.md",
    "8. Nyquist": ORIGINALS_DIR / "08_nyquist.md",
    "9. Espaco de estados": ORIGINALS_DIR / "09_espaco_de_estados.md",
}


def _read_text(path):
    return path.read_text(encoding="utf-8")


def _load_guide_markdown():
    return _read_text(GUIDE_PATH)


def _inline_markdown(text):
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def _render_table(rows):
    parsed_rows = [
        [cell.strip() for cell in row.strip().strip("|").split("|")]
        for row in rows
    ]
    parsed_rows = [
        row for row in parsed_rows
        if not all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in row)
    ]
    if not parsed_rows:
        return ""

    header = parsed_rows[0]
    body = parsed_rows[1:]
    html_rows = ["<table><thead><tr>"]
    for cell in header:
        html_rows.append(f"<th>{_inline_markdown(cell)}</th>")
    html_rows.append("</tr></thead><tbody>")
    for row in body:
        html_rows.append("<tr>")
        for cell in row:
            html_rows.append(f"<td>{_inline_markdown(cell)}</td>")
        html_rows.append("</tr>")
    html_rows.append("</tbody></table>")
    return "".join(html_rows)


def _markdown_to_html(markdown):
    lines = markdown.splitlines()
    out = []
    i = 0
    in_code = False
    code_lines = []
    list_stack = []

    def close_lists():
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    def open_list(tag):
        if not list_stack or list_stack[-1] != tag:
            list_stack.append(tag)
            out.append(f"<{tag}>")

    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()

        if stripped.startswith("```"):
            close_lists()
            if in_code:
                out.append(
                    "<pre><code>"
                    + html.escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not stripped or stripped == "---":
            close_lists()
            i += 1
            continue

        if stripped.startswith("|"):
            close_lists()
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_rows.append(lines[i])
                i += 1
            out.append(_render_table(table_rows))
            continue

        if stripped.startswith("> [!"):
            close_lists()
            kind = stripped[4:].split("]", 1)[0].lower()
            body = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                body.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(
                f"<div class='admonition {html.escape(kind)}'>"
                f"<div class='admonition-title'>{html.escape(kind.title())}</div>"
                f"<div>{_inline_markdown(' '.join(body))}</div></div>"
            )
            continue

        if stripped.startswith(">"):
            close_lists()
            out.append(f"<blockquote>{_inline_markdown(stripped[1:].strip())}</blockquote>")
            i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            close_lists()
            formula = html.escape(stripped[2:-2].strip())
            out.append(f"<div class='math-block'>\\[{formula}\\]</div>")
            i += 1
            continue

        if stripped.startswith("$$"):
            close_lists()
            formula = [stripped[2:].strip()]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                formula.append(lines[i].strip())
                i += 1
            if i < len(lines):
                formula.append(lines[i].strip()[:-2].strip())
                i += 1
            formula_html = html.escape(" ".join(formula).strip())
            out.append(f"<div class='math-block'>\\[{formula_html}\\]</div>")
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            close_lists()
            level = min(len(heading_match.group(1)), 4)
            out.append(f"<h{level}>{_inline_markdown(heading_match.group(2))}</h{level}>")
            i += 1
            continue

        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ordered_match:
            open_list("ol")
            out.append(f"<li>{_inline_markdown(ordered_match.group(1))}</li>")
            i += 1
            continue

        if stripped.startswith("- "):
            open_list("ul")
            out.append(f"<li>{_inline_markdown(stripped[2:])}</li>")
            i += 1
            continue

        close_lists()
        out.append(f"<p>{_inline_markdown(stripped)}</p>")
        i += 1

    close_lists()
    if in_code:
        out.append(
            "<pre><code>"
            + html.escape("\n".join(code_lines))
            + "</code></pre>"
        )
    return "\n".join(out)


def _load_note_sections():
    return {
        label: _markdown_to_html(_read_text(path))
        for label, path in NOTE_FILES.items()
    }


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
        </style>
        <div class="guide-hero">
            <h1>Guia de Estudos</h1>
            <p>Notas de consulta para usar junto com o Sintonia.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guia_janela(label="Guia"):
    """Renderiza uma janela flutuante e arrastavel no documento principal."""
    sections = _load_note_sections()
    payload = json.dumps(sections, ensure_ascii=False)
    topics_payload = json.dumps(list(sections.keys()), ensure_ascii=False)
    label_payload = json.dumps(label, ensure_ascii=False)

    components.html(
        f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head><meta charset="UTF-8"></head>
        <body>
        <script>
        (function() {{
          const sections = {payload};
          const topics = {topics_payload};
          const buttonLabel = {label_payload};
          const doc = window.parent && window.parent.document ? window.parent.document : document;
          const old = doc.getElementById("sintonia-guide-root");
          if (old) old.remove();

          const root = doc.createElement("div");
          root.id = "sintonia-guide-root";
          root.innerHTML = `
            <style>
            #sintonia-guide-root * {{ box-sizing: border-box; }}
            #sintonia-guide-tab {{
              position: fixed;
              right: 0;
              top: 50%;
              transform: translateY(-50%);
              z-index: 2147483000;
              border: 1px solid #333654;
              border-right: 0;
              background: #171a2b;
              color: #fbbf24;
              border-radius: 12px 0 0 12px;
              padding: 14px 10px;
              font: 800 15px system-ui, sans-serif;
              cursor: pointer;
              box-shadow: 0 12px 28px rgba(0,0,0,.28);
              writing-mode: vertical-rl;
              letter-spacing: .5px;
            }}
            #sintonia-guide-window {{
              position: fixed;
              right: 18px;
              top: 80px;
              width: min(760px, calc(100vw - 34px));
              height: min(620px, calc(100vh - 120px));
              display: none;
              flex-direction: column;
              overflow: hidden;
              z-index: 2147483001;
              border: 1px solid #333654;
              border-radius: 14px;
              background: #111422;
              box-shadow: 0 24px 70px rgba(0,0,0,.38);
              color: #e0e4f0;
              font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            #sintonia-guide-window.open {{ display: flex; }}
            .sg-titlebar {{
              display: flex;
              align-items: center;
              gap: 10px;
              padding: 10px 12px;
              background: linear-gradient(135deg, #1a1d2e, #252840);
              border-bottom: 1px solid #333654;
              cursor: move;
              user-select: none;
            }}
            .sg-title {{
              color: #fbbf24;
              font-weight: 850;
              white-space: nowrap;
              font-size: 16px;
            }}
            .sg-titlebar select {{
              flex: 1;
              min-width: 180px;
              background: #252840;
              color: #e0e4f0;
              border: 1px solid #3d4267;
              border-radius: 8px;
              padding: 9px 10px;
              font-size: 14px;
              outline: none;
              cursor: pointer;
            }}
            .sg-close {{
              width: 36px;
              height: 36px;
              border-radius: 9px;
              border: 1px solid #8a3050;
              background: #3a1520;
              color: #ff8fa3;
              font-size: 20px;
              cursor: pointer;
              line-height: 1;
            }}
            .sg-body {{
              padding: 18px 22px 24px;
              overflow: auto;
              line-height: 1.65;
              font-size: 15px;
            }}
            .sg-body h1 {{
              margin: 0 0 16px;
              color: #f8fafc;
              font-size: 25px;
              line-height: 1.22;
            }}
            .sg-body h2 {{
              margin: 24px 0 10px;
              color: #34d399;
              font-size: 20px;
              line-height: 1.3;
            }}
            .sg-body h3 {{
              margin: 18px 0 8px;
              color: #60a5fa;
              font-size: 16px;
            }}
            .sg-body h4 {{
              margin: 16px 0 8px;
              color: #a78bfa;
              font-size: 15px;
            }}
            .sg-body p {{
              color: #d8deef;
              margin: 0 0 12px;
            }}
            .sg-body strong {{ color: #ffffff; font-weight: 800; }}
            .sg-body em {{ color: #c4b5fd; }}
            .sg-body code {{
              background: #252840;
              border: 1px solid #333654;
              border-radius: 5px;
              color: #93c5fd;
              padding: 1px 5px;
              font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
              font-size: .92em;
            }}
            .sg-body pre {{
              background: #0e1117;
              border: 1px solid #333654;
              border-radius: 10px;
              padding: 12px 14px;
              overflow: auto;
              color: #dbeafe;
            }}
            .sg-body pre code {{
              background: transparent;
              border: 0;
              padding: 0;
              color: inherit;
            }}
            .sg-body ul, .sg-body ol {{
              margin: 0 0 14px 22px;
              padding: 0;
              color: #d8deef;
            }}
            .sg-body li {{ margin: 5px 0; }}
            .sg-body table {{
              width: 100%;
              border-collapse: collapse;
              margin: 12px 0 18px;
              font-size: 14px;
              overflow: hidden;
              border-radius: 9px;
            }}
            .sg-body th, .sg-body td {{
              border: 1px solid #333654;
              padding: 8px 10px;
              text-align: left;
              vertical-align: top;
            }}
            .sg-body th {{
              background: #252840;
              color: #fbbf24;
            }}
            .sg-body td {{
              background: rgba(37,40,64,.42);
              color: #d8deef;
            }}
            .math-block {{
              margin: 12px 0 18px;
              padding: 12px 14px;
              border: 1px solid #333654;
              border-radius: 10px;
              background: #0e1117;
              color: #bfdbfe;
              font-family: Cambria Math, STIX Two Math, ui-monospace, monospace;
              overflow-x: auto;
              white-space: pre-wrap;
            }}
            .admonition {{
              margin: 12px 0 16px;
              padding: 12px 14px;
              border-left: 4px solid #fbbf24;
              border-radius: 8px;
              background: rgba(251,191,36,.08);
              color: #fde68a;
            }}
            .admonition-title {{
              font-weight: 800;
              margin-bottom: 4px;
              color: #fbbf24;
            }}
            @media (max-width: 760px) {{
              #sintonia-guide-window {{
                left: 8px !important;
                right: auto;
                top: 70px !important;
                bottom: auto;
                width: calc(100vw - 16px);
                height: calc(100vh - 120px);
              }}
              .sg-titlebar {{ flex-wrap: wrap; }}
              .sg-titlebar select {{ min-width: 100%; order: 3; }}
            }}
            </style>
            <button id="sintonia-guide-tab"></button>
            <section id="sintonia-guide-window" aria-label="Guia de estudos">
              <div class="sg-titlebar" id="sintonia-guide-titlebar">
                <span class="sg-title">Guia de Estudos</span>
                <select id="sintonia-guide-select"></select>
                <button class="sg-close" id="sintonia-guide-close" title="Fechar">&times;</button>
              </div>
              <div class="sg-body" id="sintonia-guide-body"></div>
            </section>
          `;
          doc.body.appendChild(root);

          if (!doc.getElementById("sintonia-mathjax-script")) {{
            const cfg = doc.createElement("script");
            cfg.textContent = "window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\\\\\\\(', '\\\\\\\\)']]}}, svg: {{fontCache: 'global'}}}};";
            doc.head.appendChild(cfg);
            const mj = doc.createElement("script");
            mj.id = "sintonia-mathjax-script";
            mj.async = true;
            mj.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
            doc.head.appendChild(mj);
          }}

          const tab = doc.getElementById("sintonia-guide-tab");
          const win = doc.getElementById("sintonia-guide-window");
          const bar = doc.getElementById("sintonia-guide-titlebar");
          const closeBtn = doc.getElementById("sintonia-guide-close");
          const select = doc.getElementById("sintonia-guide-select");
          const body = doc.getElementById("sintonia-guide-body");
          tab.textContent = buttonLabel;

          topics.forEach((topic) => {{
            const opt = doc.createElement("option");
            opt.value = topic;
            opt.textContent = topic;
            select.appendChild(opt);
          }});

          function renderTopic(topic) {{
            body.innerHTML = sections[topic] || "";
            body.scrollTop = 0;
            const parentWindow = doc.defaultView || window;
            if (parentWindow.MathJax && parentWindow.MathJax.typesetPromise) {{
              parentWindow.MathJax.typesetPromise([body]).catch(() => {{}});
            }}
          }}
          renderTopic(topics[0]);

          tab.addEventListener("click", () => win.classList.add("open"));
          closeBtn.addEventListener("click", () => win.classList.remove("open"));
          select.addEventListener("change", () => renderTopic(select.value));

          let dragging = false;
          let startX = 0, startY = 0, startLeft = 0, startTop = 0;
          bar.addEventListener("pointerdown", (ev) => {{
            if (ev.target === select || ev.target === closeBtn) return;
            dragging = true;
            const rect = win.getBoundingClientRect();
            startX = ev.clientX;
            startY = ev.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            win.style.left = rect.left + "px";
            win.style.top = rect.top + "px";
            win.style.right = "auto";
            win.style.bottom = "auto";
            bar.setPointerCapture(ev.pointerId);
          }});
          bar.addEventListener("pointermove", (ev) => {{
            if (!dragging) return;
            const maxLeft = Math.max(0, doc.defaultView.innerWidth - win.offsetWidth);
            const maxTop = Math.max(0, doc.defaultView.innerHeight - win.offsetHeight);
            const left = Math.min(maxLeft, Math.max(0, startLeft + ev.clientX - startX));
            const top = Math.min(maxTop, Math.max(0, startTop + ev.clientY - startY));
            win.style.left = left + "px";
            win.style.top = top + "px";
          }});
          bar.addEventListener("pointerup", () => dragging = false);
          bar.addEventListener("pointercancel", () => dragging = false);
        }})();
        </script>
        </body>
        </html>
        """,
        height=1,
        scrolling=False,
    )


def modo_guia_estudos():
    with st.sidebar:
        st.header("Navegação")
        if st.button("Voltar à Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Tópicos")
        topic = st.radio(
            "Escolha uma nota",
            list(NOTE_FILES.keys()),
            label_visibility="collapsed",
        )

    _render_topic_cards()
    st.markdown(_markdown_to_html(_read_text(NOTE_FILES[topic])), unsafe_allow_html=True)
