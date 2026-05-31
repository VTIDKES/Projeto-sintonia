# -*- coding: utf-8 -*-
"""Modo Diagrama de Blocos do app Sintonia."""

from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components
from modo_guia_estudos import render_guia_janela


VISUAL_EDITOR_FIX_MARKER = "themeFixBadge"
VISUAL_EDITOR_FIX_VERSION = "fix 20260531"


def _patch_visual_editor_html(html: str) -> str:
    """Aplica o patch de contraste mesmo se o HTML publicado estiver antigo."""
    if VISUAL_EDITOR_FIX_MARKER in html:
        return html

    css_fix = """
/* STREAMLIT_RUNTIME_WHITE_THEME_FIX_20260531 */
:root[data-theme="light"] body,
:root[data-theme="light"] .app,
:root[data-theme="light"] .toolbar,
:root[data-theme="light"] .panel,
:root[data-theme="light"] .manual-sec,
:root[data-theme="light"] .results,
:root[data-theme="light"] .rcard,
:root[data-theme="light"] .modal{color:#0f172a!important}
:root[data-theme="light"] .toolbar .lbl,
:root[data-theme="light"] .panel-section h4,
:root[data-theme="light"] .pg label,
:root[data-theme="light"] .hint,
:root[data-theme="light"] .man-hint,
:root[data-theme="light"] .mbox .ml{color:#0f172a!important;opacity:1!important}
:root[data-theme="light"] .block,
:root[data-theme="light"] .block *{opacity:1!important}
:root[data-theme="light"] .block-header{
  color:#ffffff!important;
  font-weight:900!important;
  opacity:1!important;
  text-shadow:0 1px 2px #000,0 0 5px rgba(0,0,0,.95)!important;
}
:root[data-theme="light"] .block-body,
:root[data-theme="light"] .block-body *,
:root[data-theme="light"] .block-tf-disp,
:root[data-theme="light"] .block-tf-disp *{
  color:#ffffff!important;
  opacity:1!important;
  text-shadow:0 1px 2px #000,0 0 5px rgba(0,0,0,.85)!important;
}
:root[data-theme="light"] #themeSelect,
:root[data-theme="light"] select,
:root[data-theme="light"] input,
:root[data-theme="light"] .tb,
:root[data-theme="light"] .pg input,
:root[data-theme="light"] .cfg-row input{
  color:#0f172a!important;
  background:#f8fafc!important;
  border-color:#64748b!important;
  opacity:1!important;
}
:root[data-theme="light"] .tf-disp,
:root[data-theme="light"] .tf-disp *,
:root[data-theme="light"] .mgrid,
:root[data-theme="light"] .mbox,
:root[data-theme="light"] .mv{
  color:#0f172a!important;
  opacity:1!important;
  text-shadow:none!important;
}
"""

    js_fix = f"""
<script>
(function(){{
  function setImp(el,prop,val){{el.style.setProperty(prop,val,"important")}}
  function clearImp(el,props){{props.forEach(function(prop){{el.style.removeProperty(prop)}})}}
  window.forceThemeContrast=function(){{
    var light=document.documentElement.getAttribute("data-theme")==="light";
    var select=document.getElementById("themeSelect");
    if(select){{
      var opt=select.querySelector('option[value="light"]');
      if(opt)opt.textContent="Branco alto contraste";
      if(!document.getElementById("{VISUAL_EDITOR_FIX_MARKER}")){{
        var badge=document.createElement("span");
        badge.className="lbl";
        badge.id="{VISUAL_EDITOR_FIX_MARKER}";
        badge.textContent="{VISUAL_EDITOR_FIX_VERSION}";
        badge.style.cssText="font-size:10px;font-weight:800;color:#0f172a;opacity:1";
        select.insertAdjacentElement("afterend",badge);
      }}
    }}
    var badge=document.getElementById("{VISUAL_EDITOR_FIX_MARKER}");
    if(badge){{badge.textContent="{VISUAL_EDITOR_FIX_VERSION}";badge.style.opacity="1";badge.style.color=light?"#0f172a":"#c9d2e6"}}
    document.querySelectorAll("#themeSelect,select,input,.tb:not(button),.pg input,.cfg-row input").forEach(function(el){{
      if(light){{setImp(el,"color","#0f172a");setImp(el,"background","#f8fafc");setImp(el,"border-color","#64748b");setImp(el,"opacity","1")}}
      else clearImp(el,["color","background","border-color","opacity"])
    }});
    document.querySelectorAll(".toolbar .lbl,.panel-section h4,.pg label,.hint,.man-hint,.mbox .ml,.rcard h4,.rcard label").forEach(function(el){{
      if(light){{setImp(el,"color","#0f172a");setImp(el,"opacity","1");setImp(el,"text-shadow","none")}}
      else clearImp(el,["color","opacity","text-shadow"])
    }});
    document.querySelectorAll(".block,.block *").forEach(function(el){{
      if(light)setImp(el,"opacity","1");
      else clearImp(el,["opacity"])
    }});
    document.querySelectorAll(".block-header").forEach(function(el){{
      if(light){{setImp(el,"color","#ffffff");setImp(el,"font-weight","900");setImp(el,"opacity","1");setImp(el,"text-shadow","0 1px 2px #000,0 0 5px rgba(0,0,0,.95)")}}
      else clearImp(el,["color","font-weight","opacity","text-shadow"])
    }});
    document.querySelectorAll(".block-body,.block-body *,.block-tf-disp,.block-tf-disp *").forEach(function(el){{
      if(light){{setImp(el,"color","#ffffff");setImp(el,"opacity","1");setImp(el,"text-shadow","0 1px 2px #000,0 0 5px rgba(0,0,0,.85)")}}
      else clearImp(el,["color","opacity","text-shadow"])
    }});
  }};
  var oldApply=window.applyTheme;
  if(typeof oldApply==="function"){{
    window.applyTheme=function(theme){{oldApply(theme);window.forceThemeContrast();setTimeout(window.forceThemeContrast,80)}};
  }}
  var oldRender=window.render;
  if(typeof oldRender==="function"){{
    window.render=function(){{var r=oldRender.apply(this,arguments);window.forceThemeContrast();return r}};
  }}
  document.addEventListener("DOMContentLoaded",function(){{window.forceThemeContrast();setTimeout(window.forceThemeContrast,80)}});
  setTimeout(window.forceThemeContrast,0);
  setTimeout(window.forceThemeContrast,300);
}})();
</script>
"""

    html = html.replace(
        '<option value="light">Branco</option>',
        '<option value="light">Branco alto contraste</option>',
        1,
    )
    html = html.replace(
        "</select>\n</div>",
        f'</select>\n<span class="lbl" id="{VISUAL_EDITOR_FIX_MARKER}" style="font-size:10px;font-weight:800">{VISUAL_EDITOR_FIX_VERSION}</span>\n</div>',
        1,
    )
    html = html.replace("</style>", css_fix + "\n</style>", 1)
    html = html.replace("</body>", js_fix + "\n</body>", 1)
    return html


def _load_visual_editor_html():
    html_path = Path(__file__).parent / "visual_blocks_frontend" / "index.html"
    return _patch_visual_editor_html(html_path.read_text(encoding="utf-8"))


def _visual_editor_fix_status() -> str:
    html_path = Path(__file__).parent / "visual_blocks_frontend" / "index.html"
    try:
        html = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "HTML do editor visual nao encontrado."
    if VISUAL_EDITOR_FIX_MARKER in html:
        return f"{VISUAL_EDITOR_FIX_VERSION} no arquivo publicado"
    return f"{VISUAL_EDITOR_FIX_VERSION} aplicado em tempo de execucao"


# ══════════════════════════════════════════════════
# MODO CANVAS
# ══════════════════════════════════════════════════


def _coeffs_to_poly_str(coeffs):
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_val = float(c)
        if abs(c_val) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c_val:g}")
        elif power == 1:
            if abs(c_val - 1) < 1e-10:
                terms.append("s")
            elif abs(c_val + 1) < 1e-10:
                terms.append("-s")
            else:
                terms.append(f"{c_val:g}*s")
        else:
            if abs(c_val - 1) < 1e-10:
                terms.append(f"s^{power}")
            elif abs(c_val + 1) < 1e-10:
                terms.append(f"-s^{power}")
            else:
                terms.append(f"{c_val:g}*s^{power}")
    if not terms:
        return "0"
    result = terms[0]
    for t in terms[1:]:
        result += t if t.startswith('-') else '+' + t
    return result


def _build_canvas_model(blocos_df, conexoes):
    tipo_map = {
        'Planta': 'tf', 'Controlador': 'tf', 'Sensor': 'sensor',
        'Atuador': 'actuator', 'Pre-filtro': 'tf', 'Perturbação': 'tf',
    }
    nodes = []
    edges = []
    nid = 0

    def mk_id():
        nonlocal nid
        nid += 1
        return f"n{nid}"

    if not conexoes:
        inp_id = mk_id()
        nodes.append({'id': inp_id, 'type': 'input', 'x': 30, 'y': 200, 'params': {'label': 'R(s)'}})
        for i, (_, row) in enumerate(blocos_df.iterrows()):
            tf_obj = row['tf']
            bt = tipo_map.get(row['tipo'], 'tf')
            b_id = mk_id()
            nodes.append({
                'id': b_id, 'type': bt,
                'x': 200 + i * 180, 'y': 180,
                'params': {'num': _coeffs_to_poly_str(tf_obj.num[0][0]),
                           'den': _coeffs_to_poly_str(tf_obj.den[0][0])}
            })
        out_id = mk_id()
        nodes.append({'id': out_id, 'type': 'output',
                      'x': 200 + len(blocos_df) * 180, 'y': 200,
                      'params': {'label': 'Y(s)'}})
        return {'nodes': nodes, 'edges': edges}

    bloco_map = {}
    for _, row in blocos_df.iterrows():
        tf_obj = row['tf']
        bt = tipo_map.get(row['tipo'], 'tf')
        bloco_map[row['nome']] = {
            'type': bt,
            'params': {'num': _coeffs_to_poly_str(tf_obj.num[0][0]),
                       'den': _coeffs_to_poly_str(tf_obj.den[0][0])}
        }

    inp_id = mk_id()
    nodes.append({'id': inp_id, 'type': 'input', 'x': 30, 'y': 200, 'params': {'label': 'R(s)'}})

    prev_out_id = inp_id
    prev_out_port = 'out0'
    x_offset = 200

    for con in conexoes:
        tipo_con = con['tipo']
        nomes = [n for n in con['blocos'] if n in bloco_map]
        if len(nomes) < 2:
            continue

        if tipo_con == 'Série':
            for i, nome in enumerate(nomes):
                b = bloco_map[nome]
                b_id = mk_id()
                nodes.append({'id': b_id, 'type': b['type'],
                              'x': x_offset + i * 180, 'y': 180,
                              'params': dict(b['params'])})
                edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                              'srcPort': prev_out_port, 'dst': b_id, 'dstPort': 'in0'})
                prev_out_id = b_id
                prev_out_port = 'out0'
            x_offset += len(nomes) * 180

        elif tipo_con == 'Paralelo':
            nb = len(nomes)
            signs = ' '.join(['+'] * nb)
            sum_id = mk_id()
            nodes.append({'id': sum_id, 'type': 'sum',
                          'x': x_offset + 360, 'y': 210,
                          'params': {'signs': signs}})
            branch_ids = []
            cur_src_id = prev_out_id
            cur_src_port = prev_out_port
            for i in range(nb - 1):
                br_id = mk_id()
                nodes.append({'id': br_id, 'type': 'branch',
                              'x': x_offset + i * 30, 'y': 160 + i * 50,
                              'params': {}})
                edges.append({'id': f'e{len(edges)}', 'src': cur_src_id,
                              'srcPort': cur_src_port, 'dst': br_id, 'dstPort': 'in0'})
                branch_ids.append(br_id)
                cur_src_id = br_id
                cur_src_port = 'out0'
            for i, nome in enumerate(nomes):
                b = bloco_map[nome]
                b_id = mk_id()
                nodes.append({'id': b_id, 'type': b['type'],
                              'x': x_offset + 160, 'y': 100 + i * 120,
                              'params': dict(b['params'])})
                if i < nb - 1:
                    edges.append({'id': f'e{len(edges)}', 'src': branch_ids[i],
                                  'srcPort': 'out1', 'dst': b_id, 'dstPort': 'in0'})
                else:
                    edges.append({'id': f'e{len(edges)}', 'src': branch_ids[-1],
                                  'srcPort': 'out0', 'dst': b_id, 'dstPort': 'in0'})
                edges.append({'id': f'e{len(edges)}', 'src': b_id,
                              'srcPort': 'out0', 'dst': sum_id, 'dstPort': f'in{i}'})
            prev_out_id = sum_id
            prev_out_port = 'out0'
            x_offset += 520

        elif tipo_con.startswith('Realimentação'):
            is_pos = tipo_con == 'Realimentação Positiva'
            sign_str = '+ +' if is_pos else '+ -'
            sum_id = mk_id()
            nodes.append({'id': sum_id, 'type': 'sum',
                          'x': x_offset, 'y': 200,
                          'params': {'signs': sign_str}})
            edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                          'srcPort': prev_out_port, 'dst': sum_id, 'dstPort': 'in0'})
            g_data = bloco_map[nomes[0]]
            g_id = mk_id()
            nodes.append({'id': g_id, 'type': g_data['type'],
                          'x': x_offset + 160, 'y': 180,
                          'params': dict(g_data['params'])})
            edges.append({'id': f'e{len(edges)}', 'src': sum_id,
                          'srcPort': 'out0', 'dst': g_id, 'dstPort': 'in0'})
            branch_id = mk_id()
            nodes.append({'id': branch_id, 'type': 'branch',
                          'x': x_offset + 340, 'y': 210,
                          'params': {}})
            edges.append({'id': f'e{len(edges)}', 'src': g_id,
                          'srcPort': 'out0', 'dst': branch_id, 'dstPort': 'in0'})
            if len(nomes) > 1:
                h_data = bloco_map[nomes[1]]
                h_id = mk_id()
                nodes.append({'id': h_id, 'type': h_data['type'],
                              'x': x_offset + 200, 'y': 340,
                              'params': dict(h_data['params'])})
                edges.append({'id': f'e{len(edges)}', 'src': branch_id,
                              'srcPort': 'out1', 'dst': h_id, 'dstPort': 'in0'})
                edges.append({'id': f'e{len(edges)}', 'src': h_id,
                              'srcPort': 'out0', 'dst': sum_id, 'dstPort': 'in1'})
            prev_out_id = branch_id
            prev_out_port = 'out0'
            x_offset += 420

    out_id = mk_id()
    nodes.append({'id': out_id, 'type': 'output',
                  'x': x_offset, 'y': 200, 'params': {'label': 'Y(s)'}})
    edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                  'srcPort': prev_out_port, 'dst': out_id, 'dstPort': 'in0'})

    return {'nodes': nodes, 'edges': edges}


def modo_canvas():
    with st.sidebar:
        st.header("Navegação")
        if st.button("← Voltar à Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.markdown("### Como usar")
        st.markdown("""
        **Diagrama de Blocos:**
        1. Clique **+ Adicionar Bloco** ou use os botões rápidos
        2. Arraste para posicionar
        3. Conecte: porta azul (saída) → porta verde (entrada)
        4. Edite parâmetros no painel lateral
        5. Clique **CALCULAR** para ver resultados

        **Entrada Manual:**
        1. Clique em "Entrada Manual"
        2. Escolha: T(s) Direta, Malha Fechada, Malha Aberta ou Espaço de Estados
        3. Preencha os campos
        4. Clique **CALCULAR T(s)**
        """)

    render_guia_janela("Guia")
    st.caption(f"Editor visual: {_visual_editor_fix_status()}")

    html_content = _load_visual_editor_html()

    if not st.session_state.blocos.empty:
        canvas_model = _build_canvas_model(
            st.session_state.blocos, st.session_state.conexoes)
        html_content = html_content.replace(
            '__INITIAL_BLOCKS__', json.dumps(canvas_model))
    else:
        html_content = html_content.replace('__INITIAL_BLOCKS__', '[]')

    components.html(html_content, height=2800, scrolling=True)


# ══════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ══════════════════════════════════════════════════
