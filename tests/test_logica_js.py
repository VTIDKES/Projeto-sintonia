"""Logica do editor (JS inline) executada em Node com um DOM falso (harness.js).

Fora de alcance: renderizacao real (SVG, Plotly), arrastar/soltar, foco e
teclado reais, estilos. O DOM falso aceita qualquer chamada e devolve vazio.
"""
import json
import pytest

from cenarios import CENARIOS

POLOS_SNIPPET = """(function(){
  var o={};
  CENARIOS.forEach(function(c){
    var r=solve(c.modelo.nodes,c.modelo.edges);
    o[c.id]=r.e?{e:r.e}:{tf:r.tf,polos:roots(r.tf.d),zeros:roots(r.tf.n)};
  });
  return o})()"""


@pytest.fixture
def resolvidos(run_js):
    return run_js(POLOS_SNIPPET)


def _reais(ps):
    return sorted(p["r"] for p in ps)


def test_todos_os_cenarios_resolvem_sem_erro(resolvidos):
    assert set(resolvidos) == {c["id"] for c in CENARIOS}
    for id_, r in resolvidos.items():
        assert "e" not in r, (id_, r)
        assert r["tf"]["d"], id_


def test_primeira_ordem_tem_um_polo_real_em_menos_um(resolvidos):
    ps = resolvidos["primeira_ordem"]["polos"]
    assert len(ps) == 1 and ps[0]["r"] == pytest.approx(-1) and ps[0]["i"] == 0


def test_subamortecida_tem_par_complexo_estavel(resolvidos):
    ps = resolvidos["segunda_subamortecida"]["polos"]
    assert len(ps) == 2
    assert all(p["r"] == pytest.approx(-0.5) for p in ps)
    assert sorted(abs(p["i"]) for p in ps) == pytest.approx([(15 ** 0.5) / 2] * 2, abs=1e-6)


def test_criticamente_amortecido_tem_polo_duplo(resolvidos):
    assert _reais(resolvidos["segunda_critica"]["polos"]) == pytest.approx([-2, -2])


def test_superamortecido_tem_dois_polos_reais_distintos(resolvidos):
    assert _reais(resolvidos["segunda_super"]["polos"]) == pytest.approx([-4, -1])


def test_pade_tem_zero_no_semiplano_direito(resolvidos):
    r = resolvidos["atraso_pade"]
    assert _reais(r["zeros"]) == pytest.approx([2])
    assert _reais(r["polos"]) == pytest.approx([-2, -1])


def test_malha_fechada_com_k_bate_com_a_ft_documentada(resolvidos):
    tf = resolvidos["malha_fechada_k"]["tf"]
    assert tf["d"] == pytest.approx([7, 3, 1])
    assert tf["n"] == pytest.approx([5])


def test_malha_fechada_pid_e_estavel(resolvidos):
    ps = resolvidos["malha_fechada_pid"]["polos"]
    assert len(ps) == 3 and all(p["r"] < 0 for p in ps)


def test_cenario_instavel_tem_polo_com_parte_real_positiva(resolvidos):
    ps = resolvidos["instavel"]["polos"]
    assert len(ps) == 1 and ps[0]["r"] == pytest.approx(1.5)


# ---- classificacao de estabilidade ----
@pytest.mark.parametrize("polos,esperado", [
    ("[{r:-1,i:0},{r:-2,i:3},{r:-2,i:-3}]", "Estável"),
    ("[]", "Estável"),
    ("[{r:0,i:0}]", "Marginal"),
    ("[{r:0,i:2},{r:0,i:-2}]", "Marginal"),
    ("[{r:0,i:0},{r:-3,i:0}]", "Marginal"),
    ("[{r:0.5,i:0}]", "Instável"),
    ("[{r:0,i:2},{r:0,i:-2},{r:1,i:0}]", "Instável"),
])
def test_classificacao_de_estabilidade(run_js, polos, esperado):
    assert run_js(f"classEst({polos}).st") == esperado


# ---- validadores ----
@pytest.mark.parametrize("texto", ["", "abc", "s^2+", "1+x", "   ", "s^-2+1", "3,5s+1", "s+", "s*"])
def test_poly_invalido_rejeita(run_js, texto):
    assert run_js(f"polyInvalido({json.dumps(texto)})") is True


@pytest.mark.parametrize("texto", ["s^2+2s+1", "4", "s+1", "0", "2 s ^ 2 + 3 s + 1", "-s", "2*s+1", "0.5s+1"])
def test_poly_invalido_aceita(run_js, texto):
    assert run_js(f"polyInvalido({json.dumps(texto)})") is False


def test_parse_poly_ignora_espacos(run_js):
    assert run_js('parseP("2 s^2 + 3 s + 1")') == [1, 3, 2]


@pytest.mark.parametrize("a,b,c,d,invalido", [
    ("0 1; -2 -3", "0;1", "1 0", "0", False),
    ("1 2 3; 4 5 6", "0;1", "1 0", "0", True),        # A nao quadrada
    ("0 1; -2 -3", "0;1", "1 0 0", "0", True),         # C com colunas demais
    ("0 1; -2 -3", "0;1;2", "1 0", "0", True),         # B com linhas demais
    ("0 1; -2 -3", "0;1", "1 0", "0 0", True),         # D nao 1x1
    ("0 1; -2 x", "0;1", "1 0", "0", True),
    ("", "0;1", "1 0", "0", True),
    ("0 1; -2", "0;1", "1 0", "0", True),              # linhas irregulares
])
def test_ss_invalido(run_js, a, b, c, d, invalido):
    args = ",".join(json.dumps(x) for x in (a, b, c, d))
    assert run_js(f"ssInvalido({args})") is invalido


# ---- manualTF (entrada manual) ----
def _manual(run_js, campos, sub="direct"):
    sets = ";".join(f'document.getElementById("{k}").value={json.dumps(v)}' for k, v in campos.items())
    return run_js(f"""(function(){{curSubMode={json.dumps(sub)};{sets};
      var tf=manualTF();return{{tf:tf,erro:document.getElementById("rb").innerHTML}}}})()""")


@pytest.mark.parametrize("den", ["0", "0s", "0.0", "", "  ", "0s^2+0"])
def test_manual_recusa_denominador_nulo_ou_vazio(run_js, den):
    r = _manual(run_js, {"manNum": "1", "manDen": den})
    assert r["tf"] is None and "Denominador" in r["erro"]


def test_manual_aceita_polinomio_com_espacos(run_js):
    r = _manual(run_js, {"manNum": "2 s + 4", "manDen": "s ^ 2 + 3 s + 2"})
    assert r["tf"]["d"] == pytest.approx([1, 1]) and r["tf"]["n"] == pytest.approx([2])  # cancela (s+2)


def test_manual_recusa_virgula_decimal(run_js):
    r = _manual(run_js, {"manNum": "1", "manDen": "s^2+3,5s+1"})
    assert r["tf"] is None and "inválido" in r["erro"]


def test_manual_recusa_expoente_negativo(run_js):
    r = _manual(run_js, {"manNum": "1", "manDen": "s^-2+1"})
    assert r["tf"] is None and "inválido" in r["erro"]


def test_manual_ss_nao_quadrada_mostra_erro_de_matriz(run_js):
    r = _manual(run_js, {"manA": "1 2 3; 4 5 6", "manB": "0;1", "manC": "1 0", "manD": "0"}, sub="ss")
    assert r["tf"] is None and "Matriz" in r["erro"]


def test_manual_ss_2x2_converte_para_ft(run_js):
    r = _manual(run_js, {"manA": "0 1; -2 -3", "manB": "0;1", "manC": "1 0", "manD": "0"}, sub="ss")
    assert r["tf"]["n"] == pytest.approx([1]) and r["tf"]["d"] == pytest.approx([2, 3, 1])


# ---- blocos soltos e casos de borda do solve ----
def _modelo(nos, arestas):
    return json.dumps({"nodes": nos, "edges": arestas})


NO_R = {"id": "a", "type": "input", "params": {}}
NO_Y = {"id": "c", "type": "output", "params": {}}
SOMA = {"id": "b", "type": "sum", "params": {"signs": "+ -"}}


def _aresta(i, s, d, porta="in0"):
    return {"id": i, "src": s, "srcPort": "out0", "dst": d, "dstPort": porta}


def _soltos(run_js, m):
    return run_js(f"(function(){{model={m};return blocosSoltos().map(function(n){{return n.id}})}})()")


def test_blocos_soltos_acusa_somador_com_porta_sem_aresta(run_js):
    m = _modelo([NO_R, SOMA, NO_Y], [_aresta("e1", "a", "b"), _aresta("e2", "b", "c")])
    assert _soltos(run_js, m) == ["b"]


def test_blocos_soltos_nao_acusa_malha_completa(run_js):
    c = next(c for c in CENARIOS if c["id"] == "malha_fechada_k")
    assert _soltos(run_js, json.dumps(c["modelo"])) == []


def test_blocos_soltos_acusa_bloco_sem_saida(run_js):
    extra = {"id": "z", "type": "gain", "params": {"k": "2"}}
    m = _modelo([NO_R, NO_Y, extra], [_aresta("e1", "a", "c"), _aresta("e2", "a", "z")])
    assert _soltos(run_js, m) == ["z"]


def test_blocos_soltos_diagrama_vazio(run_js):
    assert _soltos(run_js, _modelo([], [])) == []


def test_solve_diagrama_vazio_pede_blocos(run_js):
    assert run_js("solve([],[])") == {"e": "Adicione blocos."}


def test_solve_sem_arestas_pede_conexao(run_js):
    m = _modelo([NO_R, NO_Y], [])
    assert run_js(f"(function(){{var m={m};return solve(m.nodes,m.edges)}})()") == {"e": "Conecte os blocos."}


def test_somador_sem_entradas_nao_quebra_o_calculo(run_js):
    sem = {"id": "b", "type": "sum", "params": {"signs": ""}}
    m = _modelo([NO_R, sem, NO_Y], [_aresta("e1", "b", "c")])
    r = run_js(f"(function(){{var m={m};try{{return solve(m.nodes,m.edges)}}catch(e){{return{{exc:e.message}}}}}})()")
    assert "exc" not in r


# ---- desfazer / refazer ----
def test_desfazer_e_refazer_restauram_o_modelo(run_js):
    r = run_js("""(function(){
      render();
      carregaCenario(CENARIOS[0]);carregaCenario(CENARIOS[1]);
      var n0=model.nodes[1].params.den;
      document.getElementById("btnUndo").click();
      var apos_undo=model.nodes[1].params.den;
      document.getElementById("btnRedo").click();
      var apos_redo=model.nodes[1].params.den;
      return{n0:n0,apos_undo:apos_undo,apos_redo:apos_redo,redoVazio:_redo.length}})()""")
    assert r["n0"] == "s^2+s+4"
    assert r["apos_undo"] == "s+1"
    assert r["apos_redo"] == "s^2+s+4" and r["redoVazio"] == 0


def test_desfazer_ate_o_inicio_volta_ao_diagrama_vazio_e_desabilita_botao(run_js):
    r = run_js("""(function(){
      render();carregaCenario(CENARIOS[0]);
      document.getElementById("btnUndo").click();
      return{nos:model.nodes.length,undoDesab:document.getElementById("btnUndo").disabled,
             redoDesab:document.getElementById("btnRedo").disabled}})()""")
    assert r == {"nos": 0, "undoDesab": True, "redoDesab": False}


def test_nova_edicao_apos_desfazer_limpa_o_refazer(run_js):
    r = run_js("""(function(){
      render();carregaCenario(CENARIOS[0]);carregaCenario(CENARIOS[1]);
      histIr(_undo,_redo);var antes=_redo.length;
      carregaCenario(CENARIOS[2]);return{antes:antes,depois:_redo.length}})()""")
    assert r == {"antes": 1, "depois": 0}


def test_cenario_carregado_duas_vezes_seguido_de_desfazer_volta_ao_anterior(run_js):
    r = run_js("""(function(){
      render();carregaCenario(CENARIOS[0]);carregaCenario(CENARIOS[1]);
      carregaCenario(CENARIOS[1]);
      histIr(_undo,_redo);
      return{den:model.nodes[1].params.den}})()""")
    # a segunda carga e identica: nao cria passo de historico, desfazer volta ao cenario 0
    assert r["den"] == "s+1"


def test_limite_de_50_passos_de_desfazer(run_js):
    r = run_js("""(function(){
      render();
      for(var i=0;i<70;i++){model.nodes.push({id:"n"+i,type:"gain",x:0,y:0,params:{k:"1"}});render()}
      var tamanho=_undo.length,passos=0;
      while(_undo.length){histIr(_undo,_redo);passos++}
      return{tamanho:tamanho,passos:passos,nos:model.nodes.length,redo:_redo.length}})()""")
    assert r["tamanho"] == 50 and r["passos"] == 50
    assert r["nos"] == 70 - 50 and r["redo"] == 50


def test_limite_de_50_tambem_vale_para_refazer(run_js):
    r = run_js("""(function(){
      render();
      for(var i=0;i<70;i++){model.nodes.push({id:"n"+i,type:"gain",x:0,y:0,params:{k:"1"}});render()}
      while(_undo.length)histIr(_undo,_redo);
      while(_redo.length)histIr(_redo,_undo);
      return{nos:model.nodes.length,undo:_undo.length,redo:_redo.length}})()""")
    assert r == {"nos": 70, "undo": 50, "redo": 0}


# ---- bugs reais (corrigidos) ----
def test_ss_3x3_tem_polinomio_caracteristico_correto(run_js):
    tf = run_js('ssToTF("0 1 0; 0 0 1; -6 -11 -6","0;0;1","1 0 0","0")')
    assert tf["d"] == pytest.approx([6, 11, 6, 1], rel=1e-6)


def _solve_tf(run_js, den):
    nos = [NO_R, {"id": "b", "type": "tf", "params": {"num": "1", "den": den}}, NO_Y]
    m = _modelo(nos, [_aresta("e1", "a", "b"), _aresta("e2", "b", "c")])
    return run_js(f"(function(){{var m={m};return solve(m.nodes,m.edges)}})()")


def test_solve_recusa_polinomio_com_virgula_decimal_em_bloco_tf(run_js):
    assert "e" in _solve_tf(run_js, "s^2+3,5s+1")


def test_solve_recusa_expoente_negativo_em_bloco_tf(run_js):
    assert "e" in _solve_tf(run_js, "s^-2+1")


# ---- correcoes da revisao (A a H) ----
def _fechada(run_js, modelo, modo="diag", sub="direct"):
    return run_js(f"(function(){{model={json.dumps(modelo)};curMode={json.dumps(modo)};curSubMode={json.dumps(sub)};"
                  "return{laco:temLaco(),fechada:tfJaFechada()}})()")


def test_paralelo_com_somador_nao_e_malha_fechada(run_js):
    nos = [NO_R, {"id": "r", "type": "branch", "params": {}},
           {"id": "g1", "type": "tf", "params": {"num": "1", "den": "s+1"}},
           {"id": "g2", "type": "tf", "params": {"num": "1", "den": "s+2"}},
           {"id": "s", "type": "sum", "params": {"signs": "+ +"}}, NO_Y]
    ar = [_aresta("e1", "a", "r"), _aresta("e2", "r", "g1"),
          {"id": "e3", "src": "r", "srcPort": "out1", "dst": "g2", "dstPort": "in0"},
          _aresta("e4", "g1", "s"), _aresta("e5", "g2", "s", "in1"), _aresta("e6", "s", "c")]
    assert _fechada(run_js, {"nodes": nos, "edges": ar}) == {"laco": False, "fechada": False}


def test_laco_real_e_malha_fechada_no_diagrama_e_na_entrada_manual(run_js):
    c = next(c for c in CENARIOS if c["id"] == "malha_fechada_k")["modelo"]
    assert _fechada(run_js, c) == {"laco": True, "fechada": True}
    vazio = {"nodes": [], "edges": []}
    assert _fechada(run_js, c, "manual", "direct")["fechada"] is False  # diagrama salvo nao contamina
    assert _fechada(run_js, vazio, "manual", "closed")["fechada"] is True


@pytest.mark.parametrize("k,esperado", [("0", [0]), ("2", [2]), ("-3", [-3])])
def test_ganho_zero_e_valor_valido(run_js, k, esperado):
    nos = [NO_R, {"id": "b", "type": "gain", "params": {"k": k}}, NO_Y]
    m = _modelo(nos, [_aresta("e1", "a", "b"), _aresta("e2", "b", "c")])
    r = run_js(f"(function(){{var m={m};return solve(m.nodes,m.edges)}})()")
    assert r["tf"]["n"] == esperado


def test_ganho_nao_numerico_e_erro(run_js):
    nos = [NO_R, {"id": "b", "type": "gain", "params": {"k": "abc"}}, NO_Y]
    m = _modelo(nos, [_aresta("e1", "a", "b"), _aresta("e2", "b", "c")])
    r = run_js(f"(function(){{var m={m};return solve(m.nodes,m.edges)}})()")
    assert "e" in r and "b" in r["e"]


@pytest.mark.parametrize("den", ["(s+1)^2", "", "  ", "0", "s^2+3,5s+1", "s^-2+1"])
def test_bloco_tf_com_denominador_invalido_e_erro(run_js, den):
    assert "e" in _solve_tf(run_js, den)


def test_ss_4x4_companheira(run_js):
    # (s+1)(s+2)(s+3)(s+4) = s^4+10s^3+35s^2+50s+24
    tf = run_js('ssToTF("0 1 0 0; 0 0 1 0; 0 0 0 1; -24 -50 -35 -10","0;0;0;1","1 0 0 0","0")')
    assert tf["d"] == pytest.approx([24, 50, 35, 10, 1], rel=1e-6)
    assert tf["n"] == pytest.approx([1], rel=1e-6)


def test_ss_1x1_2x2_3x3_contra_calculo_manual(run_js):
    t1 = run_js('ssToTF("-2","1","3","0")')
    assert t1["n"] == pytest.approx([3]) and t1["d"] == pytest.approx([2, 1])
    t3 = run_js('ssToTF("0 1 0; 0 0 1; -6 -11 -6","0;0;1","0 1 0","0")')
    assert t3["d"] == pytest.approx([6, 11, 6, 1], rel=1e-6)
    assert t3["n"] == pytest.approx([0, 1], abs=1e-6)


@pytest.mark.parametrize("polos", ["[{r:0,i:0},{r:0,i:0}]",
                                   "[{r:0,i:1},{r:0,i:-1},{r:0,i:1},{r:0,i:-1}]",
                                   "[{r:1e-8,i:0},{r:-1e-8,i:0}]"])
def test_polo_repetido_no_eixo_e_instavel(run_js, polos):
    assert run_js(f"classEst({polos}).st") == "Instável"


def test_esc_escapa_aspas(run_js):
    entrada = json.dumps("<a href=\"x\" onclick='y'>&")
    assert run_js(f"esc({entrada})") == "&lt;a href=&quot;x&quot; onclick=&#39;y&#39;&gt;&amp;"


def test_modelo_com_id_ou_tipo_invalido_e_descartado(run_js):
    r = run_js("""(function(){var m=saneia({nodes:[{id:'ok1',type:'tf',x:1,y:2,params:{}},
      {id:'"><img src=x onerror=1>',type:'tf'},{id:'n2',type:'virus'}],
      edges:[{src:'ok1',dst:'n2'}]});return{nos:m.nodes.map(function(n){return n.id}),arestas:m.edges.length}})()""")
    assert r == {"nos": ["ok1"], "arestas": 0}
