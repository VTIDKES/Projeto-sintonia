{/* Toolbar */}
      <div className="bg-gray-800 p-4 flex gap-2 flex-wrap">
        <button
          onClick={() => addBlock('transfer')}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
        >
          <Plus size={20} />
          Função Transferência
        </button>
        <button
          onClick={() => addBlock('sum')}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
        >
          <Plus size={20} />
          Somador
        </button>
        <button
          onClick={() => addBlock('gain')}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
        >
          <Plus size={20} />
          Ganho
        </button>
        <button
          onClick={() => addBlock('integrator')}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
        >
          <Plus size={20} />
          Integrador
        </button>
        <button
          onClick={calculateSystemResponse}
          disabled={blocks.length === 0 || isCalculating}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors ml-auto"
        >
          <Play size={20} />
          {isCalculating ? 'Calculando...' : 'Calcular Resposta'}
        </button>
      </div>

