# BingX • Trading Analytics — v2

Dashboard Streamlit para analisar os exports da BingX (futuros perpétuos + spot): PnL, trades, timing,
posições reconstruídas, custos, **risco** e comportamento.

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```
Coloque os exports na pasta `data/` (ou envie pela barra lateral):
- `bingx_futures_income.json` ou `.csv` → PnL, taxas, funding (fonte principal)
- `bingx_operations.json` ou `.csv`     → fills de futuros + spot (prefira o JSON: traz MARKET/TAKE_PROFIT)

Variável opcional: `BINGX_DATA_DIR=/caminho/da/pasta`.
Sem dados à mão? Ligue **"Usar dados de demonstração"** na barra lateral (ou `python demo.py` gera CSVs de exemplo em `data/`).

## Estrutura
| Arquivo | Papel |
|---|---|
| `app.py` | Interface (abas, gráficos, insights) |
| `core.py` | Cálculo puro sem Streamlit: parsing, FIFO, estatísticas, drawdown, VaR, Monte Carlo, spot |
| `ui.py` | Tema, cards de KPI, hero, template dos gráficos |
| `demo.py` | Gerador de dados sintéticos no schema da BingX |

## O que mudou na v2
**Mantido:** as 9 abas originais, todas as métricas, insights, filtros, conciliação income × fills e downloads.

**Visual:** novo tema (hero com PnL do período, cards de KPI com cor por sinal, abas em pílulas, gráficos em cartões,
tipografia Inter/JetBrains Mono), legendas e tooltips padronizados.

**Novo:**
- Aba **🛡️ Risco**: drawdown atual e máximo, dias debaixo d'água, recovery factor, VaR/CVaR, Ulcer index,
  maiores episódios de drawdown e **simulação de Monte Carlo** (probabilidade de lucro, leque de resultados, risco de ruína).
- **Insights**: radar de qualidade (0–100) + filtro por nível; novas regras (sessões de mercado, maker × taker, SQN, VaR,
  tempo debaixo d'água, concentração por símbolo).
- **Visão geral**: SQN, recovery factor, VaR/CVaR, sequência atual, contribuição por símbolo e Long × Short.
- **PnL**: acumulado por símbolo e mapa mensal. **Trades**: estatísticas por símbolo e win rate × payoff.
- **Timing**: sessões Ásia/Europa/EUA. **Posições**: efeito "revenge trading" e duração ganhos × perdas.
- **Custos**: custo acumulado e Maker × Taker. **Spot**: PnL por custo médio, volume por símbolo.
- **Sidebar**: modo demonstração, cotação USDT/BRL (PnL em reais), períodos "90 dias" e "Ano atual".
- Leitura mais robusta (colunas opcionais, JSON embrulhado, mensagem amigável se o schema estiver errado) e
  gráfico de entradas/saídas bem mais rápido.
