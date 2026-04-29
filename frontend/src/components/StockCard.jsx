/**
 * StockCard — Displays a single stock valuation result as a card
 */
export default function StockCard({ result, onClick }) {
  const { ticker, company_name, market_price, ensemble = {}, models = {} } = result

  const signal = ensemble.signal || 'N/A'
  const ensembleValue = ensemble.ensemble_value
  const mos = ensemble.margin_of_safety
  const validModels = ensemble.valid_model_count || 0

  // Signal class
  const signalClass = signal.toLowerCase().replace(/_/g, '-').replace(' ', '-')

  // Model values for bar chart
  const modelEntries = [
    { name: 'GGM', value: models.ggm?.value, valid: models.ggm?.valid },
    { name: 'DCF', value: models.dcf?.value, valid: models.dcf?.valid },
    { name: 'Comps', value: models.comps?.value, valid: models.comps?.valid },
    { name: 'RIM', value: models.rim?.value, valid: models.rim?.valid },
  ]

  // Max value for bar scaling
  const allValues = modelEntries.map(m => m.value).filter(Boolean)
  const maxVal = Math.max(...allValues, market_price || 0, ensembleValue || 0)

  const getCurrencySymbol = (t) => {
    if (!t) return '$'
    if (t.endsWith('.NS') || t.endsWith('.BO')) return '₹'
    if (t.endsWith('.L')) return '£'
    if (t.endsWith('.TO')) return 'CA$'
    if (t.match(/\.(DE|PA|MI|AS|MC)$/)) return '€'
    return '$'
  }
  const currency = getCurrencySymbol(ticker)

  const formatPrice = (v) => {
    if (v == null) return '—'
    return currency + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  const formatMos = (v) => {
    if (v == null) return '—'
    const pct = (v * 100).toFixed(1)
    return v >= 0 ? `+${pct}%` : `${pct}%`
  }

  const mosClass = mos > 0.05 ? 'positive' : mos < -0.05 ? 'negative' : 'neutral'

  return (
    <div className="stock-card" onClick={onClick} role="button" tabIndex={0} id={`card-${ticker}`}>
      <div className="card-header">
        <div className="ticker-info">
          <span className="ticker-symbol">{ticker}</span>
          <span className="company-name">{company_name || ''}</span>
        </div>
        <span className={`signal-badge ${signalClass}`}>
          {signal.replace(/_/g, ' ')}
        </span>
      </div>

      <div className="price-row">
        <div>
          <div className="price-label">Market Price</div>
          <div className="price-value">{formatPrice(market_price)}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="price-label">Intrinsic Value</div>
          <div className="price-value intrinsic">{formatPrice(ensembleValue)}</div>
        </div>
      </div>

      {/* Model Bars */}
      <div className="model-bars">
        {modelEntries.map(m => (
          <div className="model-row" key={m.name}>
            <span className="model-name">{m.name}</span>
            <div className="model-bar-track">
              <div
                className={`model-bar-fill ${!m.valid ? 'invalid' : ''}`}
                style={{ width: m.value && maxVal ? `${Math.min((m.value / maxVal) * 100, 100)}%` : '0%' }}
              />
            </div>
            <span className="model-value">{m.valid ? formatPrice(m.value) : '—'}</span>
          </div>
        ))}
      </div>

      {/* MOS */}
      <div className="mos-section">
        <span className="mos-label">Margin of Safety • {validModels}/4 models</span>
        <span className={`mos-value ${mosClass}`}>{formatMos(mos)}</span>
      </div>
    </div>
  )
}
