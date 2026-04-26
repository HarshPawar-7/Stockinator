/**
 * DetailModal — Full valuation breakdown for a selected stock
 */
import { useEffect } from 'react'

export default function DetailModal({ result, onClose }) {
  const { ticker, company_name, market_price, ensemble = {}, models = {}, inputs = {}, data_quality = {} } = result

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const formatPrice = (v) => {
    if (v == null) return '—'
    return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  const formatPct = (v) => {
    if (v == null) return '—'
    return (v * 100).toFixed(2) + '%'
  }

  const signal = ensemble.signal || 'N/A'
  const signalClass = signal.toLowerCase().replace(/_/g, '-').replace(' ', '-')
  const warnings = data_quality?.warnings || []

  const modelRows = [
    { name: 'Gordon Growth (GGM)', data: models.ggm },
    { name: 'Discounted Cash Flow (DCF)', data: models.dcf },
    { name: 'Comparable Companies', data: models.comps },
    { name: 'Residual Income (RIM)', data: models.rim },
  ]

  return (
    <div className="modal-overlay" onClick={onClose} id={`modal-${ticker}`}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="modal-header">
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>
              {ticker} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>— {company_name}</span>
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: '4px' }}>
              {result.valuation_date}
            </p>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Summary */}
        <table className="detail-table">
          <tbody>
            <tr>
              <td style={{ color: 'var(--text-muted)' }}>Market Price</td>
              <td style={{ fontWeight: 700 }}>{formatPrice(market_price)}</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }}>Ensemble Intrinsic Value</td>
              <td style={{ fontWeight: 700, color: 'var(--accent-blue)' }}>{formatPrice(ensemble.ensemble_value)}</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }}>95% Confidence Interval</td>
              <td>{formatPrice(ensemble.ci_lower)} — {formatPrice(ensemble.ci_upper)}</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }}>Margin of Safety</td>
              <td style={{ fontWeight: 700 }} className={`mos-value ${ensemble.margin_of_safety > 0.05 ? 'positive' : ensemble.margin_of_safety < -0.05 ? 'negative' : 'neutral'}`}>
                {formatPct(ensemble.margin_of_safety)}
              </td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }}>Signal</td>
              <td><span className={`signal-badge ${signalClass}`}>{signal.replace(/_/g, ' ')}</span></td>
            </tr>
          </tbody>
        </table>

        {/* Model Results */}
        <div className="detail-section-title">Model Results</div>
        <table className="detail-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Value</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {modelRows.map(m => (
              <tr key={m.name}>
                <td>{m.name}</td>
                <td style={{ fontWeight: 600 }}>{m.data?.valid ? formatPrice(m.data.value) : '—'}</td>
                <td>{m.data?.valid ? '✅' : '❌'}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Key Assumptions */}
        <div className="detail-section-title">Key Assumptions</div>
        <table className="detail-table">
          <tbody>
            <tr><td style={{ color: 'var(--text-muted)' }}>Risk-Free Rate</td><td>{formatPct(inputs.risk_free_rate)}</td></tr>
            <tr><td style={{ color: 'var(--text-muted)' }}>Market Premium</td><td>{formatPct(inputs.market_premium)}</td></tr>
            <tr><td style={{ color: 'var(--text-muted)' }}>Beta</td><td>{inputs.beta?.toFixed(3) || '—'}</td></tr>
            <tr><td style={{ color: 'var(--text-muted)' }}>Cost of Equity</td><td>{formatPct(inputs.cost_of_equity)}</td></tr>
            <tr><td style={{ color: 'var(--text-muted)' }}>WACC</td><td>{formatPct(inputs.wacc)}</td></tr>
            <tr><td style={{ color: 'var(--text-muted)' }}>Sector</td><td>{inputs.sector || '—'}</td></tr>
          </tbody>
        </table>

        {/* Model Weights */}
        {ensemble.model_weights_used && (
          <>
            <div className="detail-section-title">Ensemble Weights</div>
            <table className="detail-table">
              <tbody>
                {Object.entries(ensemble.model_weights_used).map(([model, weight]) => (
                  <tr key={model}>
                    <td style={{ color: 'var(--text-muted)' }}>{model.toUpperCase()}</td>
                    <td>{(weight * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {/* Warnings */}
        {warnings.length > 0 && (
          <>
            <div className="detail-section-title">Risk Flags ({warnings.length})</div>
            <ul className="warning-list">
              {warnings.map((w, i) => (
                <li className="warning-item" key={i}>⚠️ {w}</li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}
