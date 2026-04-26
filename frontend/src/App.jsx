import { useState, useCallback, useEffect } from 'react'
import StockCard from './components/StockCard'
import DetailModal from './components/DetailModal'
import AgentChat from './components/AgentChat'

const QUICK_PICKS = [
  { label: 'FAANG', tickers: 'AAPL, MSFT, GOOGL, AMZN, META' },
  { label: 'Dividend', tickers: 'KO, JNJ, PG, PEP' },
  { label: 'Growth', tickers: 'NVDA, TSLA, AMD, NFLX' },
]

export default function App() {
  const [view, setView] = useState('valuate') // 'valuate' | 'agent'
  
  // Theme logic
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const [query, setQuery] = useState('')
  const [includePeers, setIncludePeers] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingTicker, setLoadingTicker] = useState('')
  const [results, setResults] = useState([])
  const [selectedResult, setSelectedResult] = useState(null)
  const [error, setError] = useState(null)

  const runValuation = useCallback(async (tickerString) => {
    const tickers = tickerString
      .split(/[\s,]+/)
      .map(t => t.trim().toUpperCase())
      .filter(Boolean)

    if (tickers.length === 0) return

    setLoading(true)
    setError(null)
    setResults([])
    setLoadingTicker(tickers.join(', '))

    try {
      const res = await fetch('/api/valuate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers, include_peers: includePeers }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResults(data.results || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      setLoadingTicker('')
    }
  }, [includePeers])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) runValuation(query)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">📊</span>
          <span className="logo-gradient">Stockinator</span>
        </div>
        <nav className="header-nav">
          <button
            className="nav-btn"
            onClick={toggleTheme}
            style={{ marginRight: '8px', padding: '6px 12px', fontSize: '1rem' }}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <button
            className={`nav-btn ${view === 'valuate' ? 'active' : ''}`}
            onClick={() => setView('valuate')}
          >
            ⚡ Valuate
          </button>
          <button
            className={`nav-btn ${view === 'agent' ? 'active' : ''}`}
            onClick={() => setView('agent')}
          >
            🤖 AI Agent
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {view === 'valuate' ? (
          <>
            {/* Search */}
            <section className="search-section">
              <form className="search-bar" onSubmit={handleSubmit}>
                <input
                  id="ticker-input"
                  className="search-input"
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Enter tickers separated by commas… e.g. AAPL, MSFT, GOOGL"
                  disabled={loading}
                />
                <button
                  id="valuate-btn"
                  className="search-btn"
                  type="submit"
                  disabled={loading || !query.trim()}
                >
                  {loading ? '⏳ Analyzing...' : '🚀 Valuate'}
                </button>
              </form>

              <div className="search-options">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={includePeers}
                    onChange={e => setIncludePeers(e.target.checked)}
                  />
                  Include peer comparisons (slower)
                </label>

                <div className="quick-picks">
                  {QUICK_PICKS.map(qp => (
                    <button
                      key={qp.label}
                      className="quick-pick-btn"
                      onClick={() => { setQuery(qp.tickers); runValuation(qp.tickers) }}
                      disabled={loading}
                    >
                      {qp.label}
                    </button>
                  ))}
                </div>
              </div>
            </section>

            {/* Loading */}
            {loading && (
              <div className="loading-container">
                <div className="spinner" />
                <p className="loading-text">
                  Analyzing <span className="loading-ticker">{loadingTicker}</span>
                </p>
                <p className="loading-text" style={{ fontSize: '0.82rem' }}>
                  Running GGM · DCF · Comps · RIM models...
                </p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="loading-container">
                <p style={{ color: 'var(--signal-sell)', fontSize: '1.1rem', fontWeight: 600 }}>
                  ❌ {error}
                </p>
                <p className="loading-text">
                  Make sure the API server is running: <code>python3 api/server.py</code>
                </p>
              </div>
            )}

            {/* Results */}
            {results.length > 0 && (
              <div className="results-grid">
                {results.map(r => (
                  <StockCard
                    key={r.ticker}
                    result={r}
                    onClick={() => setSelectedResult(r)}
                  />
                ))}
              </div>
            )}

            {/* Empty State */}
            {!loading && !error && results.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">🎯</div>
                <div className="empty-title">Enter tickers to begin valuation</div>
                <div className="empty-subtitle">
                  Get intrinsic value estimates using 4 financial models
                  with ML-powered ensemble averaging
                </div>
              </div>
            )}
          </>
        ) : (
          <AgentChat />
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        ⚠️ Not investment advice. Automated analysis for educational purposes only.
      </footer>

      {/* Detail Modal */}
      {selectedResult && (
        <DetailModal
          result={selectedResult}
          onClose={() => setSelectedResult(null)}
        />
      )}
    </div>
  )
}
