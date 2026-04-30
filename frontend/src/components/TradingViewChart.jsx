import { useEffect, useRef, memo } from 'react'

function TradingViewChart({ ticker, theme }) {
  const container = useRef()

  useEffect(() => {
    // Format ticker for TradingView
    let tvTicker = ticker
    if (ticker.endsWith('.NS')) tvTicker = `NSE:${ticker.replace('.NS', '')}`
    else if (ticker.endsWith('.BO')) tvTicker = `BSE:${ticker.replace('.BO', '')}`
    else if (ticker.endsWith('.L')) tvTicker = `LSE:${ticker.replace('.L', '')}`

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = `
      {
        "autosize": true,
        "symbol": "${tvTicker}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "${theme === 'dark' ? 'dark' : 'light'}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "backgroundColor": "${theme === 'dark' ? '#151a25' : '#ffffff'}",
        "gridColor": "${theme === 'dark' ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)'}",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "allow_symbol_change": false,
        "calendar": false,
        "support_host": "https://www.tradingview.com"
      }
    `

    if (container.current) {
      container.current.innerHTML = ''
      container.current.appendChild(script)
    }
  }, [ticker, theme])

  return (
    <div className="tradingview-widget-container" style={{ height: "400px", width: "100%", marginBottom: "24px", borderRadius: "8px", overflow: "hidden", border: "1px solid var(--border-subtle)" }}>
      <div className="tradingview-widget-container__widget" ref={container} style={{ height: "100%", width: "100%" }}></div>
    </div>
  )
}

export default memo(TradingViewChart)
