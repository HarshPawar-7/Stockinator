/**
 * AgentChat — Interactive AI valuation agent (Groq-powered)
 */
import { useState, useRef, useEffect } from 'react'

export default function AgentChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Welcome to Stockinator AI! 🤖\n\nI can help you analyze stocks using fundamental valuation models. Try:\n\n• "Value AAPL"\n• "Compare MSFT GOOGL AMZN"\n• "Is KO undervalued?"\n• "What's the intrinsic value of NVDA?"\n\nI'll fetch real data and run GGM, DCF, Comps, and RIM models to give you a comprehensive analysis.`
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMsg = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setLoading(true)

    try {
      const res = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error: ${e.message}\n\nMake sure:\n1. API server is running (python3 api/server.py)\n2. GROQ_API_KEY is set in .env`
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant" style={{ opacity: 0.6 }}>
            🔄 Analyzing...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-row">
        <input
          id="chat-input"
          className="chat-input"
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about any stock... e.g. 'Value AAPL'"
          disabled={loading}
        />
        <button
          id="chat-send-btn"
          className="chat-send-btn"
          onClick={sendMessage}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}
