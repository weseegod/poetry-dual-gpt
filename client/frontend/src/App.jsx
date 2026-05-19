import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Chào bạn! Gửi cho tôi một câu lục bát, tôi sẽ đối lại. 🎭' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState({
    temperature: 0.75,
    top_k: 50,
    top_p: 0.92,
    max_tokens: 64,
  })
  const [showSettings, setShowSettings] = useState(false)
  const bottomRef = useRef(null)

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => { scrollToBottom() }, [messages])

  const send = async (e) => {
    e?.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { role: 'user', text }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, ...settings }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'bot', text: data.response }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'bot',
        text: `⚠️ ${err.message}. Hãy chắc chắn server đang chạy.`
      }])
    } finally {
      setLoading(false)
    }
  }

  // Quick prompt buttons
  const quickPrompts = [
    'Thân em như chẽn lúa đòng đòng',
    'Trăm năm trong cõi người ta',
    'Gió đưa cành trúc la đà',
  ]

  return (
    <div className="chat-app">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <span className="header-avatar">🎭</span>
          <div>
            <h1>Poetry Duel</h1>
            <p className="header-sub">Lục Bát GPT</p>
          </div>
        </div>
        <button
          className={`settings-btn ${showSettings ? 'active' : ''}`}
          onClick={() => setShowSettings(!showSettings)}
        >
          ⚙️
        </button>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="settings-panel">
          <label>
            Temperature: <strong>{settings.temperature}</strong>
            <input type="range" min="0.3" max="1.5" step="0.05"
              value={settings.temperature}
              onChange={e => setSettings(s => ({ ...s, temperature: +e.target.value }))} />
          </label>
          <label>
            Top-K: <strong>{settings.top_k}</strong>
            <input type="range" min="10" max="200" step="10"
              value={settings.top_k}
              onChange={e => setSettings(s => ({ ...s, top_k: +e.target.value }))} />
          </label>
          <label>
            Top-P: <strong>{settings.top_p}</strong>
            <input type="range" min="0.5" max="1.0" step="0.02"
              value={settings.top_p}
              onChange={e => setSettings(s => ({ ...s, top_p: +e.target.value }))} />
          </label>
          <label>
            Max tokens: <strong>{settings.max_tokens}</strong>
            <input type="range" min="32" max="128" step="8"
              value={settings.max_tokens}
              onChange={e => setSettings(s => ({ ...s, max_tokens: +e.target.value }))} />
          </label>
        </div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'bot' ? '🎭' : '👤'}
            </div>
            <div className="message-bubble">
              <div className="message-text">{msg.text}</div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="message-avatar">🎭</div>
            <div className="message-bubble typing">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Quick prompts */}
      <div className="quick-prompts">
        {quickPrompts.map(p => (
          <button key={p} onClick={() => setInput(p)}
            className="quick-btn">{p}</button>
        ))}
      </div>

      {/* Input */}
      <form className="chat-input" onSubmit={send}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Nhập câu lục bát..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Gửi
        </button>
      </form>
    </div>
  )
}

export default App
