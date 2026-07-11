import React, { useState, useRef } from 'react'
import SentimentGauge from './SentimentGauge.jsx'

const API_BASE = '/api'

export default function App() {
  const [tab, setTab] = useState('single') // 'single' | 'batch'
  const [singleText, setSingleText] = useState('')
  const [singleResult, setSingleResult] = useState(null)
  const [singleLoading, setSingleLoading] = useState(false)
  const [singleError, setSingleError] = useState(null)

  const [batchResult, setBatchResult] = useState(null)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const [fileName, setFileName] = useState(null)
  const fileInputRef = useRef(null)

  const [health, setHealth] = useState(null)

  React.useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  async function handleSinglePredict(e) {
    e.preventDefault()
    if (!singleText.trim()) return
    setSingleLoading(true)
    setSingleError(null)
    setSingleResult(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: singleText }),
      })
      if (!res.ok) throw new Error('Server tidak dapat memproses teks ini.')
      const data = await res.json()
      setSingleResult(data)
    } catch (err) {
      setSingleError(err.message || 'Terjadi kesalahan saat menghubungi server.')
    } finally {
      setSingleLoading(false)
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setBatchLoading(true)
    setBatchError(null)
    setBatchResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload-csv`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Gagal memproses file CSV.')
      }
      const data = await res.json()
      setBatchResult(data)
    } catch (err) {
      setBatchError(err.message || 'Terjadi kesalahan saat mengunggah file.')
    } finally {
      setBatchLoading(false)
    }
  }

  const positifPercent = batchResult
    ? batchResult.summary?.positif?.percent ?? 0
    : singleResult
    ? (singleResult.label === 'positif' ? singleResult.confidence * 100 : (1 - singleResult.confidence) * 100)
    : 50

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.brand}>
            <span style={styles.brandMark}>◐</span>
            <span style={styles.brandName}>Sentimeter</span>
          </div>
          <p style={styles.tagline}>Baca suhu review produkmu, sebelum pelanggan lain membacanya.</p>
        </div>
      </header>

      <main style={styles.main}>
        <section style={styles.hero}>
          <div style={styles.heroText}>
            <span style={styles.eyebrow}>Analisis sentimen review e-commerce</span>
            <h1 style={styles.h1}>
              Tahu produkmu disukai atau dikeluhkan, <em style={styles.em}>dalam hitungan detik.</em>
            </h1>
            <p style={styles.lead}>
              Tempel satu komentar pelanggan, atau unggah file CSV berisi ratusan review dari Shopee maupun
              Tokopedia. Model akan membaca campuran Bahasa Indonesia dan Inggris sekaligus.
            </p>
            {health && (
              <div style={styles.modelBadge}>
                Model aktif: <strong>{health.model === 'naive_bayes' ? 'Naive Bayes' : 'Logistic Regression'}</strong>{' '}
                &middot; akurasi {Math.round(health.accuracy * 100)}%
              </div>
            )}
            {!health && (
              <div style={{ ...styles.modelBadge, color: 'var(--negatif)' }}>
                Backend belum terhubung — jalankan server FastAPI terlebih dahulu (lihat README).
              </div>
            )}
          </div>
          <div style={styles.heroGauge}>
            <SentimentGauge value={positifPercent} />
          </div>
        </section>

        <section style={styles.panel}>
          <div style={styles.tabs}>
            <button
              onClick={() => setTab('single')}
              style={{ ...styles.tabBtn, ...(tab === 'single' ? styles.tabBtnActive : {}) }}
            >
              Satu komentar
            </button>
            <button
              onClick={() => setTab('batch')}
              style={{ ...styles.tabBtn, ...(tab === 'batch' ? styles.tabBtnActive : {}) }}
            >
              Unggah CSV (banyak review)
            </button>
          </div>

          {tab === 'single' && (
            <form onSubmit={handleSinglePredict} style={styles.formBlock}>
              <label htmlFor="review-text" style={styles.label}>
                Tempel komentar pelanggan
              </label>
              <textarea
                id="review-text"
                value={singleText}
                onChange={(e) => setSingleText(e.target.value)}
                placeholder="Contoh: Barangnya bagus banget, pengiriman cepat, recommended seller!"
                style={styles.textarea}
                rows={4}
              />
              <button type="submit" style={styles.primaryBtn} disabled={singleLoading || !singleText.trim()}>
                {singleLoading ? 'Menganalisis…' : 'Analisis sentimen'}
              </button>

              {singleError && <p style={styles.errorText}>{singleError}</p>}

              {singleResult && (
                <div style={styles.resultCard}>
                  <div style={styles.resultRow}>
                    <span
                      style={{
                        ...styles.resultPill,
                        background: singleResult.label === 'positif' ? 'var(--positif-bg)' : 'var(--negatif-bg)',
                        color: singleResult.label === 'positif' ? 'var(--positif)' : 'var(--negatif)',
                      }}
                    >
                      {singleResult.label === 'positif' ? 'Positif' : 'Negatif'}
                    </span>
                    <span style={styles.confidenceText}>
                      Keyakinan model: {(singleResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={styles.cleanedText}>
                    Kata kunci yang dibaca model: <span style={{ color: 'var(--ink)' }}>{singleResult.cleaned_text || '—'}</span>
                  </p>

                  {singleResult.aspects && singleResult.aspects.length > 0 && (
                    <div style={styles.absaBlock}>
                      <p style={styles.absaLabel}>Rincian per aspek</p>
                      <div style={styles.absaChips}>
                        {singleResult.aspects.map((a, i) => (
                          <span
                            key={i}
                            style={{
                              ...styles.absaChip,
                              background: a.sentiment === 'positif' ? 'var(--positif-bg)' : a.sentiment === 'negatif' ? 'var(--negatif-bg)' : 'var(--paper)',
                              color: a.sentiment === 'positif' ? 'var(--positif)' : a.sentiment === 'negatif' ? 'var(--negatif)' : 'var(--ink-soft)',
                            }}
                          >
                            {a.aspect}: {a.sentiment}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </form>
          )}

          {tab === 'batch' && (
            <div style={styles.formBlock}>
              <label style={styles.label}>Unggah file CSV review (kolom: review / komentar / ulasan)</label>
              <div style={styles.dropZone} onClick={() => fileInputRef.current?.click()}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <span style={styles.dropZoneIcon}>↥</span>
                <span style={{ color: 'var(--ink-soft)' }}>
                  {fileName ? fileName : 'Klik untuk pilih file .csv dari komputermu'}
                </span>
              </div>

              {batchLoading && <p style={styles.loadingText}>Memproses file, mohon tunggu…</p>}
              {batchError && <p style={styles.errorText}>{batchError}</p>}

              {batchResult && (
                <div style={styles.batchResults}>
                  <div style={styles.statsRow}>
                    <div style={{ ...styles.statCard, borderColor: 'var(--positif)' }}>
                      <div style={{ ...styles.statNumber, color: 'var(--positif)' }}>
                        {batchResult.summary?.positif?.count ?? 0}
                      </div>
                      <div style={styles.statLabel}>Review positif ({batchResult.summary?.positif?.percent ?? 0}%)</div>
                    </div>
                    <div style={{ ...styles.statCard, borderColor: 'var(--negatif)' }}>
                      <div style={{ ...styles.statNumber, color: 'var(--negatif)' }}>
                        {batchResult.summary?.negatif?.count ?? 0}
                      </div>
                      <div style={styles.statLabel}>Review negatif ({batchResult.summary?.negatif?.percent ?? 0}%)</div>
                    </div>
                    <div style={styles.statCard}>
                      <div style={styles.statNumber}>{batchResult.total}</div>
                      <div style={styles.statLabel}>Total dianalisis</div>
                    </div>
                  </div>

                  <h3 style={styles.previewHeading}>Pratinjau hasil (maks. 50 baris pertama)</h3>
                  <div style={styles.tableWrap}>
                    <table style={styles.table}>
                      <thead>
                        <tr>
                          <th style={styles.th}>Komentar</th>
                          <th style={styles.th}>Label</th>
                          <th style={styles.th}>Keyakinan</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResult.preview.map((row, i) => (
                          <tr key={i}>
                            <td style={styles.td}>{row.text.length > 90 ? row.text.slice(0, 90) + '…' : row.text}</td>
                            <td style={styles.td}>
                              <span
                                style={{
                                  ...styles.miniPill,
                                  background: row.label === 'positif' ? 'var(--positif-bg)' : 'var(--negatif-bg)',
                                  color: row.label === 'positif' ? 'var(--positif)' : 'var(--negatif)',
                                }}
                              >
                                {row.label}
                              </span>
                            </td>
                            <td style={styles.td}>{(row.confidence * 100).toFixed(0)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer style={styles.footer}>
        <span>Dibangun dengan TF-IDF + Naive Bayes &middot; data Shopee &amp; Tokopedia</span>
      </footer>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    borderBottom: '1px solid var(--line)',
    background: 'var(--paper)',
  },
  headerInner: {
    maxWidth: 1040,
    margin: '0 auto',
    padding: '20px 24px',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  brandMark: {
    fontSize: 22,
    color: 'var(--positif)',
  },
  brandName: {
    fontFamily: 'var(--font-display)',
    fontWeight: 600,
    fontSize: 20,
    letterSpacing: -0.3,
  },
  tagline: {
    margin: '4px 0 0 32px',
    color: 'var(--ink-soft)',
    fontSize: 14,
  },
  main: {
    maxWidth: 1040,
    margin: '0 auto',
    padding: '48px 24px 80px',
    width: '100%',
  },
  hero: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 40,
    flexWrap: 'wrap',
    marginBottom: 56,
  },
  heroText: {
    flex: '1 1 460px',
    maxWidth: 560,
  },
  eyebrow: {
    fontSize: 13,
    letterSpacing: 1,
    textTransform: 'uppercase',
    color: 'var(--accent)',
    fontWeight: 600,
  },
  h1: {
    fontFamily: 'var(--font-display)',
    fontSize: 'clamp(32px, 5vw, 46px)',
    lineHeight: 1.12,
    fontWeight: 600,
    margin: '12px 0 16px',
    letterSpacing: -0.5,
  },
  em: {
    fontStyle: 'italic',
    color: 'var(--positif)',
  },
  lead: {
    fontSize: 16,
    lineHeight: 1.6,
    color: 'var(--ink-soft)',
    margin: 0,
  },
  modelBadge: {
    marginTop: 20,
    fontSize: 13,
    color: 'var(--ink-soft)',
    background: 'var(--paper-raised)',
    border: '1px solid var(--line)',
    borderRadius: 8,
    padding: '8px 14px',
    display: 'inline-block',
  },
  heroGauge: {
    flex: '0 0 auto',
    background: 'var(--paper-raised)',
    border: '1px solid var(--line)',
    borderRadius: 16,
    padding: '24px 28px 16px',
  },
  panel: {
    background: 'var(--paper-raised)',
    border: '1px solid var(--line)',
    borderRadius: 16,
    overflow: 'hidden',
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid var(--line)',
  },
  tabBtn: {
    flex: 1,
    padding: '16px 20px',
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--ink-soft)',
    cursor: 'pointer',
  },
  tabBtnActive: {
    color: 'var(--positif)',
    borderBottomColor: 'var(--positif)',
  },
  formBlock: {
    padding: 28,
  },
  label: {
    display: 'block',
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 8,
    color: 'var(--ink)',
  },
  textarea: {
    width: '100%',
    border: '1px solid var(--line)',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    fontFamily: 'var(--font-body)',
    resize: 'vertical',
    background: 'var(--paper)',
    color: 'var(--ink)',
  },
  primaryBtn: {
    marginTop: 14,
    background: 'var(--positif)',
    color: '#fff',
    border: 'none',
    borderRadius: 10,
    padding: '12px 22px',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
  errorText: {
    color: 'var(--negatif)',
    fontSize: 13,
    marginTop: 12,
  },
  loadingText: {
    color: 'var(--ink-soft)',
    fontSize: 13,
    marginTop: 12,
  },
  resultCard: {
    marginTop: 20,
    border: '1px solid var(--line)',
    borderRadius: 10,
    padding: 18,
    background: 'var(--paper)',
  },
  resultRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  resultPill: {
    padding: '4px 12px',
    borderRadius: 100,
    fontSize: 13,
    fontWeight: 700,
    textTransform: 'capitalize',
  },
  confidenceText: {
    fontSize: 13,
    color: 'var(--ink-soft)',
  },
  cleanedText: {
    marginTop: 12,
    fontSize: 13,
    color: 'var(--ink-soft)',
    fontStyle: 'italic',
  },
  absaBlock: {
    marginTop: 14,
    paddingTop: 14,
    borderTop: '1px solid var(--line)',
  },
  absaLabel: {
    fontSize: 12,
    fontWeight: 600,
    color: 'var(--ink-soft)',
    margin: '0 0 8px',
  },
  absaChips: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
  },
  absaChip: {
    padding: '4px 12px',
    borderRadius: 100,
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'capitalize',
  },
  dropZone: {
    border: '1.5px dashed var(--line)',
    borderRadius: 12,
    padding: '36px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    background: 'var(--paper)',
  },
  dropZoneIcon: {
    fontSize: 24,
    color: 'var(--accent)',
  },
  batchResults: {
    marginTop: 28,
  },
  statsRow: {
    display: 'flex',
    gap: 16,
    flexWrap: 'wrap',
    marginBottom: 28,
  },
  statCard: {
    flex: '1 1 140px',
    border: '1px solid var(--line)',
    borderLeft: '4px solid var(--ink)',
    borderRadius: 10,
    padding: '16px 18px',
    background: 'var(--paper)',
  },
  statNumber: {
    fontFamily: 'var(--font-display)',
    fontSize: 30,
    fontWeight: 600,
  },
  statLabel: {
    fontSize: 12,
    color: 'var(--ink-soft)',
    marginTop: 4,
  },
  previewHeading: {
    fontSize: 14,
    fontWeight: 600,
    marginBottom: 12,
  },
  tableWrap: {
    border: '1px solid var(--line)',
    borderRadius: 10,
    overflow: 'hidden',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 13,
  },
  th: {
    textAlign: 'left',
    padding: '10px 14px',
    background: 'var(--paper)',
    borderBottom: '1px solid var(--line)',
    fontWeight: 600,
    color: 'var(--ink-soft)',
  },
  td: {
    padding: '10px 14px',
    borderBottom: '1px solid var(--line)',
  },
  miniPill: {
    padding: '2px 10px',
    borderRadius: 100,
    fontSize: 11,
    fontWeight: 700,
    textTransform: 'capitalize',
  },
  footer: {
    textAlign: 'center',
    padding: '20px',
    fontSize: 12,
    color: 'var(--ink-soft)',
    borderTop: '1px solid var(--line)',
  },
}
