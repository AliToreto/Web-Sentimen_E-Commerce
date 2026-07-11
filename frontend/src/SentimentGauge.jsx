import React from 'react'

// Gauge melengkung 180 derajat. value: 0 (semua negatif) - 100 (semua positif)
export default function SentimentGauge({ value = 50, size = 280 }) {
  const clamped = Math.max(0, Math.min(100, value))
  const angle = (clamped / 100) * 180 // 0..180 derajat
  const radians = ((180 - angle) * Math.PI) / 180
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 24

  const needleX = cx + r * Math.cos(radians)
  const needleY = cy - r * Math.sin(radians)

  // arc background segments
  const describeArc = (startAngle, endAngle, radius) => {
    const start = polarToCartesian(cx, cy, radius, endAngle)
    const end = polarToCartesian(cx, cy, radius, startAngle)
    const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1'
    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`
  }

  function polarToCartesian(centerX, centerY, radius, angleDeg) {
    const angleRad = ((180 - angleDeg) * Math.PI) / 180
    return {
      x: centerX + radius * Math.cos(angleRad),
      y: centerY - radius * Math.sin(angleRad),
    }
  }

  const label = clamped >= 60 ? 'Cenderung Positif' : clamped <= 40 ? 'Cenderung Negatif' : 'Berimbang'
  const labelColor = clamped >= 60 ? 'var(--positif)' : clamped <= 40 ? 'var(--negatif)' : 'var(--accent)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width={size} height={size / 1.65} viewBox={`0 0 ${size} ${size / 1.65}`}>
        <path d={describeArc(0, 180, r)} fill="none" stroke="var(--line)" strokeWidth="18" strokeLinecap="round" />
        <path
          d={describeArc(0, clamped >= 50 ? 90 : angle, r)}
          fill="none"
          stroke="var(--negatif)"
          strokeWidth="18"
          strokeLinecap="round"
          opacity="0.85"
        />
        <path
          d={describeArc(90, angle >= 90 ? angle : 90, r)}
          fill="none"
          stroke="var(--positif)"
          strokeWidth="18"
          strokeLinecap="round"
          opacity="0.85"
        />
        {/* needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke="var(--ink)"
          strokeWidth="3"
          strokeLinecap="round"
          style={{ transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
        />
        <circle cx={cx} cy={cy} r="7" fill="var(--ink)" />
      </svg>
      <div style={{ textAlign: 'center', marginTop: -8 }}>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 40, fontWeight: 600, color: labelColor }}>
          {clamped.toFixed(0)}%
        </div>
        <div style={{ fontSize: 13, color: 'var(--ink-soft)', letterSpacing: 0.3, marginTop: 2 }}>
          {label} &middot; skor positif
        </div>
      </div>
    </div>
  )
}
