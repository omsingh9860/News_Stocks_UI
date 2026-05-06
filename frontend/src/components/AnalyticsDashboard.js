import React, { useMemo } from 'react';
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { TrendingUp, TrendingDown, Newspaper, BarChart2, Briefcase, AlertCircle } from 'lucide-react';
import { formatPrice } from '../utils/helpers';

const PIE_COLORS = ['#10b981', '#ef4444', '#6b7280'];
const ALLOC_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#ec4899', '#14b8a6', '#f97316'];

const StatCard = ({ label, value, color }) => (
  <div className="stat-card">
    <div className="stat-card-value" style={color ? { color } : {}}>{value}</div>
    <div className="stat-card-label">{label}</div>
  </div>
);

const AnalyticsDashboard = ({ news = [], tradingIdeas = {}, recentlyViewed = [], holdings = [] }) => {
  // ── Sentiment breakdown ──────────────────────────────────────────────────
  const sentimentData = useMemo(() => {
    const counts = { positive: 0, negative: 0, neutral: 0 };
    news.forEach(a => {
      const label = a.sentiment?.label || 'neutral';
      if (counts[label] !== undefined) counts[label]++;
      else counts.neutral++;
    });
    return [
      { name: 'Positive', value: counts.positive },
      { name: 'Negative', value: counts.negative },
      { name: 'Neutral',  value: counts.neutral  },
    ];
  }, [news]);

  const totalArticles = news.length;
  const positiveCount = sentimentData.find(d => d.name === 'Positive')?.value || 0;
  const positivePct = totalArticles ? ((positiveCount / totalArticles) * 100).toFixed(1) : '0.0';
  const avgScore = useMemo(() => {
    const scored = news.filter(a => a.sentiment?.score != null);
    if (!scored.length) return 'N/A';
    const sum = scored.reduce((acc, a) => acc + Number(a.sentiment.score), 0);
    return (sum / scored.length).toFixed(2);
  }, [news]);

  // ── Portfolio calculations (real holdings) ───────────────────────────────
  const portfolioRows = useMemo(() => holdings.map(h => {
    const investedValue = h.qty * h.buyPrice;
    const currentPrice  = h.manualCurrentPrice ?? null;
    const hasPrice      = currentPrice !== null;
    const currentValue  = hasPrice ? h.qty * currentPrice : null;
    const pnl           = hasPrice ? currentValue - investedValue : null;
    const pnlPct        = hasPrice ? ((pnl / investedValue) * 100).toFixed(2) : null;
    return { ...h, investedValue, currentValue, pnl, pnlPct, hasPrice };
  }), [holdings]);

  const pricedRows = portfolioRows.filter(r => r.hasPrice);

  const totalInvested = portfolioRows.reduce((s, r) => s + r.investedValue, 0);
  const totalCurrent  = pricedRows.reduce((s, r) => s + r.currentValue, 0);
  const totalPnL      = pricedRows.reduce((s, r) => s + r.pnl, 0);
  const totalPnLPct   = totalInvested > 0 ? ((totalPnL / totalInvested) * 100).toFixed(2) : '0.00';

  const portfolioChartData = pricedRows.map(r => ({
    symbol: r.symbol,
    'Current Value': Math.round(r.currentValue),
    'P&L': Math.round(r.pnl),
  }));

  // Allocation by invested value
  const allocationData = portfolioRows.map(r => ({
    name: r.symbol,
    value: Math.round(r.investedValue),
  }));

  // Top gainers / losers (only rows with price)
  const sortedByPnlPct = [...pricedRows].sort((a, b) => parseFloat(b.pnlPct) - parseFloat(a.pnlPct));
  const topGainers = sortedByPnlPct.filter(r => parseFloat(r.pnlPct) >= 0).slice(0, 3);
  const topLosers  = sortedByPnlPct.filter(r => parseFloat(r.pnlPct) < 0).slice(-3).reverse();

  return (
    <div className="analytics-section">
      <h2 className="section-title" style={{ marginBottom: '1.5rem' }}>
        <BarChart2 size={20} /> Analytics Dashboard
      </h2>

      {/* ── Key Stats ─────────────────────────────────────────────────────── */}
      <div className="stat-cards">
        <StatCard label="Total Articles" value={totalArticles} />
        <StatCard label="Positive %" value={`${positivePct}%`} color="#10b981" />
        <StatCard label="Avg Sentiment Score" value={avgScore} />
        <StatCard label="Trading Ideas" value={tradingIdeas?.summary?.total_ideas ?? 0} color="#3b82f6" />
        <StatCard label="Recently Viewed" value={recentlyViewed.length} />
      </div>

      {/* ── Charts row ────────────────────────────────────────────────────── */}
      <div className="analytics-grid">
        {/* Sentiment PieChart */}
        <div className="analytics-card">
          <div className="analytics-card-title">
            <Newspaper size={16} /> News Sentiment Distribution
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {sentimentData.map((entry, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* News Volume BarChart */}
        <div className="analytics-card">
          <div className="analytics-card-title">
            <BarChart2 size={16} /> News Volume by Sentiment
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sentimentData}>
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" name="Articles">
                {sentimentData.map((entry, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Portfolio Allocation PieChart */}
        {allocationData.length > 0 && (
          <div className="analytics-card">
            <div className="analytics-card-title">
              <Briefcase size={16} /> Portfolio Allocation (by invested)
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={allocationData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name }) => name}
                >
                  {allocationData.map((entry, i) => (
                    <Cell key={i} fill={ALLOC_COLORS[i % ALLOC_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `₹${formatPrice(v)}`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Portfolio P&L BarChart */}
        {portfolioChartData.length > 0 && (
          <div className="analytics-card">
            <div className="analytics-card-title">
              <Briefcase size={16} /> Portfolio – Current Value vs P&amp;L
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={portfolioChartData}>
                <XAxis dataKey="symbol" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `₹${formatPrice(v)}`} />
                <Legend />
                <Bar dataKey="Current Value" fill="#3b82f6" />
                <Bar dataKey="P&L" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ── Top Gainers / Losers ─────────────────────────────────────────── */}
      {(topGainers.length > 0 || topLosers.length > 0) && (
        <div className="analytics-grid">
          {topGainers.length > 0 && (
            <div className="analytics-card">
              <div className="analytics-card-title" style={{ color: '#10b981' }}>
                <TrendingUp size={16} /> Top Gainers
              </div>
              <div className="gainer-loser-list">
                {topGainers.map(r => (
                  <div key={r.symbol} className="gl-row">
                    <strong className="gl-symbol">{r.symbol}</strong>
                    <span className="gl-pnl" style={{ color: '#10b981' }}>
                      +₹{formatPrice(r.pnl)} ({r.pnlPct}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {topLosers.length > 0 && (
            <div className="analytics-card">
              <div className="analytics-card-title" style={{ color: '#ef4444' }}>
                <TrendingDown size={16} /> Top Losers
              </div>
              <div className="gainer-loser-list">
                {topLosers.map(r => (
                  <div key={r.symbol} className="gl-row">
                    <strong className="gl-symbol">{r.symbol}</strong>
                    <span className="gl-pnl" style={{ color: '#ef4444' }}>
                      ₹{formatPrice(r.pnl)} ({r.pnlPct}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Portfolio Table ───────────────────────────────────────────────── */}
      <div className="analytics-card">
        <div className="analytics-card-title">
          <Briefcase size={16} /> Portfolio Summary
        </div>

        {portfolioRows.length === 0 ? (
          <div className="empty-state" style={{ padding: '2rem 0' }}>
            <Briefcase size={36} />
            <p>No holdings yet. Go to the <strong>Holdings</strong> tab to add your portfolio.</p>
          </div>
        ) : (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table className="portfolio-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                    <th>Buy Price</th>
                    <th>Current</th>
                    <th>Invested</th>
                    <th>Current Value</th>
                    <th>P&amp;L</th>
                    <th>%</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolioRows.map(r => (
                    <tr key={r.symbol}>
                      <td><strong>{r.symbol}</strong></td>
                      <td>{r.qty}</td>
                      <td>₹{formatPrice(r.buyPrice)}</td>
                      <td>
                        {r.hasPrice
                          ? `₹${formatPrice(r.manualCurrentPrice)}`
                          : <span className="price-na"><AlertCircle size={13} /> N/A</span>}
                      </td>
                      <td>₹{formatPrice(r.investedValue)}</td>
                      <td>{r.hasPrice ? `₹${formatPrice(r.currentValue)}` : '—'}</td>
                      <td style={{ color: r.pnl != null ? (r.pnl >= 0 ? '#10b981' : '#ef4444') : undefined }}>
                        {r.pnl != null ? `${r.pnl >= 0 ? '+' : ''}₹${formatPrice(r.pnl)}` : '—'}
                      </td>
                      <td style={{ color: r.pnlPct != null ? (parseFloat(r.pnlPct) >= 0 ? '#10b981' : '#ef4444') : undefined }}>
                        {r.pnlPct != null
                          ? <>{parseFloat(r.pnlPct) >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />} {r.pnlPct}%</>
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="portfolio-summary">
              <div className="portfolio-stat">
                <span className="portfolio-stat-label">Total Invested</span>
                <span className="portfolio-stat-value">₹{formatPrice(totalInvested)}</span>
              </div>
              {pricedRows.length > 0 && (
                <>
                  <div className="portfolio-stat">
                    <span className="portfolio-stat-label">Current Value</span>
                    <span className="portfolio-stat-value">₹{formatPrice(totalCurrent)}</span>
                  </div>
                  <div className="portfolio-stat">
                    <span className="portfolio-stat-label">Unrealized P&amp;L</span>
                    <span className="portfolio-stat-value" style={{ color: totalPnL >= 0 ? '#10b981' : '#ef4444' }}>
                      {totalPnL >= 0 ? '+' : ''}₹{formatPrice(totalPnL)} ({totalPnLPct}%)
                    </span>
                  </div>
                </>
              )}
              {pricedRows.length < portfolioRows.length && (
                <div className="portfolio-stat">
                  <span className="portfolio-stat-label" style={{ color: '#f59e0b' }}>
                    <AlertCircle size={14} style={{ verticalAlign: 'middle' }} />
                    {' '}{portfolioRows.length - pricedRows.length} holding(s) missing current price
                  </span>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
