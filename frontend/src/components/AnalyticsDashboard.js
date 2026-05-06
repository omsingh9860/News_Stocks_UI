import React, { useMemo } from 'react';
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { TrendingUp, TrendingDown, Newspaper, BarChart2, Briefcase } from 'lucide-react';
import { formatPrice } from '../utils/helpers';

const PORTFOLIO = [
  { symbol: 'RELIANCE', name: 'Reliance Industries', qty: 10, buyPrice: 2450, currentPrice: 2680 },
  { symbol: 'TCS',      name: 'Tata Consultancy',    qty: 5,  buyPrice: 3800, currentPrice: 3950 },
  { symbol: 'INFY',     name: 'Infosys',              qty: 15, buyPrice: 1480, currentPrice: 1520 },
  { symbol: 'HDFCBANK', name: 'HDFC Bank',            qty: 8,  buyPrice: 1620, currentPrice: 1700 },
  { symbol: 'WIPRO',    name: 'Wipro',                qty: 20, buyPrice: 470,  currentPrice: 490  },
  { symbol: 'SBIN',     name: 'State Bank of India',  qty: 25, buyPrice: 590,  currentPrice: 615  },
];

// eslint-disable-next-line no-unused-vars
const SENTIMENT_COLORS = { positive: '#10b981', negative: '#ef4444', neutral: '#6b7280' };
const PIE_COLORS = ['#10b981', '#ef4444', '#6b7280'];

const StatCard = ({ label, value, color }) => (
  <div className="stat-card">
    <div className="stat-card-value" style={color ? { color } : {}}>{value}</div>
    <div className="stat-card-label">{label}</div>
  </div>
);

const AnalyticsDashboard = ({ news = [], tradingIdeas = {}, recentlyViewed = [] }) => {
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

  // ── Portfolio calculations ───────────────────────────────────────────────
  const portfolioRows = PORTFOLIO.map(s => {
    const investedValue = s.qty * s.buyPrice;
    const currentValue  = s.qty * s.currentPrice;
    const pnl           = currentValue - investedValue;
    const pnlPct        = ((pnl / investedValue) * 100).toFixed(2);
    return { ...s, investedValue, currentValue, pnl, pnlPct };
  });

  const totalInvested = portfolioRows.reduce((s, r) => s + r.investedValue, 0);
  const totalCurrent  = portfolioRows.reduce((s, r) => s + r.currentValue,  0);
  const totalPnL      = totalCurrent - totalInvested;
  const totalPnLPct   = ((totalPnL / totalInvested) * 100).toFixed(2);

  const portfolioChartData = portfolioRows.map(r => ({
    symbol: r.symbol,
    'Current Value': Math.round(r.currentValue),
    'P&L': Math.round(r.pnl),
  }));

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

        {/* Portfolio BarChart */}
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
      </div>

      {/* ── Portfolio Table ───────────────────────────────────────────────── */}
      <div className="analytics-card">
        <div className="analytics-card-title">
          <Briefcase size={16} /> Mock Portfolio Simulation
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="portfolio-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Name</th>
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
                  <td>{r.name}</td>
                  <td>{r.qty}</td>
                  <td>₹{formatPrice(r.buyPrice)}</td>
                  <td>₹{formatPrice(r.currentPrice)}</td>
                  <td>₹{formatPrice(r.investedValue)}</td>
                  <td>₹{formatPrice(r.currentValue)}</td>
                  <td style={{ color: r.pnl >= 0 ? '#10b981' : '#ef4444' }}>
                    {r.pnl >= 0 ? '+' : ''}₹{formatPrice(r.pnl)}
                  </td>
                  <td style={{ color: r.pnl >= 0 ? '#10b981' : '#ef4444' }}>
                    {r.pnl >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                    {r.pnlPct}%
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
          <div className="portfolio-stat">
            <span className="portfolio-stat-label">Current Value</span>
            <span className="portfolio-stat-value">₹{formatPrice(totalCurrent)}</span>
          </div>
          <div className="portfolio-stat">
            <span className="portfolio-stat-label">Total P&amp;L</span>
            <span className="portfolio-stat-value" style={{ color: totalPnL >= 0 ? '#10b981' : '#ef4444' }}>
              {totalPnL >= 0 ? '+' : ''}₹{formatPrice(totalPnL)} ({totalPnLPct}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
