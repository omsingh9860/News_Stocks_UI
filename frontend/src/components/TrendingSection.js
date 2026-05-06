import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, Minus, Clock, BarChart3, RefreshCw, AlertCircle } from 'lucide-react';

const getSentimentColor = (label) => {
  switch (label) {
    case 'positive': return '#10b981';
    case 'negative': return '#ef4444';
    default: return '#6b7280';
  }
};

const getSentimentIcon = (label) => {
  switch (label) {
    case 'positive': return <TrendingUp size={14} />;
    case 'negative': return <TrendingDown size={14} />;
    default: return <Minus size={14} />;
  }
};

const TrendingSection = ({ recentlyViewed, onSelectStock, news }) => {
  const [trendingStocks, setTrendingStocks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Compute trending from news prop if available, otherwise fetch from API
  const computeTrendingFromNews = useCallback((newsItems) => {
    const stockData = {};
    (newsItems || []).forEach(article => {
      const stocks = article.stocks_with_sentiment || article.stocks || [];
      stocks.forEach(stock => {
        const sym = stock.symbol;
        if (!sym) return;
        if (!stockData[sym]) {
          stockData[sym] = { name: stock.name || sym, symbol: sym, mentions: 0, sentiments: [] };
        }
        stockData[sym].mentions += 1;
        if (typeof stock.sentiment === 'number') {
          stockData[sym].sentiments.push(stock.sentiment);
        }
      });
    });

    return Object.values(stockData)
      .map(s => {
        const avg = s.sentiments.length
          ? s.sentiments.reduce((a, b) => a + b, 0) / s.sentiments.length
          : 0;
        return {
          ...s,
          average_sentiment: Math.round(avg * 1000) / 1000,
          sentiment_label: avg > 0.1 ? 'positive' : avg < -0.1 ? 'negative' : 'neutral',
          trending_score: s.mentions * (1 + Math.abs(avg)),
          mention_count: s.mentions,
        };
      })
      .sort((a, b) => b.trending_score - a.trending_score)
      .slice(0, 10);
  }, []);

  const fetchTrending = useCallback(async () => {
    setLoading(true);
    try {
      const apiUrl = process.env.REACT_APP_API_URL;
      const res = await fetch(`${apiUrl}/api/stocks/trending`);
      if (res.ok) {
        const data = await res.json();
        setTrendingStocks(data.trending_stocks || []);
        setLastUpdated(new Date());
        return;
      }
    } catch (e) {
      // fall through to local computation
    }
    // Fallback: compute from news prop
    if (news && news.length > 0) {
      setTrendingStocks(computeTrendingFromNews(news));
      setLastUpdated(new Date());
    }
    setLoading(false);
  }, [news, computeTrendingFromNews]);

  useEffect(() => {
    // Prefer local computation from existing news to avoid extra network calls
    if (news && news.length > 0) {
      setTrendingStocks(computeTrendingFromNews(news));
      setLastUpdated(new Date());
    } else {
      fetchTrending();
    }
  }, [news, computeTrendingFromNews, fetchTrending]);

  const formatTime = (date) =>
    date
      ? new Intl.DateTimeFormat('en-IN', { hour: '2-digit', minute: '2-digit' }).format(date)
      : '';

  return (
    <section className="trending-section">
      {/* Trending Stocks */}
      <div className="trending-stocks-panel">
        <div className="section-header">
          <h2 className="section-title">
            <BarChart3 size={20} />
            Trending Stocks
            {lastUpdated && (
              <span className="last-updated-small">
                Updated {formatTime(lastUpdated)}
              </span>
            )}
          </h2>
          <button
            className="btn btn-secondary btn-sm"
            onClick={fetchTrending}
            disabled={loading}
            title="Refresh trending"
          >
            <RefreshCw size={14} className={loading ? 'spinning' : ''} />
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>

        {trendingStocks.length === 0 ? (
          <div className="empty-state" style={{ minHeight: 150 }}>
            <AlertCircle size={36} />
            <p>No trending data yet. Data updates as news loads.</p>
          </div>
        ) : (
          <div className="trending-stocks-grid">
            {trendingStocks.map((stock, idx) => (
              <div
                key={stock.symbol}
                className="trending-stock-card"
                onClick={() => onSelectStock && onSelectStock({ symbol: stock.symbol, name: stock.name, exchange: 'NSE' })}
                title={`Click to view ${stock.symbol} chart`}
              >
                <div className="trending-rank">#{idx + 1}</div>
                <div className="trending-stock-info">
                  <span className="trending-symbol">{stock.symbol}</span>
                  <span className="trending-name">{stock.name}</span>
                </div>
                <div className="trending-meta">
                  <div
                    className="trending-sentiment"
                    style={{ color: getSentimentColor(stock.sentiment_label) }}
                  >
                    {getSentimentIcon(stock.sentiment_label)}
                    <span>{stock.mention_count} mention{stock.mention_count !== 1 ? 's' : ''}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recently Viewed */}
      {recentlyViewed.length > 0 && (
        <div className="recently-viewed-panel">
          <h3 className="section-title" style={{ fontSize: '1.25rem' }}>
            <Clock size={18} />
            Recently Viewed
          </h3>
          <div className="recently-viewed-list">
            {recentlyViewed.map((item) => (
              <div key={item.id + item.viewedAt} className="recently-viewed-item">
                <div className="rv-icon">
                  {item.type === 'stock' ? '📈' : '📰'}
                </div>
                <div className="rv-info">
                  <span className="rv-title">{item.title}</span>
                  <span className="rv-time">
                    <Clock size={12} />
                    {item.viewedAt
                      ? new Intl.DateTimeFormat('en-IN', { hour: '2-digit', minute: '2-digit' }).format(new Date(item.viewedAt))
                      : ''}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
};

export default TrendingSection;
