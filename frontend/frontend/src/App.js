import React, { useEffect, useState, useCallback } from "react";
import "./App.css";

// Enhanced sentiment analysis with more sophisticated logic
const analyzeMarketSentiment = (title) => {
  const positiveWords = [
    'gain', 'rise', 'up', 'surge', 'boost', 'growth', 'profit', 'increase', 
    'bullish', 'strong', 'positive', 'buy', 'target', 'rally', 'jump', 'climb',
    'soar', 'spike', 'recovery', 'rebound', 'outperform', 'beat', 'exceed'
  ];
  const negativeWords = [
    'fall', 'drop', 'down', 'decline', 'loss', 'crash', 'bearish', 'weak', 
    'negative', 'plunge', 'dip', 'sell', 'cut', 'slump', 'tumble', 'slide',
    'retreat', 'underperform', 'miss', 'disappoint', 'concern', 'risk'
  ];
  
  const titleLower = title.toLowerCase();
  const positiveCount = positiveWords.filter(word => titleLower.includes(word)).length;
  const negativeCount = negativeWords.filter(word => titleLower.includes(word)).length;
  
  // Weighted scoring
  const score = positiveCount - negativeCount;
  
  if (score > 1) return 'bullish';
  if (score < -1) return 'bearish';
  return 'neutral';
};

// Enhanced stock information extraction
const extractStockInfo = (title) => {
  const companies = [
    { name: 'Reliance', symbol: 'RIL', price: 2845.60, change: +15.30, changePercent: 0.54 },
    { name: 'TCS', symbol: 'TCS', price: 4120.75, change: -22.45, changePercent: -0.54 },
    { name: 'Infosys', symbol: 'INFY', price: 1678.90, change: +8.75, changePercent: 0.52 },
    { name: 'HDFC Bank', symbol: 'HDFCBANK', price: 1542.30, change: -5.60, changePercent: -0.36 },
    { name: 'Wipro', symbol: 'WIPRO', price: 425.80, change: +2.15, changePercent: 0.51 },
    { name: 'Motilal Oswal', symbol: 'MOFS', price: 875.20, change: +12.45, changePercent: 1.44 },
    { name: 'Emcure Pharmaceuticals', symbol: 'EMCURE', price: 1250.44, change: -8.30, changePercent: -0.66 },
    { name: 'Dreamfolks Services', symbol: 'DREAMFOLKS', price: 445.80, change: +3.25, changePercent: 0.73 },
    { name: 'Nifty', symbol: 'NIFTY', price: 24855.85, change: +125.40, changePercent: 0.51 },
    { name: 'Sensex', symbol: 'SENSEX', price: 81332.72, change: -45.22, changePercent: -0.06 }
  ];
  
  return companies.find(company => 
    title.toLowerCase().includes(company.name.toLowerCase()) || 
    title.toLowerCase().includes(company.symbol.toLowerCase())
  );
};

const MoodMeter = ({ sentiment }) => {
  const getMoodEmoji = () => {
    switch (sentiment) {
      case 'bullish': return '📈';
      case 'bearish': return '📉';
      default: return '➖';
    }
  };

  const getMoodText = () => {
    switch (sentiment) {
      case 'bullish': return 'Bullish';
      case 'bearish': return 'Bearish';
      default: return 'Neutral';
    }
  };

  return (
    <div className={`mood-meter ${sentiment}`}>
      <span className="mood-emoji">{getMoodEmoji()}</span>
      <span className="mood-text">{getMoodText()}</span>
    </div>
  );
};

const StockInfo = ({ stock }) => {
  if (!stock) return null;

  const isPositive = stock.change > 0;
  const changeClass = isPositive ? 'positive' : 'negative';

  return (
    <div className="stock-info">
      <div className="stock-header">
        <div className="stock-name">
          <h4>{stock.name}</h4>
          <span className="stock-symbol">{stock.symbol}</span>
        </div>
        <div className="stock-price">
          <span className="price">₹{stock.price.toLocaleString()}</span>
          <div className={`change ${changeClass}`}>
            <span className="change-arrow">{isPositive ? '▲' : '▼'}</span>
            <span className="change-value">
              {isPositive ? '+' : ''}{stock.change} ({isPositive ? '+' : ''}{stock.changePercent}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

const SearchBar = ({ onSearch, loading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="search-bar">
      <div className="search-input-wrapper">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search news by keyword..."
          className="search-input"
          disabled={loading}
        />
        <button type="submit" className="search-btn" disabled={loading || !query.trim()}>
          🔍
        </button>
      </div>
    </form>
  );
};

const MarketIndices = ({ marketData }) => {
  if (!marketData) return null;

  return (
    <div className="market-indices">
      <h3>Market Indices</h3>
      <div className="indices-grid">
        {Object.entries(marketData.indices).map(([key, index]) => (
          <div key={key} className={`index-card ${index.trend}`}>
            <div className="index-name">{index.name}</div>
            <div className="index-value">₹{index.value.toLocaleString()}</div>
            <div className={`index-change ${index.trend}`}>
              {index.trend === 'up' ? '▲' : '▼'} {index.change} ({index.changePercent}%)
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const NewsCard = ({ item, index }) => {
  const sentiment = analyzeMarketSentiment(item.title);
  const stockInfo = extractStockInfo(item.title);
  
  const formatDate = (dateString) => {
    if (!dateString) return null;
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-IN', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="news-card">
      <div className="news-header">
        <div className="news-number">{index + 1}</div>
        <MoodMeter sentiment={sentiment} />
        <div className="market-impact">
          <span className="impact-icon">⚡</span>
          <span>Market Impact</span>
        </div>
      </div>

      <h2 className="news-title">{item.title}</h2>

      {item.description && (
        <p className="news-description">{item.description}</p>
      )}

      <div className="news-meta">
        {item.publishedAt && (
          <div className="meta-item">
            <span className="meta-icon">📅</span>
            <span>{formatDate(item.publishedAt)}</span>
          </div>
        )}
        {item.author && (
          <div className="meta-item">
            <span className="meta-icon">👤</span>
            <span>{item.author}</span>
          </div>
        )}
        {item.source && (
          <div className="meta-item">
            <span className="meta-icon">📰</span>
            <span>{item.source}</span>
          </div>
        )}
      </div>

      {stockInfo && <StockInfo stock={stockInfo} />}

      <div className="news-actions">
        <a
          href={item.link}
          target="_blank"
          rel="noopener noreferrer"
          className="read-more-btn"
        >
          <span>Read Full Article</span>
          <span className="external-icon">🔗</span>
        </a>
      </div>
    </div>
  );
};

const MarketSummary = ({ marketSentiment }) => {
  const total = marketSentiment.bullish + marketSentiment.bearish + marketSentiment.neutral;
  const bullishPercent = total > 0 ? Math.round((marketSentiment.bullish / total) * 100) : 0;
  const bearishPercent = total > 0 ? Math.round((marketSentiment.bearish / total) * 100) : 0;
  const neutralPercent = total > 0 ? Math.round((marketSentiment.neutral / total) * 100) : 0;

  return (
    <div className="market-summary">
      <h3>Market Sentiment Overview</h3>
      <div className="sentiment-grid">
        <div className="sentiment-item bullish">
          <span className="sentiment-emoji">📈</span>
          <span className="sentiment-label">Bullish</span>
          <span className="sentiment-count">{marketSentiment.bullish}</span>
          <span className="sentiment-percent">({bullishPercent}%)</span>
        </div>
        <div className="sentiment-item bearish">
          <span className="sentiment-emoji">📉</span>
          <span className="sentiment-label">Bearish</span>
          <span className="sentiment-count">{marketSentiment.bearish}</span>
          <span className="sentiment-percent">({bearishPercent}%)</span>
        </div>
        <div className="sentiment-item neutral">
          <span className="sentiment-emoji">➖</span>
          <span className="sentiment-label">Neutral</span>
          <span className="sentiment-count">{marketSentiment.neutral}</span>
          <span className="sentiment-percent">({neutralPercent}%)</span>
        </div>
      </div>
    </div>
  );
};

function App() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [marketData, setMarketData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  const API_BASE = "http://localhost:5000/api";

  const fetchNews = useCallback(async (query = '') => {
    try {
      setLoading(true);
      setError(null);
      
      const url = query ? `${API_BASE}/news/search?q=${encodeURIComponent(query)}` : `${API_BASE}/news`;
      const res = await fetch(url);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setNews(data);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Error fetching news:", err);
      setError(`Failed to load news: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchMarketData = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/market-summary`);
      if (res.ok) {
        const data = await res.json();
        setMarketData(data);
      }
    } catch (err) {
      console.error("Error fetching market data:", err);
    }
  }, []);

  const handleSearch = useCallback(async (query) => {
    setIsSearching(true);
    setSearchQuery(query);
    await fetchNews(query);
    setIsSearching(false);
  }, [fetchNews]);

  const handleRefresh = useCallback(() => {
    setSearchQuery('');
    fetchNews();
    fetchMarketData();
  }, [fetchNews, fetchMarketData]);

  useEffect(() => {
    fetchNews();
    fetchMarketData();
  }, [fetchNews, fetchMarketData]);

  const marketSentiment = React.useMemo(() => {
    if (news.length === 0) return { bullish: 0, bearish: 0, neutral: 0 };
    
    const sentiment = { bullish: 0, bearish: 0, neutral: 0 };
    news.forEach(item => {
      const mood = analyzeMarketSentiment(item.title);
      sentiment[mood]++;
    });
    
    return sentiment;
  }, [news]);

  return (
    <div className="App">
      {/* Header */}
      <header>
        <div className="header-content">
          <h1>📊 Market Pulse</h1>
          <div className="header-actions">
            {lastUpdated && (
              <div className="last-updated">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="refresh-btn"
            >
              <span className={`refresh-icon ${loading ? 'spinning' : ''}`}>🔄</span>
              Refresh
            </button>
          </div>
        </div>
        
        {/* Search Bar */}
        <div className="search-container">
          <SearchBar onSearch={handleSearch} loading={loading || isSearching} />
        </div>
        
        {/* Market Indices */}
        {marketData && <MarketIndices marketData={marketData} />}
        
        {/* Market Sentiment Summary */}
        {news.length > 0 && <MarketSummary marketSentiment={marketSentiment} />}
      </header>

      {/* Main Content */}
      <main>
        {loading && (
          <div className="loader">
            <div className="loading-spinner">
              <div className="spinner"></div>
              <span className="loading-icon">📊</span>
            </div>
            <span>Loading latest market news...</span>
          </div>
        )}

        {error && (
          <div className="error">
            <div className="error-icon">⚠️</div>
            <p>{error}</p>
            <button onClick={handleRefresh} className="retry-btn">
              <span>🔄</span> Try Again
            </button>
          </div>
        )}

        {!loading && !error && news.length === 0 && (
          <div className="no-news">
            <div className="no-news-icon">📰</div>
            <p>{searchQuery ? `No news found for "${searchQuery}"` : 'No news articles found.'}</p>
            {searchQuery && (
              <button onClick={handleRefresh} className="clear-search-btn">
                Clear Search
              </button>
            )}
          </div>
        )}

        {!loading && !error && news.length > 0 && (
          <>
            {searchQuery && (
              <div className="search-results-header">
                <h2>Search Results for "{searchQuery}"</h2>
                <span className="results-count">{news.length} articles found</span>
                <button onClick={handleRefresh} className="clear-search-btn">
                  Clear Search
                </button>
              </div>
            )}
            {news.map((item, index) => (
              <NewsCard key={index} item={item} index={index} />
            ))}
          </>
        )}
      </main>

      {/* Footer */}
      <footer>
        <div className="footer-content">
          <p>&copy; {new Date().getFullYear()} Market Pulse. Made with ❤️ and React.</p>
          <div className="footer-meta">
            <span>Powered by MoneyControl</span>
            <span>•</span>
            <span>Real-time sentiment analysis</span>
            <span>•</span>
            <span>Market data integration</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;