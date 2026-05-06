import React, { useState, useEffect, useCallback } from 'react';
import { 
  ExternalLink, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Clock, 
  User, 
  Tag, 
  BarChart3, 
  Eye, 
  Target, 
  AlertCircle, 
  RefreshCw,
  Filter,
  Search,
  Globe,
  DollarSign,
  Activity,
  X,
  MessageSquare,
} from 'lucide-react';
import "./App.css";
import { useWatchlist } from './hooks/useWatchlist';
import { useRecentlyViewed } from './hooks/useRecentlyViewed';
import Watchlist from './components/Watchlist';
import PersonalizedNews from './components/PersonalizedNews';
import TrendingSection from './components/TrendingSection';
import StockChart from './components/StockChart';
import FeedbackForm from './components/FeedbackForm';
import OnboardingModal, { shouldShowOnboarding } from './components/OnboardingModal';

// Simple Google Analytics event tracker
const trackEvent = (eventName, params = {}) => {
  if (window.gtag) {
    window.gtag('event', eventName, params);
  }
};

const App = () => {
  // State management
  const [indices, setIndices] = useState([]);
  const [news, setNews] = useState([]);
  const [tradingIdeas, setTradingIdeas] = useState({
    ideas: [],
    grouped_ideas: {
      buy_signals: [],
      sell_signals: [],
      educational: []
    },
    summary: {
      total_ideas: 0,
      buy_signals_count: 0,
      sell_signals_count: 0,
      educational_count: 0
    }
  });

  // Phase 1 — Watchlist & Recently Viewed (localStorage-persisted)
  const { watchlist, addToWatchlist, removeFromWatchlist, isInWatchlist } = useWatchlist();
  const { recentlyViewed, addRecentlyViewed } = useRecentlyViewed();

  // Phase 1 — navigation tabs
  const [activeNav, setActiveNav] = useState('market');

  // Phase 1 — modals
  const [chartStock, setChartStock] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(() => shouldShowOnboarding());

  // UI state
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedArticles, setExpandedArticles] = useState({});
  const [selectedTab, setSelectedTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [showFilters, setShowFilters] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Error handling
  const [error, setError] = useState(null); // eslint-disable-line no-unused-vars

  // Phase 1 — track stock chart view
  const handleSelectStock = useCallback((stock) => {
    setChartStock(stock);
    addRecentlyViewed({ id: stock.symbol, title: `${stock.symbol} — ${stock.name}`, type: 'stock' });
    trackEvent('chart_view', { symbol: stock.symbol });
  }, [addRecentlyViewed]);

  // Phase 1 — track article view
  const handleArticleView = useCallback((item) => {
    addRecentlyViewed(item);
    trackEvent('news_view', { title: item.title });
  }, [addRecentlyViewed]);

  // Fetch functions
  const fetchIndices = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/indices/live`);
      if (!response.ok) throw new Error('Failed to fetch indices');
      const data = await response.json();
      
      const indicesArray = Object.entries(data.indices).map(([key, value]) => ({
        id: key,
        ...value
      }));
      setIndices(indicesArray);
    } catch (err) {
      console.error('Error fetching indices:', err);
      setError(prev => ({ ...prev, indices: err.message }));
    }
  }, []);

  const fetchNewsAndIdeas = useCallback(async () => {
    try {
      // Try the combined endpoint first
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/news-with-ideas`);
      if (response.ok) {
        const data = await response.json();
        setNews(data.news || []);
        setTradingIdeas({
          ideas: data.tradingview_ideas || [],
          grouped_ideas: data.grouped_ideas || {
            buy_signals: [],
            sell_signals: [],
            educational: []
          },
          summary: data.summary || {
            total_ideas: 0,
            buy_signals_count: 0,
            sell_signals_count: 0,
            educational_count: 0
          }
        });
      } else {
        // Fallback: fetch news and ideas separately
        const [newsRes, ideasRes] = await Promise.all([
          fetch(`${process.env.REACT_APP_API_URL}/api/news/summary`),
          fetch(`${process.env.REACT_APP_API_URL}/api/tradingview/ideas/enhanced`),
        ]);
        if (newsRes.ok) {
          const newsData = await newsRes.json();
          setNews(newsData.news || []);
        }
        if (ideasRes.ok) {
          const ideasData = await ideasRes.json();
          setTradingIdeas(prev => ({ ...prev, ideas: ideasData.ideas || [] }));
        }
      }
    } catch (err) {
      console.error('Error fetching news and ideas:', err);
      setError(prev => ({ ...prev, newsIdeas: err.message }));
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps


  // Initial data fetch
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchIndices(),
          fetchNewsAndIdeas(),
        ]);
        setLastUpdated(new Date());
      } catch (err) {
        console.error('Error initializing data:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeData();
  }, [fetchIndices, fetchNewsAndIdeas]);

  // Auto-refresh functionality
  useEffect(() => {
    const interval = setInterval(async () => {
      await fetchIndices();
      setLastUpdated(new Date());
    }, 60000); // Refresh indices every minute

    return () => clearInterval(interval);
  }, [fetchIndices]);

  // Manual refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        fetchIndices(),
        fetchNewsAndIdeas(),
        
      ]);
      setLastUpdated(new Date());
      addNotification('Data refreshed successfully', 'success');
    } catch (err) {
      addNotification('Failed to refresh data', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  // Utility functions
  const addNotification = (message, type = 'info') => {
    const notification = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date()
    };
    setNotifications(prev => [...prev, notification]);
    
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== notification.id));
    }, 5000);
  };

  const toggleSummary = (articleId) => {
    setExpandedArticles(prev => ({
      ...prev,
      [articleId]: !prev[articleId]
    }));
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '#10b981';
      case 'negative': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return <TrendingUp size={16} />;
      case 'negative': return <TrendingDown size={16} />;
      default: return <Minus size={16} />;
    }
  };

  const getSignalColor = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'BUY': return '#10b981';
      case 'SELL': return '#ef4444';
      default: return '#3b82f6';
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  const formatTime = (date) => {
    return new Intl.DateTimeFormat('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }).format(date);
  };

  // Filtered data
  const filteredNews = news.filter(article => {
    const matchesSearch = article.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         article.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         article.summary?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSentiment = sentimentFilter === 'all' || article.sentiment?.label === sentimentFilter;
    return matchesSearch && matchesSentiment;
  });

  const filteredIdeas = tradingIdeas.ideas.filter(idea => {
    if (selectedTab === 'all') return true;
    return idea.signal_label?.toLowerCase().includes(selectedTab.toLowerCase());
  });

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner">
          <Activity size={48} />
          <p>Loading market data...</p>
        </div>
      </div>
    );
  }

  // Nav tab definitions
  const NAV_TABS = [
    { id: 'market', label: 'Market Overview', icon: '🌐' },
    { id: 'watchlist', label: `Watchlist${watchlist.length > 0 ? ` (${watchlist.length})` : ''}`, icon: '⭐' },
    { id: 'personalized', label: 'Personalized News', icon: '📰' },
    { id: 'trending', label: 'Trending & Analytics', icon: '🔥' },
  ];

  return (
    <div className="app-container">
      {/* Onboarding Modal */}
      {showOnboarding && (
        <OnboardingModal onComplete={() => setShowOnboarding(false)} />
      )}

      {/* Stock Chart Modal */}
      {chartStock && (
        <StockChart stock={chartStock} onClose={() => setChartStock(null)} />
      )}

      {/* Feedback Modal */}
      {showFeedback && (
        <FeedbackForm onClose={() => setShowFeedback(false)} />
      )}

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="notifications-container">
          {notifications.map(notification => (
            <div key={notification.id} className={`notification ${notification.type}`}>
              <span>{notification.message}</span>
              <button onClick={() => setNotifications(prev => prev.filter(n => n.id !== notification.id))}>
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1 className="main-title">
              📈 Market Insights Dashboard
            </h1>
            <p className="subtitle">
              Real-time indices, news sentiment, watchlist &amp; personalized analytics
            </p>
          </div>
          <div className="header-right">
            <div className="header-actions">
              {lastUpdated && (
                <div className="last-updated">
                  <Clock size={16} />
                  <span>Last updated: {formatTime(lastUpdated)}</span>
                </div>
              )}
              <div className="header-btn-group">
                <button
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="refresh-btn"
                >
                  <RefreshCw size={16} className={refreshing ? 'spinning' : ''} />
                  {refreshing ? 'Refreshing...' : 'Refresh'}
                </button>
                <button
                  className="btn btn-secondary btn-sm feedback-trigger-btn"
                  onClick={() => setShowFeedback(true)}
                  title="Share feedback"
                >
                  <MessageSquare size={16} />
                  Feedback
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="main-nav">
          {NAV_TABS.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeNav === tab.id ? 'active' : ''}`}
              onClick={() => {
                setActiveNav(tab.id);
                trackEvent('tab_switch', { tab: tab.id });
              }}
            >
              <span className="nav-tab-icon">{tab.icon}</span>
              <span className="nav-tab-label">{tab.label}</span>
            </button>
          ))}
        </nav>
      </header>

      {/* ===== MARKET OVERVIEW TAB ===== */}
      {activeNav === 'market' && (
        <>
          {/* Market Indices */}
          <section className="indices-section">
            <h2 className="section-title">
              <Globe size={20} />
              Live Market Indices
            </h2>
            <div className="indices-grid">
              {indices.map((index) => (
                <div
                  key={index.id}
                  className="index-card"
                  onClick={() => handleSelectStock({ symbol: index.id, name: index.name, exchange: 'NSE' })}
                  style={{ cursor: 'pointer' }}
                  title={`Click to view ${index.name} chart`}
                >
                  <div className="index-header">
                    <h3 className="index-name">{index.name}</h3>
                    <div className="index-status">
                      <span className={`status-indicator ${index.market_status?.toLowerCase()}`}>
                        {index.market_status}
                      </span>
                    </div>
                  </div>
                  <div className="index-price">
                    <span className="price-value">
                      {formatPrice(index.current_price)}
                    </span>
                    <span className="currency">{index.currency}</span>
                  </div>
                  <div className={`index-change ${parseFloat(index.change) >= 0 ? 'positive' : 'negative'}`}>
                    {parseFloat(index.change) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                    <span>
                      {index.change} ({index.change_percent}%)
                    </span>
                  </div>
                  <div className="index-chart-hint">📊 Click to view chart</div>
                </div>
              ))}
            </div>
          </section>

          {/* News Section */}
          <section className="news-section">
            <div className="section-header">
              <h2 className="section-title">
                📰 Market News &amp; Analysis
              </h2>
              <div className="section-controls">
                <div className="search-container">
                  <Search size={16} />
                  <input
                    type="text"
                    placeholder="Search news..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="search-input"
                  />
                </div>
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="filter-toggle"
                >
                  <Filter size={16} />
                  Filters
                </button>
              </div>
            </div>

            {showFilters && (
              <div className="filters-panel">
                <div className="filter-group">
                  <label>Sentiment:</label>
                  <select
                    value={sentimentFilter}
                    onChange={(e) => setSentimentFilter(e.target.value)}
                    className="filter-select"
                  >
                    <option value="all">All Sentiments</option>
                    <option value="positive">Positive</option>
                    <option value="negative">Negative</option>
                    <option value="neutral">Neutral</option>
                  </select>
                </div>
              </div>
            )}

            <div className="news-list">
              {filteredNews.length === 0 ? (
                <div className="empty-state">
                  <AlertCircle size={48} />
                  <p>No news articles found matching your criteria.</p>
                </div>
              ) : (
                filteredNews.map((article, index) => {
                  const articleId = article.article_id || index;

                  return (
                    <div
                      key={articleId}
                      className="news-card"
                      onClick={() => handleArticleView({ id: articleId, title: article.title, type: 'news' })}
                    >
                      <div className="article-header">
                        <h3 className="article-title">
                          <a href={article.link} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()}>
                            {article.title}
                          </a>
                        </h3>
                        <div className="article-meta">
                          <div className="meta-item">
                            <Tag size={14} />
                            <span>{article.source}</span>
                          </div>
                          {article.author && (
                            <div className="meta-item">
                              <User size={14} />
                              <span>{article.author}</span>
                            </div>
                          )}
                          {article.publishedAt && (
                            <div className="meta-item">
                              <Clock size={14} />
                              <span>{article.publishedAt}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {article.sentiment && (
                        <div
                          className="sentiment-badge"
                          style={{
                            backgroundColor: getSentimentColor(article.sentiment.label) + '20',
                            color: getSentimentColor(article.sentiment.label),
                          }}
                        >
                          {getSentimentIcon(article.sentiment.label)}
                          <span>
                            {article.sentiment.label.charAt(0).toUpperCase() + article.sentiment.label.slice(1)}
                            {article.sentiment.score && ` (${article.sentiment.score})`}
                          </span>
                        </div>
                      )}

                      {(article.stocks || article.stocks_with_sentiment) && (
                        <div className="stocks-section">
                          <h4 className="stocks-title">
                            <DollarSign size={16} />
                            Stocks Mentioned:
                          </h4>
                          <div className="stocks-list">
                            {(article.stocks_with_sentiment || article.stocks || []).map((stock, idx) => (
                              <div
                                key={idx}
                                className="stock-badge"
                                style={{
                                  backgroundColor: stock.sentiment_label ?
                                    getSentimentColor(stock.sentiment_label) + '15' : '#f3f4f6',
                                  borderColor: stock.sentiment_label ?
                                    getSentimentColor(stock.sentiment_label) + '40' : '#d1d5db',
                                  cursor: 'pointer'
                                }}
                                onClick={e => {
                                  e.stopPropagation();
                                  handleSelectStock({ symbol: stock.symbol, name: stock.name || stock.symbol, exchange: 'NSE' });
                                }}
                                title={`View ${stock.symbol} chart`}
                              >
                                <span className="stock-symbol">{stock.symbol}</span>
                                {stock.sentiment_label && (
                                  <div
                                    className="stock-sentiment"
                                    style={{ color: getSentimentColor(stock.sentiment_label) }}
                                  >
                                    {getSentimentIcon(stock.sentiment_label)}
                                    <span>({stock.sentiment})</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {article.summary && (
                        <div className="article-summary">
                          <p>{article.summary}</p>
                        </div>
                      )}

                      <div className="article-actions">
                        <button
                          onClick={e => { e.stopPropagation(); toggleSummary(articleId); }}
                          className="btn btn-secondary"
                        >
                          <Eye size={16} />
                          {expandedArticles[articleId] ? 'Hide Details' : 'Show Details'}
                        </button>
                        <a
                          href={article.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn btn-primary"
                          onClick={e => e.stopPropagation()}
                        >
                          <ExternalLink size={16} />
                          Read Full Article
                        </a>
                      </div>

                      {expandedArticles[articleId] && (
                        <div className="expanded-content">
                          <div className="article-description">
                            <h4>Description:</h4>
                            <p>{article.description || 'No description available'}</p>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </section>

          {/* Trading Ideas Section */}
          <section className="trading-ideas-section">
            <div className="section-header">
              <h2 className="section-title">
                💡 TradingView Ideas
              </h2>
              <div className="tab-buttons">
                {['all', 'buy', 'sell', 'educational'].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setSelectedTab(tab)}
                    className={`tab-btn ${selectedTab === tab ? 'active' : ''}`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                    {tab === 'buy' && tradingIdeas.summary.buy_signals_count > 0 && (
                      <span className="tab-count">{tradingIdeas.summary.buy_signals_count}</span>
                    )}
                    {tab === 'sell' && tradingIdeas.summary.sell_signals_count > 0 && (
                      <span className="tab-count">{tradingIdeas.summary.sell_signals_count}</span>
                    )}
                    {tab === 'educational' && tradingIdeas.summary.educational_count > 0 && (
                      <span className="tab-count">{tradingIdeas.summary.educational_count}</span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className="ideas-grid">
              {filteredIdeas.length === 0 ? (
                <div className="empty-state">
                  <AlertCircle size={48} />
                  <p>No trading ideas found for the selected category.</p>
                </div>
              ) : (
                filteredIdeas.map((idea, index) => (
                  <div
                    key={index}
                    className="idea-card"
                    onClick={() => handleSelectStock({ symbol: idea.stock_symbol, name: idea.title, exchange: 'NSE' })}
                    style={{ cursor: 'pointer' }}
                    title={`Click to view ${idea.stock_symbol} chart`}
                  >
                    <div className="idea-header">
                      <div className="idea-info">
                        <h3 className="idea-symbol">{idea.stock_symbol}</h3>
                        <p className="idea-title">{idea.title}</p>
                      </div>
                      <div
                        className="signal-badge"
                        style={{
                          backgroundColor: getSignalColor(idea.signal_label) + '20',
                          color: getSignalColor(idea.signal_label),
                        }}
                      >
                        {idea.signal_label}
                      </div>
                    </div>

                    {idea.condition && (
                      <div className="idea-condition">
                        <BarChart3 size={16} />
                        <span>{idea.condition}</span>
                      </div>
                    )}

                    <div className="idea-footer">
                      <a
                        href={idea.link?.startsWith('http') ? idea.link : `https://in.tradingview.com${idea.link}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-primary btn-sm"
                        onClick={e => e.stopPropagation()}
                      >
                        <Target size={14} />
                        View Idea
                      </a>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </>
      )}

      {/* ===== WATCHLIST TAB ===== */}
      {activeNav === 'watchlist' && (
        <Watchlist
          watchlist={watchlist}
          onAdd={(stock) => {
            addToWatchlist(stock);
            trackEvent('watchlist_add', { symbol: stock.symbol });
          }}
          onRemove={(symbol) => {
            removeFromWatchlist(symbol);
            trackEvent('watchlist_remove', { symbol });
          }}
          isInWatchlist={isInWatchlist}
          onSelectStock={handleSelectStock}
        />
      )}

      {/* ===== PERSONALIZED NEWS TAB ===== */}
      {activeNav === 'personalized' && (
        <PersonalizedNews
          news={news}
          watchlist={watchlist}
          onArticleView={handleArticleView}
        />
      )}

      {/* ===== TRENDING & ANALYTICS TAB ===== */}
      {activeNav === 'trending' && (
        <TrendingSection
          news={news}
          recentlyViewed={recentlyViewed}
          onSelectStock={handleSelectStock}
        />
      )}

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p className="footer-main">
            📊 Real-time market data and sentiment analysis powered by AI
          </p>
          <p className="footer-sources">
            Data sources: MoneyControl, Economic Times, TradingView
          </p>
          <p className="footer-tech">
            🚀 Built with React &amp; Flask · Phase 1 SaaS Features Active
          </p>
          <p className="footer-privacy">
            💾 Your watchlist &amp; preferences are saved locally in your browser — no account required.
          </p>
          <button
            className="btn btn-secondary btn-sm footer-feedback-btn"
            onClick={() => setShowFeedback(true)}
          >
            <MessageSquare size={14} />
            Share Feedback
          </button>
        </div>
      </footer>
    </div>
  );
};

export default App;