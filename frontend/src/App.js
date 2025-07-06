import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, TrendingUp, TrendingDown, Minus, Clock, User, Tag, BarChart3, Eye, BookOpen, Target, AlertCircle, Bookmark, BookmarkX } from 'lucide-react';
import "./App.css";
const StockNewsApp = () => {
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
  const [marketData, setMarketData] = useState({
    nifty: { price: 0, change: 0 },
    banknifty: { price: 0, change: 0 },
    nasdaq: { price: 0, change: 0 },
    hangseng: { price: 0, change: 0 }
  });
  const [loading, setLoading] = useState(true);
  const [expandedArticles, setExpandedArticles] = useState({});
  const [showFullArticle, setShowFullArticle] = useState({});
  const [selectedTab, setSelectedTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [bookmarks, setBookmarks] = useState([]);

  useEffect(() => {
    const mockMarketData = {
      nifty: { price: 24350.85, change: 1.2 },
      banknifty: { price: 52847.30, change: -0.8 },
      nasdaq: { price: 19860.45, change: 0.5 },
      hangseng: { price: 19245.67, change: -1.1 }
    };
    setMarketData(mockMarketData);
  }, []);

  useEffect(() => {
    fetchNewsAndIdeas();
    fetchBookmarks();
  }, []);

  const fetchNewsAndIdeas = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/news-with-ideas');
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
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchBookmarks = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/bookmarks');
      const data = await response.json();
      setBookmarks(data.bookmarks || []);
    } catch (error) {
      console.error('Error fetching bookmarks:', error);
    }
  };

  const toggleSummary = (articleId) => {
    setExpandedArticles(prev => ({
      ...prev,
      [articleId]: !prev[articleId]
    }));
  };

  const showFullArticleContent = async (articleId, url) => {
    try {
      const response = await fetch(`/api/article?url=${encodeURIComponent(url)}`);
      const articleData = await response.json();
      setShowFullArticle(prev => ({
        ...prev,
        [articleId]: articleData
      }));
    } catch (error) {
      console.error('Error fetching full article:', error);
    }
  };

  const toggleBookmark = async (articleId, articleData) => {
    try {
      const response = await fetch('http://localhost:5000/api/bookmarks', {
        method: articleData.bookmarked ? 'DELETE' : 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          article_id: articleId,
          title: articleData.title,
          url: articleData.link,
          source: articleData.source,
          sentiment: articleData.sentiment?.label || 'neutral'
        })
      });
      const data = await response.json();
      if (response.ok) {
        fetchBookmarks();
      }
    } catch (error) {
      console.error('Error toggling bookmark:', error);
    }
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
    switch (signal) {
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

  const filteredNews = news.filter(article => {
    const matchesSearch = article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                        article.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSentiment = sentimentFilter === 'all' || article.sentiment?.label === sentimentFilter;
    return matchesSearch && matchesSentiment;
  });

  const filteredIdeas = tradingIdeas.ideas.filter(idea => {
    if (selectedTab === 'all') return true;
    return idea.signal_label.toLowerCase() === selectedTab;
  });

  if (loading) {
    return (
      <div className="loading-container">
        <div>Loading market data...</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1 className="main-title">
          📈 Stock Market News & Trading Ideas
        </h1>
        <p className="subtitle">
          Real-time market news with sentiment analysis and trading insights
        </p>
      </header>

      {/* Market Data */}
      <div className="market-data-grid">
        {Object.entries(marketData).map(([key, data]) => (
          <div key={key} className="market-card">
            <h3 className="market-index-name">
              {key.replace('nifty', 'NIFTY').replace('banknifty', 'BANK NIFTY')}
            </h3>
            <div className="market-price">
              {formatPrice(data.price)}
            </div>
            <div className={`market-change ${data.change >= 0 ? 'positive' : 'negative'}`}>
              {data.change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span className="change-text">
                {data.change >= 0 ? '+' : ''}{data.change}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* News Section */}
      <section className="news-section">
        <div className="section-header">
          <h2 className="section-title">
            📰 Latest Market News
          </h2>
          <div className="filters-container">
            <input
              type="text"
              placeholder="Search news..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <select
              value={sentimentFilter}
              onChange={(e) => setSentimentFilter(e.target.value)}
              className="sentiment-filter"
            >
              <option value="all">All Sentiments</option>
              <option value="positive">Positive</option>
              <option value="negative">Negative</option>
              <option value="neutral">Neutral</option>
            </select>
          </div>
        </div>
        <div className="news-list">
          {filteredNews.map((article, index) => {
            const isBookmarked = bookmarks.some(b => b.article_id === article.article_id);
            return (
              <div key={index} className="news-card">
                {/* Article Header */}
                <div className="article-header">
                  <h3 className="article-title">
                    {article.title}
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
                    <button
                      onClick={() => toggleBookmark(article.article_id, article)}
                      className={`bookmark-btn ${isBookmarked ? 'bookmarked' : ''}`}
                    >
                      {isBookmarked ? (
                        <>
                          <BookmarkX size={16} />
                          <span>Remove Bookmark</span>
                        </>
                      ) : (
                        <>
                          <Bookmark size={16} />
                          <span>Bookmark</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Overall Sentiment */}
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
                      ({article.sentiment.score})
                    </span>
                  </div>
                )}

                {/* Stock Mentions */}
                {article.stocks_with_sentiment && article.stocks_with_sentiment.length > 0 && (
                  <div className="stocks-section">
                    <h4 className="stocks-title">
                      📊 Stocks Mentioned:
                    </h4>
                    <div className="stocks-list">
                      {article.stocks_with_sentiment.map((stock, idx) => (
                        <div
                          key={idx}
                          className="stock-badge"
                          style={{
                            backgroundColor: getSentimentColor(stock.sentiment_label) + '15',
                            borderColor: getSentimentColor(stock.sentiment_label) + '40',
                          }}
                        >
                          <span className="stock-symbol">{stock.symbol}</span>
                          <div
                            className="stock-sentiment"
                            style={{
                              color: getSentimentColor(stock.sentiment_label)
                            }}
                          >
                            {getSentimentIcon(stock.sentiment_label)}
                            <span>({stock.sentiment})</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Article Summary */}
                {article.summary && (
                  <p className="article-summary">
                    {article.summary}
                  </p>
                )}

                {/* Action Buttons */}
                <div className="action-buttons">
                  <button
                    onClick={() => toggleSummary(article.article_id)}
                    className="btn btn-secondary"
                  >
                    <Eye size={16} />
                    {expandedArticles[article.article_id] ? 'Hide' : 'Show'} Summary
                  </button>
                  <a
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                  >
                    <ExternalLink size={16} />
                    Original Source
                  </a>
                </div>

                {/* Full Article Content */}
                {expandedArticles[article.article_id] && (
                  <div className="full-article-container">
                    <h4 className="full-article-title">
                      📖 Full Article:
                    </h4>
                    {showFullArticle[index] && showFullArticle[index].content ? (
                      <div className="full-article-content">
                        {showFullArticle[index].content.split('\n\n').map((paragraph, idx) => (
                          <p key={idx} className="article-paragraph">
                            {paragraph}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <p className="loading-text">
                        Loading full article content...
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
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
                {tab}
              </button>
            ))}
          </div>
          
        </div>
        <div className="ideas-grid">
          {filteredIdeas.map((idea, index) => (
            <div key={index} className="idea-card">
              <div className="idea-header">
                <div className="idea-info">
                  <h3 className="idea-symbol">
                    {idea.stock_symbol}
                  </h3>
                  <p className="idea-title">
                    {idea.title}
                  </p>
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
              <div className="idea-footer">
                <div className="idea-condition">
                  <BarChart3 size={16} />
                  <span>Condition: {idea.condition}</span>
                </div>
                <a
                  href={idea.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-primary btn-sm"
                >
                  <Target size={14} />
                  View Idea
                </a>
              </div>
            </div>
          ))}
          {filteredIdeas.length === 0 && (
            <div className="empty-state">
              <AlertCircle size={48} className="empty-icon" />
              <p>No trading ideas found for the selected category.</p>
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>📊 Real-time market data and sentiment analysis powered by AI</p>
        <p className="footer-sources">
          Data sources: MoneyControl, Economic Times, TradingView
        </p>
      </footer>
    </div>
  );
};

export default StockNewsApp;