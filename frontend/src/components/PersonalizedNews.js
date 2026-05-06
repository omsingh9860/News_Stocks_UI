import React, { useMemo } from 'react';
import { AlertCircle, ExternalLink, TrendingUp, TrendingDown, Minus, Clock, Tag, Star } from 'lucide-react';

const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case 'positive': return '#10b981';
    case 'negative': return '#ef4444';
    default: return '#6b7280';
  }
};

const getSentimentIcon = (sentiment) => {
  switch (sentiment) {
    case 'positive': return <TrendingUp size={14} />;
    case 'negative': return <TrendingDown size={14} />;
    default: return <Minus size={14} />;
  }
};

const PersonalizedNews = ({ news, watchlist, onArticleView }) => {
  const watchlistSymbols = useMemo(
    () => new Set(watchlist.map(s => s.symbol.toUpperCase())),
    [watchlist]
  );

  const personalizedNews = useMemo(() => {
    if (watchlist.length === 0) return [];
    return news.filter(article => {
      const stocksInArticle = article.stocks_with_sentiment || article.stocks || [];
      return stocksInArticle.some(s => watchlistSymbols.has((s.symbol || '').toUpperCase()));
    });
  }, [news, watchlist, watchlistSymbols]);

  if (watchlist.length === 0) {
    return (
      <section className="personalized-news-section">
        <h2 className="section-title">
          <Star size={20} />
          Personalized News Feed
        </h2>
        <div className="empty-state">
          <Star size={48} />
          <p>Add stocks to your watchlist to see personalized news here.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="personalized-news-section">
      <div className="section-header">
        <h2 className="section-title">
          <Star size={20} />
          Personalized News Feed
          {personalizedNews.length > 0 && (
            <span className="watchlist-count">{personalizedNews.length}</span>
          )}
        </h2>
        <p className="section-subtitle">
          News filtered for: {watchlist.map(s => s.symbol).join(', ')}
        </p>
      </div>

      {personalizedNews.length === 0 ? (
        <div className="empty-state">
          <AlertCircle size={48} />
          <p>No recent news found for your watchlist stocks.</p>
          <p className="watchlist-hint">Check back later or broaden your watchlist.</p>
        </div>
      ) : (
        <div className="news-list">
          {personalizedNews.map((article, idx) => {
            const articleId = article.article_id || idx;
            const relevantStocks = (article.stocks_with_sentiment || article.stocks || []).filter(
              s => watchlistSymbols.has((s.symbol || '').toUpperCase())
            );
            return (
              <div
                key={articleId}
                className="news-card personalized-card"
                onClick={() => onArticleView && onArticleView({ id: articleId, title: article.title, type: 'news' })}
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
                    </span>
                  </div>
                )}

                {relevantStocks.length > 0 && (
                  <div className="stocks-section">
                    <div className="stocks-list">
                      {relevantStocks.map((stock, i) => (
                        <div
                          key={i}
                          className="stock-badge watchlist-stock-badge"
                          style={{
                            backgroundColor: getSentimentColor(stock.sentiment_label) + '15',
                            borderColor: getSentimentColor(stock.sentiment_label) + '50',
                          }}
                        >
                          <span className="stock-symbol">⭐ {stock.symbol}</span>
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
                  <a
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-primary btn-sm"
                    onClick={e => e.stopPropagation()}
                  >
                    <ExternalLink size={14} />
                    Read Full Article
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};

export default PersonalizedNews;
