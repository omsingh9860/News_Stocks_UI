const BASE_URL = process.env.REACT_APP_API_URL || '';

/**
 * Fetches live market index data.
 * @returns {Promise<Object>} Object with `indices` map.
 */
export const fetchIndices = async () => {
  const response = await fetch(`${BASE_URL}/api/indices/live`);
  if (!response.ok) throw new Error('Failed to fetch indices');
  return response.json();
};

/**
 * Fetches news articles combined with TradingView ideas.
 * Falls back to separate endpoints if the combined one fails.
 * @returns {Promise<Object>} Object with `news`, `tradingview_ideas`, `grouped_ideas`, `summary`.
 */
export const fetchNewsWithIdeas = async () => {
  const response = await fetch(`${BASE_URL}/api/news-with-ideas`);
  if (response.ok) {
    return response.json();
  }

  // Fallback: fetch separately and merge
  const [newsRes, ideasRes] = await Promise.all([
    fetch(`${BASE_URL}/api/news/summary`),
    fetch(`${BASE_URL}/api/tradingview/ideas/enhanced`),
  ]);

  const result = {
    news: [],
    tradingview_ideas: [],
    grouped_ideas: { buy_signals: [], sell_signals: [], educational: [] },
    summary: { total_ideas: 0, buy_signals_count: 0, sell_signals_count: 0, educational_count: 0 },
  };

  if (newsRes.ok) {
    const newsData = await newsRes.json();
    result.news = newsData.news || [];
  }
  if (ideasRes.ok) {
    const ideasData = await ideasRes.json();
    result.tradingview_ideas = ideasData.ideas || [];
  }

  return result;
};

/**
 * Fetches the list of trending stocks.
 * @returns {Promise<Object>} Trending stocks payload.
 */
export const fetchTrendingStocks = async () => {
  const response = await fetch(`${BASE_URL}/api/stocks/trending`);
  if (!response.ok) throw new Error('Failed to fetch trending stocks');
  return response.json();
};

/**
 * Submits user feedback.
 * @param {Object} data - Feedback payload.
 * @returns {Promise<Object>} Server response.
 */
export const submitFeedback = async (data) => {
  const response = await fetch(`${BASE_URL}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('Failed to submit feedback');
  return response.json();
};
