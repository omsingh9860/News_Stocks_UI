import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchNewsWithIdeas } from '../services/api';

const DEFAULT_POLL_INTERVAL = 300000; // 5 minutes

/**
 * Polls fetchNewsWithIdeas on a fixed interval.
 * @param {number} [pollInterval=300000] Interval in ms.
 * @returns {{ news: Array, tradingIdeas: Object, loading: boolean, error: string|null, lastUpdated: Date|null, refresh: Function }}
 */
export const useNewsPolling = (pollInterval = DEFAULT_POLL_INTERVAL) => {
  const [news, setNews] = useState([]);
  const [tradingIdeas, setTradingIdeas] = useState({
    ideas: [],
    grouped_ideas: { buy_signals: [], sell_signals: [], educational: [] },
    summary: { total_ideas: 0, buy_signals_count: 0, sell_signals_count: 0, educational_count: 0 },
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const intervalRef = useRef(null);

  const load = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true);
    setError(null);
    try {
      const data = await fetchNewsWithIdeas();
      setNews(data.news || []);
      setTradingIdeas({
        ideas: data.tradingview_ideas || [],
        grouped_ideas: data.grouped_ideas || { buy_signals: [], sell_signals: [], educational: [] },
        summary: data.summary || { total_ideas: 0, buy_signals_count: 0, sell_signals_count: 0, educational_count: 0 },
      });
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      if (showLoading) setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(true);
    intervalRef.current = setInterval(() => load(false), pollInterval);
    return () => clearInterval(intervalRef.current);
  }, [load, pollInterval]);

  const refresh = useCallback(() => load(true), [load]);

  return { news, tradingIdeas, loading, error, lastUpdated, refresh };
};
