import { useState, useEffect, useCallback } from 'react';

const WATCHLIST_KEY = 'market_watchlist_v1';

export const useWatchlist = () => {
  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem(WATCHLIST_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(WATCHLIST_KEY, JSON.stringify(watchlist));
    } catch (e) {
      console.warn('Failed to save watchlist to localStorage', e);
    }
  }, [watchlist]);

  const addToWatchlist = useCallback((stock) => {
    setWatchlist(prev => {
      if (prev.find(s => s.symbol === stock.symbol)) return prev;
      return [...prev, { ...stock, addedAt: new Date().toISOString() }];
    });
  }, []);

  const removeFromWatchlist = useCallback((symbol) => {
    setWatchlist(prev => prev.filter(s => s.symbol !== symbol));
  }, []);

  const isInWatchlist = useCallback(
    (symbol) => watchlist.some(s => s.symbol === symbol),
    [watchlist]
  );

  const clearWatchlist = useCallback(() => setWatchlist([]), []);

  return { watchlist, addToWatchlist, removeFromWatchlist, isInWatchlist, clearWatchlist };
};
