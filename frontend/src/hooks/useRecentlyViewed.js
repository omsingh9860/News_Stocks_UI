import { useState, useEffect, useCallback } from 'react';

const RECENTLY_VIEWED_KEY = 'recently_viewed_v1';
const MAX_ITEMS = 10;

export const useRecentlyViewed = () => {
  const [recentlyViewed, setRecentlyViewed] = useState(() => {
    try {
      const saved = localStorage.getItem(RECENTLY_VIEWED_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(RECENTLY_VIEWED_KEY, JSON.stringify(recentlyViewed));
    } catch (e) {
      console.warn('Failed to save recently viewed to localStorage', e);
    }
  }, [recentlyViewed]);

  const addRecentlyViewed = useCallback((item) => {
    setRecentlyViewed(prev => {
      const filtered = prev.filter(r => r.id !== item.id);
      return [{ ...item, viewedAt: new Date().toISOString() }, ...filtered].slice(0, MAX_ITEMS);
    });
  }, []);

  const clearRecentlyViewed = useCallback(() => setRecentlyViewed([]), []);

  return { recentlyViewed, addRecentlyViewed, clearRecentlyViewed };
};
