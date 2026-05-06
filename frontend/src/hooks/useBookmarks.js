import { useState, useEffect, useCallback } from 'react';

const BOOKMARKS_KEY = 'news_bookmarks_v1';

export const useBookmarks = () => {
  const [bookmarks, setBookmarks] = useState(() => {
    try {
      const saved = localStorage.getItem(BOOKMARKS_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks));
    } catch (e) {
      console.warn('Failed to save bookmarks to localStorage', e);
    }
  }, [bookmarks]);

  const addBookmark = useCallback((article) => {
    setBookmarks(prev => {
      if (prev.find(b => b.id === article.id)) return prev;
      return [
        ...prev,
        {
          id: article.id,
          title: article.title,
          link: article.link,
          source: article.source,
          publishedAt: article.publishedAt,
          sentiment: article.sentiment,
          bookmarkedAt: new Date().toISOString(),
        },
      ];
    });
  }, []);

  const removeBookmark = useCallback((id) => {
    setBookmarks(prev => prev.filter(b => b.id !== id));
  }, []);

  const isBookmarked = useCallback(
    (id) => bookmarks.some(b => b.id === id),
    [bookmarks]
  );

  const clearBookmarks = useCallback(() => setBookmarks([]), []);

  return { bookmarks, addBookmark, removeBookmark, isBookmarked, clearBookmarks };
};
