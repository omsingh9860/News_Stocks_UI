import React from 'react';

export const SkeletonText = ({ className = '' }) => (
  <div className={`skeleton skeleton-line ${className}`} />
);

export const SkeletonIndexCard = () => (
  <div className="skeleton skeleton-card index-card" style={{ minHeight: 120 }}>
    <div className="skeleton skeleton-line medium" style={{ marginBottom: 12 }} />
    <div className="skeleton skeleton-line short" style={{ marginBottom: 8 }} />
    <div className="skeleton skeleton-line long" />
  </div>
);

export const SkeletonNewsCard = () => (
  <div className="skeleton skeleton-card news-card" style={{ minHeight: 140 }}>
    <div className="skeleton skeleton-line long" style={{ marginBottom: 10 }} />
    <div className="skeleton skeleton-line medium" style={{ marginBottom: 8 }} />
    <div className="skeleton skeleton-line short" style={{ marginBottom: 8 }} />
    <div className="skeleton skeleton-line long" />
  </div>
);

export const SkeletonCard = ({ children }) => (
  <div className="skeleton skeleton-card">{children}</div>
);

const SkeletonLoader = { SkeletonCard, SkeletonNewsCard, SkeletonIndexCard, SkeletonText };
export default SkeletonLoader;
