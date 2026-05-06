import React, { useState } from 'react';
import { X, ChevronRight } from 'lucide-react';

const ONBOARDING_KEY = 'onboarding_completed_v1';

const STEPS = [
  {
    icon: '📈',
    title: 'Welcome to Market Insights Dashboard!',
    description:
      "Your all-in-one platform for real-time stock news, market indices, and personalized analytics — no account required.",
  },
  {
    icon: '⭐',
    title: 'Build Your Watchlist',
    description:
      'Add NSE stocks, US indices, and NASDAQ stocks to your personal watchlist. Everything is saved locally in your browser.',
  },
  {
    icon: '📰',
    title: 'Personalized News Feed',
    description:
      'Once you add stocks to your watchlist, the Personalized News tab will filter news articles specifically about those stocks.',
  },
  {
    icon: '🔥',
    title: 'Trending & Analytics',
    description:
      'See which stocks are being mentioned most in the latest news. The Trending tab tracks news volume and sentiment in real time.',
  },
  {
    icon: '🕯️',
    title: 'Candlestick Charts',
    description:
      'Click any stock in your watchlist or the trending section to open an interactive TradingView candlestick chart.',
  },
  {
    icon: '💬',
    title: 'Your Feedback Matters',
    description:
      "Use the Feedback button at the bottom of the page to share suggestions or report issues. We're building this for you!",
  },
];

const OnboardingModal = ({ onComplete }) => {
  const [step, setStep] = useState(0);

  const handleNext = () => {
    if (step < STEPS.length - 1) {
      setStep(s => s + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = () => {
    try {
      localStorage.setItem(ONBOARDING_KEY, 'true');
    } catch (e) {
      // ignore
    }
    onComplete && onComplete();
  };

  const current = STEPS[step];

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-modal">
        <button className="onboarding-skip" onClick={handleComplete}>
          <X size={16} />
          Skip tour
        </button>

        <div className="onboarding-step">
          <div className="onboarding-icon">{current.icon}</div>
          <h3 className="onboarding-title">{current.title}</h3>
          <p className="onboarding-desc">{current.description}</p>
        </div>

        {/* Progress dots */}
        <div className="onboarding-dots">
          {STEPS.map((_, i) => (
            <button
              key={i}
              className={`onboarding-dot ${i === step ? 'active' : ''}`}
              onClick={() => setStep(i)}
              aria-label={`Go to step ${i + 1}`}
            />
          ))}
        </div>

        <div className="onboarding-actions">
          {step > 0 && (
            <button className="btn btn-secondary" onClick={() => setStep(s => s - 1)}>
              Back
            </button>
          )}
          <button className="btn btn-primary onboarding-next-btn" onClick={handleNext}>
            {step < STEPS.length - 1 ? (
              <>
                Next <ChevronRight size={16} />
              </>
            ) : (
              "Get Started 🚀"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export const shouldShowOnboarding = () => {
  try {
    return !localStorage.getItem(ONBOARDING_KEY);
  } catch {
    return false;
  }
};

export default OnboardingModal;
