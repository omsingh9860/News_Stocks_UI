"""
Sentiment analysis and NLP summarization utilities.
"""
import re
import heapq
import logging
from collections import Counter

from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from app.config import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, INDIAN_STOCKS

logger = logging.getLogger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def analyze_sentiment(text):
    """Analyze sentiment of text and return a score in [-1, 1]."""
    if not text:
        return 0.0

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        text_lower = text.lower()
        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

        if positive_count + negative_count > 0:
            keyword_score = (positive_count - negative_count) / (positive_count + negative_count)
            final_score = (polarity * 0.4) + (keyword_score * 0.6)
        else:
            final_score = polarity

        return max(-1.0, min(1.0, final_score))

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return 0.0


def simple_summarize(text, max_sentences=3):
    """Simple extractive summarization using sentence scoring."""
    if not text or len(text.strip()) < 50:
        return text

    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text

        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w.isalpha() and w not in stop_words]

        if not words:
            return ' '.join(sentences[:max_sentences])

        word_freq = Counter(words)

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sent_words = word_tokenize(sentence.lower())
            sent_words = [w for w in sent_words if w.isalpha() and w not in stop_words]
            if sent_words:
                word_score = sum(word_freq.get(w, 0) for w in sent_words) / len(sent_words)
                position_score = 1.0 / (i + 1) * 0.1
                sentence_scores[sentence] = word_score + position_score

        if sentence_scores:
            top_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
            summary_sentences = [s for s in sentences if s in top_sentences]
            return ' '.join(summary_sentences)
        else:
            return ' '.join(sentences[:max_sentences])

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        try:
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:max_sentences])
        except Exception:
            return text[:500] + "..." if len(text) > 500 else text


def extract_stock_names_with_sentiment(text):
    """Extract stock names and their per-mention sentiment from text."""
    if not text:
        return []

    found_stocks = []
    text_lower = text.lower()
    overall_sentiment = analyze_sentiment(text)

    for stock in INDIAN_STOCKS:
        for alias in stock['aliases']:
            if alias.lower() in text_lower:
                pattern = re.compile(
                    r'.{0,150}' + re.escape(alias.lower()) + r'.{0,150}',
                    re.IGNORECASE,
                )
                matches = pattern.findall(text)

                if matches:
                    context_sentiment = analyze_sentiment(' '.join(matches))
                    final_sentiment = (
                        context_sentiment if abs(context_sentiment) > 0.15 else overall_sentiment
                    )

                    if final_sentiment > 0.1:
                        sentiment_label = 'positive'
                    elif final_sentiment < -0.1:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'

                    found_stocks.append({
                        'name': stock['name'],
                        'symbol': stock['symbol'],
                        'sentiment': round(final_sentiment, 3),
                        'sentiment_label': sentiment_label,
                    })
                    break

    # Deduplicate
    unique_stocks = []
    seen = set()
    for stock in found_stocks:
        if stock['symbol'] not in seen:
            unique_stocks.append(stock)
            seen.add(stock['symbol'])

    return unique_stocks


def extract_stock_names(text):
    """Legacy helper — returns stocks without sentiment scores."""
    stocks_with_sentiment = extract_stock_names_with_sentiment(text)
    return [{'name': s['name'], 'symbol': s['symbol']} for s in stocks_with_sentiment]
