# 📊 Market Pulse - Real-time Market News & Sentiment Analysis

A modern web application that provides real-time market news with sentiment analysis, market indices, and intelligent stock information extraction.

## ✨ Features

### 🎯 Core Features
- **Real-time News Scraping**: Automatically fetches latest market news from MoneyControl
- **Sentiment Analysis**: AI-powered sentiment analysis for market news headlines
- **Market Indices**: Live display of NIFTY 50 and SENSEX with real-time changes
- **Stock Information**: Automatic extraction and display of relevant stock data
- **Search Functionality**: Search news by keywords with instant results
- **Responsive Design**: Beautiful, modern UI that works on all devices

### 🚀 Advanced Features
- **Caching System**: Redis-based caching for improved performance
- **Rate Limiting**: API protection against abuse
- **Error Handling**: Robust error handling and user feedback
- **Market Summary**: Visual sentiment overview with percentages
- **Auto-refresh**: Manual refresh with loading states
- **Health Monitoring**: API health check endpoints

## 🛠️ Tech Stack

### Backend
- **Flask**: Python web framework
- **BeautifulSoup4**: Web scraping
- **Redis**: Caching and session management
- **Requests**: HTTP client
- **Pytz**: Timezone handling

### Frontend
- **React 19**: Modern React with hooks
- **CSS3**: Custom styling with gradients and animations
- **Responsive Design**: Mobile-first approach

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis (optional, for caching)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Redis (optional):**
   ```bash
   # On Windows (with WSL or Docker)
   redis-server
   
   # On macOS
   brew services start redis
   
   # On Linux
   sudo systemctl start redis
   ```

5. **Run the Flask server:**
   ```bash
   python app.py
   ```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## 🔧 Configuration

### Environment Variables (Backend)

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
REDIS_URL=redis://localhost:6379/0
CACHE_DURATION=300
RATE_LIMIT_PER_MINUTE=60
```

### API Endpoints

- `GET /api/health` - Health check
- `GET /api/news` - Get latest news (cached)
- `GET /api/news/search?q=keyword` - Search news by keyword
- `GET /api/market-summary` - Get market indices and sector data

## 🎨 Features in Detail

### Sentiment Analysis
The app uses a sophisticated sentiment analysis algorithm that:
- Analyzes news headlines for positive/negative keywords
- Provides weighted scoring for more accurate results
- Categorizes news as Bullish, Bearish, or Neutral
- Shows sentiment percentages in the market summary

### Stock Information Extraction
Automatically detects and displays:
- Stock prices and symbols
- Price changes and percentages
- Visual indicators (up/down arrows)
- Color-coded positive/negative changes

### Search Functionality
- Real-time search through news articles
- Instant results with loading states
- Clear search option
- Search result count display

### Market Indices
- Live NIFTY 50 and SENSEX data
- Real-time price changes
- Visual trend indicators
- Responsive grid layout

## 🚀 Performance Optimizations

- **Redis Caching**: 5-minute cache for news, 10-minute for market data
- **Rate Limiting**: 60 requests per minute per IP
- **Lazy Loading**: Efficient data fetching
- **Optimized Images**: Web-optimized assets
- **CDN Ready**: Static assets optimized for CDN

## 🔒 Security Features

- **CORS Configuration**: Proper cross-origin resource sharing
- **Rate Limiting**: Protection against API abuse
- **Input Validation**: Sanitized search queries
- **Error Handling**: Secure error messages

## 📱 Responsive Design

The application is fully responsive with:
- Mobile-first design approach
- Touch-friendly interfaces
- Optimized layouts for tablets and desktops
- Flexible grid systems

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 🚀 Deployment

### Backend Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t market-pulse-backend .
docker run -p 5000:5000 market-pulse-backend
```

### Frontend Deployment
```bash
cd frontend
npm run build
# Deploy the build folder to your hosting service
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MoneyControl**: News source
- **React Community**: Amazing framework
- **Flask Community**: Python web framework
- **Open Source Contributors**: All the amazing libraries used

## 📞 Support

For support, email support@marketpulse.com or create an issue in the repository.

---

**Made with ❤️ and React** 