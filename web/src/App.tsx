import { Routes, Route, Link } from 'react-router-dom'
import PredictPage from './pages/PredictPage'
import AboutPage from './pages/AboutPage'
import ReportsPage from './pages/ReportsPage'

function App() {
    return (
        <div className="app">
            <header className="header">
                <div className="header-content">
                    <h1 className="logo">
                        <svg className="logo-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="32" cy="32" r="30" fill="#dc2626" />
                            <path d="M32 52l-2.1-1.9C18.5 39.8 12 34 12 26.5 12 20.4 16.8 16 23 16c3.7 0 7.2 1.7 9 4.4C33.8 17.7 37.3 16 41 16c6.2 0 11 4.4 11 10.5 0 7.5-6.5 13.3-17.9 23.6L32 52z" fill="white" />
                            <path d="M8 32h12l3-8 4 16 4-12 3 6h22" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                        </svg>
                        <span className="logo-text">Heart Disease <span className="logo-subtitle">Prediction & Analysis</span></span>
                    </h1>
                    <nav className="nav">
                        <Link to="/" className="nav-link">Predict</Link>
                        <Link to="/reports" className="nav-link">Reports</Link>
                        <Link to="/about" className="nav-link">About</Link>
                    </nav>
                </div>
            </header>

            <main className="main">
                <Routes>
                    <Route path="/" element={<PredictPage />} />
                    <Route path="/reports" element={<ReportsPage />} />
                    <Route path="/about" element={<AboutPage />} />
                </Routes>
            </main>

            <footer className="footer">
                <p>Heart Disease Prediction &mdash; Data Mining Project</p>
                <p className="disclaimer">For educational purposes only. Not medical advice.</p>
            </footer>
        </div>
    )
}

export default App
