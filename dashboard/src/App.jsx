import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar       from './components/Layout/Sidebar'
import TickerStrip   from './components/UI/TickerStrip'
import ChatWidget    from './components/Chat/ChatWidget'
import ErrorBoundary from './components/ErrorBoundary'

// ── Market Graph core ─────────────────────────────────────────────
import Overview      from './pages/Overview'
import Sources       from './pages/Sources'
import Discoveries   from './pages/Discoveries'
import NetworkPage   from './pages/Network'

// ── Research tools ────────────────────────────────────────────────
import Analyzer      from './pages/Analyzer'
import Research      from './pages/Research'
import Screener      from './pages/Screener'
import Backtest      from './pages/Backtest'

// ── System ────────────────────────────────────────────────────────
import DatabasePage  from './pages/DatabasePage'
import EvolutionLog  from './pages/EvolutionLog'

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } }
})

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="flex h-screen overflow-hidden bg-bg">
          <Sidebar />
          <main className="flex-1 ml-16 lg:ml-60 overflow-hidden flex flex-col">
            <TickerStrip />
            <div className="flex-1 overflow-hidden">
              <ErrorBoundary>
                <Routes>
                  {/* Market Graph core */}
                  <Route path="/"             element={<Overview />}     />
                  <Route path="/sources"      element={<Sources />}      />
                  <Route path="/discoveries"  element={<Discoveries />}  />
                  <Route path="/network"      element={<NetworkPage />}  />

                  {/* Research tools */}
                  <Route path="/analyze"      element={<Analyzer />}     />
                  <Route path="/research"     element={<Research />}     />
                  <Route path="/screener"     element={<Screener />}     />
                  <Route path="/backtest"     element={<Backtest />}     />

                  {/* System */}
                  <Route path="/database"     element={<DatabasePage />} />
                  <Route path="/evolution"    element={<EvolutionLog />} />
                </Routes>
              </ErrorBoundary>
            </div>
          </main>
        </div>
        <ChatWidget />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
