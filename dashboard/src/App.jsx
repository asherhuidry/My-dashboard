import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar       from './components/Layout/Sidebar'
import ErrorBoundary from './components/ErrorBoundary'

// ── Market Graph core ─────────────────────────────────────────────
import Overview      from './pages/Overview'
import Sources       from './pages/Sources'
import Discoveries   from './pages/Discoveries'
import NetworkPage   from './pages/Network'

// ── Secondary (accessible via URL, not in main nav) ─────────────
import Analyzer      from './pages/Analyzer'
import Research      from './pages/Research'
import Screener      from './pages/Screener'
import Backtest      from './pages/Backtest'
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
            <div className="flex-1 overflow-hidden">
              <ErrorBoundary>
                <Routes>
                  {/* Market Graph core */}
                  <Route path="/"             element={<Overview />}     />
                  <Route path="/network"      element={<NetworkPage />}  />
                  <Route path="/discoveries"  element={<Discoveries />}  />
                  <Route path="/sources"      element={<Sources />}      />

                  {/* Secondary — still accessible, not in sidebar */}
                  <Route path="/analyze"      element={<Analyzer />}     />
                  <Route path="/research"     element={<Research />}     />
                  <Route path="/screener"     element={<Screener />}     />
                  <Route path="/backtest"     element={<Backtest />}     />
                  <Route path="/database"     element={<DatabasePage />} />
                  <Route path="/evolution"    element={<EvolutionLog />} />
                </Routes>
              </ErrorBoundary>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
