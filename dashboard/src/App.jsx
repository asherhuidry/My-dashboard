import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar       from './components/Layout/Sidebar'
import TickerStrip   from './components/UI/TickerStrip'
import ChatWidget    from './components/Chat/ChatWidget'
import ErrorBoundary from './components/ErrorBoundary'
import Overview      from './pages/Overview'
import Analyzer      from './pages/Analyzer'
import Backtest      from './pages/Backtest'
import Research      from './pages/Research'
import NetworkPage   from './pages/Network'
import DatabasePage  from './pages/DatabasePage'
import EvolutionLog  from './pages/EvolutionLog'
import Screener      from './pages/Screener'
import Intelligence  from './pages/Intelligence'
import Experiments   from './pages/Experiments'
import Discoveries   from './pages/Discoveries'
import Sources       from './pages/Sources'

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
                  <Route path="/"           element={<Overview />}     />
                  <Route path="/analyze"    element={<Analyzer />}     />
                  <Route path="/research"   element={<Research />}     />
                  <Route path="/backtest"   element={<Backtest />}     />
                  <Route path="/network"    element={<NetworkPage />}  />
                  <Route path="/database"   element={<DatabasePage />} />
                  <Route path="/evolution"  element={<EvolutionLog />} />
                  <Route path="/screener"     element={<Screener />}      />
                  <Route path="/intelligence"  element={<Intelligence />} />
                  <Route path="/experiments"  element={<Experiments />}  />
                  <Route path="/discoveries"  element={<Discoveries />}  />
                  <Route path="/sources"      element={<Sources />}      />
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
