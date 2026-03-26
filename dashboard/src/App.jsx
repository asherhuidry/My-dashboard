import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar      from './components/Layout/Sidebar'
import TickerStrip  from './components/UI/TickerStrip'
import ChatWidget   from './components/Chat/ChatWidget'
import Overview     from './pages/Overview'
import Analyzer     from './pages/Analyzer'
import NetworkPage  from './pages/Network'
import DatabasePage from './pages/DatabasePage'
import EvolutionLog from './pages/EvolutionLog'
import Screener     from './pages/Screener'

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } }
})

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="flex h-screen overflow-hidden bg-bg">
          <Sidebar />
          <main className="flex-1 ml-16 lg:ml-56 overflow-hidden flex flex-col">
            <TickerStrip />
            <div className="flex-1 overflow-hidden">
              <Routes>
                <Route path="/"          element={<Overview />}     />
                <Route path="/analyze"   element={<Analyzer />}     />
                <Route path="/network"   element={<NetworkPage />}  />
                <Route path="/database"  element={<DatabasePage />} />
                <Route path="/evolution" element={<EvolutionLog />} />
                <Route path="/screener"  element={<Screener />}     />
              </Routes>
            </div>
          </main>
        </div>
        <ChatWidget />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
