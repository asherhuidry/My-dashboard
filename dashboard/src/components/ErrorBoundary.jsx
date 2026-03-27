import { Component } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
          <div className="w-12 h-12 rounded-xl bg-negative/10 border border-negative/30 flex items-center justify-center">
            <AlertTriangle size={20} className="text-negative" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-text mb-1">Something went wrong</p>
            <p className="text-xs text-text-muted max-w-xs">{this.state.error?.message}</p>
          </div>
          <button
            onClick={() => this.setState({ error: null })}
            className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border border-border hover:bg-bg-hover text-text-muted hover:text-text transition-colors"
          >
            <RefreshCw size={11} /> Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
