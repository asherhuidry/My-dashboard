export default function Spinner({ size = 'md', text }) {
  const sz = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' }[size]
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className={`${sz} border-2 border-border border-t-accent rounded-full animate-spin`} />
      {text && <p className="text-xs text-text-muted">{text}</p>}
    </div>
  )
}
