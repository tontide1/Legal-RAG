import { useEffect, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import client from '../api/client'

interface GraphProviderOption {
  value: string
  label: string
}

type StatusKind = 'idle' | 'loading' | 'saving' | 'success' | 'error'

export default function GraphProviderSettings() {
  const [options, setOptions] = useState<GraphProviderOption[]>([])
  const [savedProvider, setSavedProvider] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('')
  const [status, setStatus] = useState<StatusKind>('loading')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const clearStatusTimer = useRef<number | null>(null)
  const isMountedRef = useRef(true)

  const clearTransientState = () => {
    if (clearStatusTimer.current !== null) {
      window.clearTimeout(clearStatusTimer.current)
      clearStatusTimer.current = null
    }
  }

  const scheduleClearStatus = () => {
    clearTransientState()
    clearStatusTimer.current = window.setTimeout(() => {
      setStatus('idle')
      setMessage('')
      setError('')
      clearStatusTimer.current = null
    }, 2500)
  }

  useEffect(() => {
    isMountedRef.current = true

    const loadSettings = async () => {
      clearTransientState()
      setStatus('loading')
      setMessage('')
      setError('')

      try {
        const [providerResponse, optionsResponse] = await Promise.all([
          client.get('/settings/graph-provider'),
          client.get('/settings/graph-provider/options'),
        ])

        if (!isMountedRef.current) {
          return
        }

        const provider = providerResponse.data?.provider || ''
        const loadedOptions: GraphProviderOption[] = Array.isArray(optionsResponse.data?.options)
          ? optionsResponse.data.options
          : []

        setOptions(loadedOptions)
        if (loadedOptions.length === 0) {
          setSavedProvider('')
          setSelectedProvider('')
          setStatus('idle')
          setMessage('No providers available.')
          return
        }

        const fallbackProvider = loadedOptions[0]?.value || ''
        const nextProvider = loadedOptions.some((option) => option.value === provider) ? provider : fallbackProvider
        setSavedProvider(nextProvider)
        setSelectedProvider(nextProvider)
        setStatus('idle')
      } catch (requestError: any) {
        if (!isMountedRef.current) {
          return
        }

        setOptions([])
        setStatus('error')
        setError(
          requestError.response?.data?.detail ||
          requestError.message ||
          'Failed to load graph provider settings.'
        )
      }
    }

    void loadSettings()

    return () => {
      isMountedRef.current = false
      clearTransientState()
    }
  }, [reloadKey])

  const handleReloadProviders = () => {
    setReloadKey((current) => current + 1)
  }

  const handleSave = async () => {
    if (!selectedProvider || selectedProvider === savedProvider) {
      return
    }

    clearTransientState()
    setError('')
    setMessage('')
    setStatus('saving')

    const previousProvider = savedProvider

    if (selectedProvider === '9router') {
      setMessage('Checking 9router...')
    }

    try {
      const response = await client.put('/settings/graph-provider', {
        provider: selectedProvider,
      })

      const nextProvider = response.data?.provider || selectedProvider
      setSavedProvider(nextProvider)
      setSelectedProvider(nextProvider)
      setStatus('success')
      setMessage(response.data?.message || 'Graph provider saved.')
      scheduleClearStatus()
    } catch (requestError: any) {
      setSavedProvider(previousProvider)
      setSelectedProvider(previousProvider)
      setStatus('error')
      setMessage('')
      setError(
        requestError.response?.data?.detail ||
        requestError.message ||
        'Failed to save graph provider.'
      )
    }
  }

  const isDirty = selectedProvider !== savedProvider
  const isSaving = status === 'saving'

  return (
    <div className="space-y-3 rounded-xl border bg-muted/30 p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Graph Build Provider</p>
          <p className="text-[11px] text-muted-foreground mt-1">Select the provider used to build the graph.</p>
        </div>
      </div>

      <label className="block space-y-2">
        <span className="text-xs font-medium text-foreground">Graph Build Provider</span>
        <select
          value={selectedProvider}
          onChange={(event) => {
            setSelectedProvider(event.target.value)
            setError('')
            if (status === 'error' || status === 'success') {
              setStatus('idle')
              setMessage('')
            }
          }}
          disabled={status === 'loading' || isSaving || options.length === 0}
          className="w-full rounded-lg border bg-card px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status === 'loading' ? (
            <option value="">Loading options...</option>
          ) : status === 'error' ? (
            <option value="">Failed to load providers</option>
          ) : options.length === 0 ? (
            <option value="">No providers available</option>
          ) : (
            options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))
          )}
        </select>
        {status === 'error' && (
          <button
            type="button"
            onClick={handleReloadProviders}
            className="text-xs text-primary underline underline-offset-2"
          >
            Retry loading providers
          </button>
        )}
      </label>

      <button
        onClick={handleSave}
        disabled={status === 'loading' || isSaving || !isDirty || !selectedProvider}
        className="w-full rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground transition-all hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isSaving ? 'Saving...' : 'Save'}
      </button>

      {status === 'loading' && (
        <div className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2 text-[10px] font-medium text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>Loading graph provider settings...</span>
        </div>
      )}

      {status === 'saving' && (
        <div className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2 text-[10px] font-medium text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>{message || 'Saving graph provider...'}</span>
        </div>
      )}

      {status === 'success' && message && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 px-3 py-2 text-[10px] font-bold text-emerald-600">
          <CheckCircle2 className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'idle' && message && !error && (
        <div className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2 text-[10px] font-medium text-muted-foreground">
          <AlertCircle className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'error' && error && (
        <div className="flex items-center gap-2 rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-[10px] font-bold text-red-600">
          <AlertCircle className="h-3 w-3" />
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}
