import { useEffect, useRef, useState } from 'react'

import { getJson } from './api.ts'

type JobStatus =
  | 'starting'
  | 'running'
  | 'stopping'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

type JobSnapshot = { id: string; status: JobStatus }
type OutputEvent = { sequence: number }

type JobCallbacks<TJob, TOutput> = {
  onRestore?: (job: TJob | null) => void
  onOutput: (output: TOutput) => void
  onComplete?: (job: TJob) => void
}

export function useJobLifecycle<TJob extends JobSnapshot, TOutput extends OutputEvent>(
  jobsPath: string,
  label: string,
  callbacks: JobCallbacks<TJob, TOutput>,
) {
  const [job, setJob] = useState<TJob | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(true)
  const [restoreError, setRestoreError] = useState<Error | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const callbacksRef = useRef(callbacks)

  useEffect(() => {
    callbacksRef.current = callbacks
  })

  useEffect(() => {
    let cancelled = false
    getJson<{ job: TJob | null }>(`${jobsPath}/current`)
      .then(({ job: currentJob }) => {
        if (cancelled) return
        setJob(currentJob)
        if (
          currentJob?.status === 'starting' ||
          currentJob?.status === 'running' ||
          currentJob?.status === 'stopping'
        ) {
          setActiveJobId(currentJob.id)
        }
        callbacksRef.current.onRestore?.(currentJob)
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRestoreError(
            error instanceof Error ? error : new Error(`Could not restore ${label} job`),
          )
        }
      })
      .finally(() => {
        if (!cancelled) setRestoring(false)
      })
    return () => {
      cancelled = true
    }
  }, [jobsPath, label])

  useEffect(() => {
    if (activeJobId === null) return

    const source = new EventSource(`${jobsPath}/${encodeURIComponent(activeJobId)}/events`)
    const receivedSequences = new Set<number>()
    let completeReceived = false
    let terminalStateReceived = false

    source.addEventListener('state', (event: MessageEvent<string>) => {
      const nextJob = JSON.parse(event.data) as TJob
      terminalStateReceived =
        nextJob.status === 'succeeded' ||
        nextJob.status === 'failed' ||
        nextJob.status === 'cancelled'
      setStreamError(null)
      setJob(nextJob)
    })
    source.addEventListener('output', (event: MessageEvent<string>) => {
      const output = JSON.parse(event.data) as TOutput
      if (receivedSequences.has(output.sequence)) return
      receivedSequences.add(output.sequence)
      setStreamError(null)
      callbacksRef.current.onOutput(output)
    })
    source.addEventListener('complete', (event: MessageEvent<string>) => {
      const completedJob = JSON.parse(event.data) as TJob
      completeReceived = true
      setStreamError(null)
      setJob(completedJob)
      setActiveJobId(null)
      source.close()
      callbacksRef.current.onComplete?.(completedJob)
    })
    source.onerror = () => {
      if (!completeReceived && !terminalStateReceived) {
        setStreamError(`The ${label} progress connection was interrupted.`)
      }
    }

    return () => {
      completeReceived = true
      source.close()
    }
  }, [activeJobId, jobsPath, label])

  return {
    job,
    setJob,
    activeJobId,
    setActiveJobId,
    restoring,
    restoreError,
    streamError,
    setStreamError,
  }
}
