import { ChartBar } from '@phosphor-icons/react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Checkbox,
  Group,
  Loader,
  Paper,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { ApiError, deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { getModels } from '../models.ts'
import { joinOutputPath } from '../utils/output-path.ts'
import { parentPath } from '../utils/parent-path.ts'

type AuditFormValues = {
  source_path: string | null
  output_folder: string | null
  output_name: string
  device: 'cpu' | 'cuda'
  overwrite: boolean
}

type AuditResult = {
  source_path: string
  audit_path: string
  audited_layer_count: number
  skipped_tensor_count: number
  warning: string
}

type AuditProgress = {
  current: number
  total: number
  tensor_name: string
}

type AuditJobSnapshot = {
  id: string
  status: 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
  started_at: string
  completed_at: string | null
  progress: AuditProgress | null
  result: AuditResult | null
  error: string | null
}

type AuditOutputEvent = {
  sequence: number
  line: string
  progress: AuditProgress
}

type CurrentAuditJobResponse = {
  job: AuditJobSnapshot | null
}

type StartAuditJobResponse = {
  job: AuditJobSnapshot
}

type FoldersResponse = {
  folders: string[]
}

const DEVICE_OPTIONS = [
  { label: 'CPU', value: 'cpu' },
  { label: 'CUDA', value: 'cuda' },
]

function AuditPage() {
  const [job, setJob] = useState<AuditJobSnapshot | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [startPending, setStartPending] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [restoring, setRestoring] = useState(true)
  const [lines, setLines] = useState<string[]>([])
  const [requestError, setRequestError] = useState<Error | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const receivedSequences = useRef(new Set<number>())
  const outputViewportRef = useRef<HTMLDivElement>(null)
  const outputAtBottom = useRef(true)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })
  const outputFoldersQuery = useQuery({
    queryKey: ['output-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/output-folders'),
  })

  useEffect(() => {
    let cancelled = false
    getJson<CurrentAuditJobResponse>('/api/audit/jobs/current')
      .then(({ job: currentJob }) => {
        if (cancelled || currentJob === null) return
        setJob(currentJob)
        if (
          currentJob.status === 'starting' ||
          currentJob.status === 'running' ||
          currentJob.status === 'stopping'
        ) {
          setActiveJobId(currentJob.id)
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRequestError(
            error instanceof Error ? error : new Error('Could not restore Audit job'),
          )
        }
      })
      .finally(() => {
        if (!cancelled) setRestoring(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (activeJobId === null) return

    const source = new EventSource(
      `/api/audit/jobs/${encodeURIComponent(activeJobId)}/events`,
    )
    let completeReceived = false
    let terminalStateReceived = false
    receivedSequences.current.clear()

    const handleState = (event: MessageEvent<string>) => {
      const nextJob = JSON.parse(event.data) as AuditJobSnapshot
      terminalStateReceived =
        nextJob.status === 'succeeded' ||
        nextJob.status === 'failed' ||
        nextJob.status === 'cancelled'
      setStreamError(null)
      setJob(nextJob)
    }
    const handleOutput = (event: MessageEvent<string>) => {
      const output = JSON.parse(event.data) as AuditOutputEvent
      if (receivedSequences.current.has(output.sequence)) return
      receivedSequences.current.add(output.sequence)
      setStreamError(null)
      setLines((previous) => [...previous, output.line])
      setJob((previous) =>
        previous === null ? previous : { ...previous, progress: output.progress },
      )
    }
    const handleComplete = (event: MessageEvent<string>) => {
      const completedJob = JSON.parse(event.data) as AuditJobSnapshot
      completeReceived = true
      setStreamError(null)
      setJob(completedJob)
      setActiveJobId(null)
      source.close()
    }

    source.addEventListener('state', handleState)
    source.addEventListener('output', handleOutput)
    source.addEventListener('complete', handleComplete)
    source.onerror = () => {
      if (!completeReceived && !terminalStateReceived) {
        setStreamError('The Audit progress connection was interrupted.')
      }
    }

    return () => {
      completeReceived = true
      source.close()
    }
  }, [activeJobId])

  useEffect(() => {
    const viewport = outputViewportRef.current
    if (viewport !== null && outputAtBottom.current) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [lines])

  const form = useForm<AuditFormValues>({
    initialValues: {
      source_path: null,
      output_folder: null,
      output_name: '',
      device: 'cuda',
      overwrite: false,
    },
    validate: {
      source_path: (value) => (value ? null : 'Source checkpoint is required'),
      output_folder: (value) => (value ? null : 'Audit report folder is required'),
      output_name: (value) => {
        const name = value.trim()
        if (!name) return 'Audit report filename is required'
        if (name.includes('/') || name.includes('\\')) {
          return 'Enter a filename, not a path'
        }
        return null
      },
    },
  })

  const models = modelsQuery.data?.models ?? []
  const outputFolders = outputFoldersQuery.data?.folders ?? []
  const nameCounts = new Map<string, number>()
  for (const model of models) {
    const key = model.name.toLowerCase()
    nameCounts.set(key, (nameCounts.get(key) ?? 0) + 1)
  }
  const modelOptions = models.map((model) => ({
    label:
      (nameCounts.get(model.name.toLowerCase()) ?? 0) > 1
        ? `${model.name} — ${parentPath(model.path)}`
        : model.name,
    value: model.path,
  }))
  const outputFolderOptions = outputFolders.map((folder) => ({
    label: folder,
    value: folder,
  }))
  const isJobActive =
    job !== null &&
    activeJobId !== null &&
    (job.status === 'starting' || job.status === 'running' || job.status === 'stopping')
  const formDisabled = restoring || startPending || isJobActive
  const result = job?.status === 'succeeded' ? job.result : null
  const jobError = job?.status === 'failed' ? job.error : null
  const jobCancelled = job?.status === 'cancelled'
  const errorMessage = requestError?.message ?? jobError ?? streamError
  const showProgress = isJobActive || lines.length > 0
  const statusLabel =
    job?.status === 'starting'
      ? 'Starting audit'
        : job?.status === 'running'
          ? 'Auditing tensors'
          : job?.status === 'stopping'
            ? 'Stopping audit'
          : job?.status === 'succeeded'
            ? 'Audit complete'
            : job?.status === 'failed'
              ? 'Audit failed'
              : 'Audit output'

  const submit = async (values: AuditFormValues) => {
    if (values.output_folder === null) return
    setStartPending(true)
    setRequestError(null)
    setStreamError(null)
    setJob(null)
    setActiveJobId(null)
    setLines([])
    try {
      const response = await postJson<StartAuditJobResponse>('/api/audit/jobs', {
        source_path: values.source_path,
        output: joinOutputPath(values.output_folder, values.output_name.trim()),
        device: values.device,
        overwrite: values.overwrite,
      })
      setJob(response.job)
      setActiveJobId(response.job.id)
    } catch (error: unknown) {
      setRequestError(
        error instanceof Error ? error : new Error('Could not start Audit job'),
      )
    } finally {
      setStartPending(false)
    }
  }

  const stop = async () => {
    if (activeJobId === null) return
    setStopPending(true)
    try {
      const response = await deleteJson<StartAuditJobResponse>(
        `/api/audit/jobs/${encodeURIComponent(activeJobId)}`,
        {},
      )
      setJob(response.job)
      setRequestError(null)
    } catch (error: unknown) {
      setRequestError(error instanceof Error ? error : new Error('Could not stop Audit job'))
    } finally {
      setStopPending(false)
    }
  }

  return (
    <Stack gap="xl">
      <PageHeader
        title="Audit checkpoint"
        description="Measure supported weight reconstruction formats for a configured checkpoint."
      />

      <Card withBorder padding="lg" radius="md">
        <form onSubmit={form.onSubmit(submit)}>
          <Stack>
            <Group align="center" gap="sm">
              <ChartBar size={22} />
              <Stack gap={2}>
                <Title order={3}>Audit checkpoint</Title>
                <Text c="dimmed" size="sm">
                  Create a JSON weight-audit report.
                </Text>
              </Stack>
            </Group>

            <Select
              label="Source checkpoint"
              placeholder="Choose a configured checkpoint"
              data={modelOptions}
              disabled={formDisabled || modelsQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No safetensors checkpoints found"
              withAsterisk
              {...form.getInputProps('source_path')}
            />
            {modelsQuery.error ? (
              <Alert color="red">
                {modelsQuery.error instanceof Error
                  ? modelsQuery.error.message
                  : 'Could not load models'}
              </Alert>
            ) : !modelsQuery.isPending && models.length === 0 ? (
              <Text size="sm" c="dimmed">
                No safetensors checkpoints found. Configure a model folder in{' '}
                <Anchor component={Link} to="/settings">
                  Settings
                </Anchor>
                .
              </Text>
            ) : null}

            <Select
              label="Audit report folder"
              placeholder="Choose a configured output folder"
              data={outputFolderOptions}
              disabled={formDisabled || outputFoldersQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No output folders found"
              withAsterisk
              {...form.getInputProps('output_folder')}
            />
            {outputFoldersQuery.error ? (
              <Alert color="red">
                {outputFoldersQuery.error instanceof Error
                  ? outputFoldersQuery.error.message
                  : 'Could not load output folders'}
              </Alert>
            ) : !outputFoldersQuery.isPending && outputFolders.length === 0 ? (
              <Text size="sm" c="dimmed">
                No output folders found. Configure one in{' '}
                <Anchor component={Link} to="/settings">
                  Settings
                </Anchor>
                .
              </Text>
            ) : null}

            <TextInput
              label="Audit report filename"
              description=".json is added automatically when omitted"
              placeholder="weight-audit.json"
              disabled={formDisabled}
              withAsterisk
              {...form.getInputProps('output_name')}
            />

            <Select
              label="Device"
              data={DEVICE_OPTIONS}
              allowDeselect={false}
              disabled={formDisabled}
              {...form.getInputProps('device')}
            />
            <Text size="sm" c="dimmed">
              CUDA accelerates supported ConvRot INT8, W4A4, and packed ConvRot INT5/INT6 work;
              plain INT8/INT6 stay on CPU.
            </Text>

            <Checkbox
              label="Replace existing report"
              disabled={formDisabled}
              {...form.getInputProps('overwrite', { type: 'checkbox' })}
            />

            <Group justify="flex-end" mt="sm" w="100%">
              {isJobActive && (
                <Button
                  color="red"
                  variant="light"
                  loading={stopPending}
                  disabled={job?.status === 'stopping'}
                  onClick={stop}
                >
                  {job?.status === 'stopping' ? 'Stopping' : 'Stop'}
                </Button>
              )}
              <Button
                type="submit"
                loading={startPending || isJobActive}
                disabled={formDisabled}
                w={{ base: '100%', md: 'auto' }}
              >
                {startPending || isJobActive ? 'Auditing' : 'Audit'}
              </Button>
            </Group>
          </Stack>
        </form>

        {showProgress && (
          <Paper withBorder p="sm" mt="md">
            <Stack gap="xs">
              <Group gap="sm" aria-live="polite">
                {isJobActive && <Loader size="sm" />}
                <Text>{statusLabel}</Text>
              </Group>
              {job?.progress && (
                <>
                  <Text size="sm">
                    {job.progress.current} / {job.progress.total} tensors
                  </Text>
                  <Text size="sm" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
                    {job.progress.tensor_name}
                  </Text>
                  {job.progress.total > 0 && (
                    <Progress
                      value={Math.min(
                        100,
                        (job.progress.current / job.progress.total) * 100,
                      )}
                    />
                  )}
                </>
              )}
              <Text size="sm" fw={500} mt="xs">
                Audit output
              </Text>
              <ScrollArea
                h={220}
                type="auto"
                offsetScrollbars
                viewportRef={outputViewportRef}
                onScrollPositionChange={({ y }) => {
                  const viewport = outputViewportRef.current
                  outputAtBottom.current =
                    viewport !== null && y + viewport.clientHeight >= viewport.scrollHeight - 8
                }}
                aria-label="Audit output"
              >
                <Stack gap={2} p="xs">
                  {lines.length === 0 ? (
                    <Text size="sm" c="dimmed">
                      Waiting for the first tensor...
                    </Text>
                  ) : (
                    lines.map((line, index) => (
                      <Text
                        key={`${index}-${line}`}
                        size="sm"
                        style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}
                      >
                        {line}
                      </Text>
                    ))
                  )}
                </Stack>
              </ScrollArea>
            </Stack>
          </Paper>
        )}

        {errorMessage && (
          <Alert color="red" mt="md">
            {errorMessage}
            {requestError instanceof ApiError && requestError.status === 409 && (
              <Text mt="xs">
                That audit report already exists. Choose a different filename or enable
                replacement.
              </Text>
            )}
          </Alert>
        )}
        {jobCancelled && (
          <Alert color="yellow" mt="md">
            Audit stopped. No report was written.
          </Alert>
        )}
      </Card>

      {result && (
        <Paper withBorder p="lg" radius="md">
          <Stack gap="lg">
            <Title order={3}>Audit complete</Title>
            <Stack gap={4} style={{ minWidth: 0 }}>
              <Text c="dimmed" size="sm">
                Report path
              </Text>
              <Text
                component="div"
                title={result.audit_path}
                style={{
                  minWidth: 0,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  userSelect: 'text',
                }}
              >
                {result.audit_path}
              </Text>
            </Stack>
            <SimpleGrid cols={{ base: 1, xs: 2 }}>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">
                  Audited layers
                </Text>
                <Text fw={500}>{result.audited_layer_count}</Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">
                  Skipped tensors
                </Text>
                <Text fw={500}>{result.skipped_tensor_count}</Text>
              </Paper>
            </SimpleGrid>
            <Alert color="yellow">{result.warning}</Alert>
          </Stack>
        </Paper>
      )}
    </Stack>
  )
}

export const Route = createFileRoute('/audit')({ component: AuditPage })
