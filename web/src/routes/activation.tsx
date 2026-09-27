import { Pulse } from '@phosphor-icons/react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Checkbox,
  Divider,
  Group,
  Loader,
  Paper,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { ApiError, deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { getModels } from '../models.ts'
import { formatBytes } from '../utils/format-bytes.ts'
import { joinOutputPath } from '../utils/output-path.ts'
import { parentPath } from '../utils/parent-path.ts'
import { useJobLifecycle } from '../use-job-lifecycle.ts'

type FoldersResponse = {
  folders: string[]
}

type ArtifactOption = {
  folder: string
  name: string
  path: string
  version?: number
  collection_mode?: 'cvar' | 'full' | null
}

type ArtifactsResponse = {
  calibrations?: ArtifactOption[]
  caches?: ArtifactOption[]
}

type MethodOption = {
  label: string
  value: string
}

type MethodsResponse = {
  methods: MethodOption[]
  default_methods: string[]
}

type MetricsResponse = {
  metrics: MethodOption[]
  default_metric: string
}

type ActivationAuditFormValues = {
  source_path: string | null
  activation_calibration: string | null
  output_folder: string | null
  output_name: string
  device: 'cpu' | 'cuda'
  methods: string[]
  overwrite: boolean
}

type ActivationAuditResult = {
  source_path: string
  activation_calibration: string
  cache_metadata_path: string
  cache_tensors_path: string
  method_count: number
  layer_count: number
  warning: string
}

type ActivationProgress = {
  current: number
  total: number
  tensor_name: string
}

type ActivationJobSnapshot = {
  id: string
  status: 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
  started_at: string
  completed_at: string | null
  progress: ActivationProgress | null
  result: ActivationAuditResult | null
  error: string | null
}

type ActivationOutputEvent = {
  sequence: number
  line: string
  progress: ActivationProgress
}

type ScoreReport = {
  summary: {
    layer_count: number
    candidate_count: number
    available_candidate_count: number
    unavailable_candidate_count: number
  }
  results: Array<{
    tensor_name: string
    method: string
    action: string
    storage_bytes: number | null
    available: boolean
    objective_cost: number | null
    status: string
  }>
}

type RankedEntry = {
  tensor_name: string
  method?: string
  value: number
  metric: string
}

type InspectReport = {
  summary: {
    layer_count: number
    candidate_count: number
    available_candidate_count: number
    unavailable_candidate_count: number
  }
  rankings: Record<string, RankedEntry[]>
}

type OptimizeResult = {
  audit_cache: string
  profile_path: string | null
  summary_path: string | null
  profile_id: string
  metric: string
  target_bytes: number | null
  promotion_budget_bytes: number | null
  estimated_output_bytes: number
  dry_run: boolean
  warning: string
}

type OptimizeFormValues = {
  audit_cache: string | null
  activation_calibration: string | null
  metric: string
  budget_mode: 'target' | 'promotion'
  budget_value: string
  output_folder: string | null
  output_name: string
  profile_id: string
  summary_name: string
  baseline_method: string
  overwrite: boolean
  dry_run: boolean
}

const DEVICE_OPTIONS = [
  { label: 'CPU', value: 'cpu' },
  { label: 'CUDA', value: 'cuda' },
]

function groupedArtifactOptions(items: ArtifactOption[]) {
  const groups = new Map<string, { label: string; value: string }[]>()
  for (const item of items) {
    const group = groups.get(item.folder) ?? []
    const calibrationVersion =
      item.version === undefined
        ? ''
        : ` (v${item.version}${item.collection_mode ? ` ${item.collection_mode}` : ''})`
    group.push({ label: `${item.name}${calibrationVersion}`, value: item.path })
    groups.set(item.folder, group)
  }
  return Array.from(groups, ([group, options]) => ({ group, items: options }))
}

function formatMetric(value: number | null) {
  return value === null ? 'Unavailable' : value.toPrecision(6)
}

function ActivationAuditPanel() {
  const queryClient = useQueryClient()
  const {
    job,
    setJob,
    activeJobId,
    setActiveJobId,
    restoring,
    restoreError,
    streamError,
    setStreamError,
  } = useJobLifecycle<ActivationJobSnapshot, ActivationOutputEvent>(
    '/api/activation/audit/jobs',
    'activation audit',
    {
      onRestore: (currentJob) => {
        if (currentJob === null) setLines([])
      },
      onOutput: (output) => {
        setLines((previous) => [...previous, output.line])
        setJob((previous) =>
          previous === null ? previous : { ...previous, progress: output.progress },
        )
      },
      onComplete: (completedJob) => {
        if (completedJob.status === 'succeeded') {
          void queryClient.invalidateQueries({ queryKey: ['activation-audit-caches'] })
        }
      },
    },
  )
  const [startPending, setStartPending] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [lines, setLines] = useState<string[]>([])
  const [requestError, setRequestError] = useState<Error | null>(null)
  const defaultMethodsApplied = useRef(false)
  const outputViewportRef = useRef<HTMLDivElement>(null)
  const outputAtBottom = useRef(true)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })
  const calibrationsQuery = useQuery({
    queryKey: ['activation-calibrations'],
    queryFn: () => getJson<ArtifactsResponse>('/api/activation-calibrations'),
  })
  const foldersQuery = useQuery({
    queryKey: ['activation-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/activation-folders'),
  })
  const methodsQuery = useQuery({
    queryKey: ['activation-methods'],
    queryFn: () => getJson<MethodsResponse>('/api/activation-methods'),
  })

  const form = useForm<ActivationAuditFormValues>({
    initialValues: {
      source_path: null,
      activation_calibration: null,
      output_folder: null,
      output_name: '',
      device: 'cpu',
      methods: [],
      overwrite: false,
    },
    validate: {
      source_path: (value) => (value ? null : 'Source checkpoint is required'),
      activation_calibration: (value) => (value ? null : 'Calibration is required'),
      output_folder: (value) => (value ? null : 'Cache folder is required'),
      output_name: (value) => {
        const name = value.trim()
        if (!name) return 'Cache base name is required'
        if (name.includes('/') || name.includes('\\')) return 'Enter a filename, not a path'
        return null
      },
      methods: (value) => (value.length > 0 ? null : 'Select at least one method'),
    },
  })

  useEffect(() => {
    if (!defaultMethodsApplied.current && methodsQuery.data) {
      form.setFieldValue('methods', methodsQuery.data.default_methods)
      defaultMethodsApplied.current = true
    }
  }, [form, methodsQuery.data])

  useEffect(() => {
    const viewport = outputViewportRef.current
    if (viewport !== null && outputAtBottom.current) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [lines])

  const models = modelsQuery.data?.models ?? []
  const calibrations = calibrationsQuery.data?.calibrations ?? []
  const folders = foldersQuery.data?.folders ?? []
  const methods = methodsQuery.data?.methods ?? []
  const modelNameCounts = new Map<string, number>()
  for (const model of models) {
    const key = model.name.toLowerCase()
    modelNameCounts.set(key, (modelNameCounts.get(key) ?? 0) + 1)
  }
  const modelOptions = models.map((model) => ({
    label:
      (modelNameCounts.get(model.name.toLowerCase()) ?? 0) > 1
        ? model.name + ' — ' + parentPath(model.path)
        : model.name,
    value: model.path,
  }))
  const calibrationOptions = groupedArtifactOptions(calibrations)
  const folderOptions = folders.map((folder) => ({ label: folder, value: folder }))
  const selectedMethodCount = form.values.methods.length
  const allMethodsSelected = methods.length > 0 && selectedMethodCount === methods.length
  const isJobActive =
    job !== null &&
    activeJobId !== null &&
    (job.status === 'starting' || job.status === 'running' || job.status === 'stopping')
  const formDisabled = restoring || startPending || isJobActive
  const result = job?.status === 'succeeded' ? job.result : null
  const errorMessage =
    requestError?.message ?? restoreError?.message ??
    (job?.status === 'failed' ? job.error : null) ?? streamError

  const submit = async (values: ActivationAuditFormValues) => {
    if (values.output_folder === null) return
    setStartPending(true)
    setRequestError(null)
    setStreamError(null)
    setJob(null)
    setActiveJobId(null)
    setLines([])
    try {
      const response = await postJson<{ job: ActivationJobSnapshot }>(
        '/api/activation/audit/jobs',
        {
          source_path: values.source_path,
          activation_calibration: values.activation_calibration,
          output: joinOutputPath(values.output_folder, values.output_name.trim()),
          device: values.device,
          methods: values.methods,
          overwrite: values.overwrite,
        },
      )
      setJob(response.job)
      setActiveJobId(response.job.id)
    } catch (error: unknown) {
      setRequestError(
        error instanceof Error ? error : new Error('Could not start activation audit job'),
      )
    } finally {
      setStartPending(false)
    }
  }

  const stop = async () => {
    if (activeJobId === null) return
    setStopPending(true)
    try {
      const response = await deleteJson<{ job: ActivationJobSnapshot }>(
        '/api/activation/audit/jobs/' + encodeURIComponent(activeJobId),
        {},
      )
      setJob(response.job)
      setRequestError(null)
    } catch (error: unknown) {
      setRequestError(error instanceof Error ? error : new Error('Could not stop activation audit job'))
    } finally {
      setStopPending(false)
    }
  }

  return (
    <Card withBorder padding="lg" radius="md">
      <form onSubmit={form.onSubmit(submit)}>
        <Stack>
          <Group align="center" gap="sm">
            <Pulse size={22} />
            <Stack gap={2}>
              <Title order={3}>Create audit cache</Title>
              <Text c="dimmed" size="sm">
                Measure every calibration layer once and reuse the paired cache for review or optimization.
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
          <Select
            label="Calibration"
            placeholder="Choose a paired calibration"
            data={calibrationOptions}
            disabled={formDisabled || calibrationsQuery.isPending}
            searchable
            clearable
            nothingFoundMessage="No complete calibration pairs found"
            withAsterisk
            {...form.getInputProps('activation_calibration')}
          />
          <Select
            label="Cache destination folder"
            placeholder="Choose a configured activation folder"
            data={folderOptions}
            disabled={formDisabled || foldersQuery.isPending}
            searchable
            clearable
            nothingFoundMessage="No activation folders found"
            withAsterisk
            {...form.getInputProps('output_folder')}
          />
          <TextInput
            label="Cache base name"
            description="The server creates the paired JSON and Safetensors cache files."
            placeholder="checkpoint.activation-audit"
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
            CUDA accelerates supported ConvRot activation measurements only. INT7 artifacts require the matching INT7-capable custom-node release. Review and optimize run from the existing cache.
          </Text>

          <Stack gap={0}>
            <Text fw={500} size="sm">Methods</Text>
            <Text c="dimmed" size="xs">
              Select candidate methods; the calibration layers are always audited in full.
            </Text>
            <Group gap="md" mt="xs" wrap="wrap">
              <Checkbox
                label="Select all"
                checked={allMethodsSelected}
                indeterminate={selectedMethodCount > 0 && !allMethodsSelected}
                disabled={formDisabled || methods.length === 0}
                onChange={(event) =>
                  form.setFieldValue(
                    'methods',
                    event.currentTarget.checked ? methods.map((method) => method.value) : [],
                  )
                }
              />
              {methods.map((method) => (
                <Checkbox
                  key={method.value}
                  label={method.label}
                  value={method.value}
                  checked={form.values.methods.includes(method.value)}
                  disabled={formDisabled}
                  onChange={(event) => {
                    const next = new Set(form.values.methods)
                    if (event.currentTarget.checked) next.add(method.value)
                    else next.delete(method.value)
                    form.setFieldValue('methods', Array.from(next))
                  }}
                />
              ))}
            </Group>
            {form.errors.methods && <Text c="red" size="xs" mt="xs">{form.errors.methods}</Text>}
          </Stack>

          <Checkbox
            label="Replace existing cache pair"
            disabled={formDisabled}
            {...form.getInputProps('overwrite', { type: 'checkbox' })}
          />
          <Group justify="flex-end" mt="sm" w="100%">
            {isJobActive && (
              <Button
                color="red"
                variant="light"
                type="button"
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
              {startPending || isJobActive ? 'Auditing' : 'Create cache'}
            </Button>
          </Group>
        </Stack>
      </form>

      {(isJobActive || lines.length > 0) && (
        <Paper withBorder p="sm" mt="md">
          <Stack gap="xs">
            <Group gap="sm" aria-live="polite">
              {isJobActive && <Loader size="sm" />}
              <Text>
                {job?.status === 'starting'
                  ? 'Starting activation audit'
                  : job?.status === 'running'
                    ? 'Auditing tensors'
                    : job?.status === 'stopping'
                      ? 'Stopping activation audit'
                      : job?.status === 'succeeded'
                        ? 'Activation audit complete'
                        : job?.status === 'failed'
                          ? 'Activation audit failed'
                          : 'Activation audit output'}
              </Text>
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
                  <Progress value={Math.min(100, (job.progress.current / job.progress.total) * 100)} />
                )}
              </>
            )}
            <Text size="sm" fw={500} mt="xs">Audit output</Text>
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
              aria-label="Activation audit output"
            >
              <Stack gap={2} p="xs">
                {lines.length === 0 ? (
                  <Text size="sm" c="dimmed">
                    {isJobActive
                      ? 'Loading calibration statistics and preparing the source...'
                      : 'Waiting for audit output...'}
                  </Text>
                ) : (
                  lines.map((line, index) => (
                    <Text
                      key={String(index) + line}
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
            <Text mt="xs">That cache pair already exists. Choose another base name or enable replacement.</Text>
          )}
        </Alert>
      )}
      {job?.status === 'cancelled' && (
        <Alert color="yellow" mt="md">Activation audit stopped. No completed cache is available.</Alert>
      )}
      {result && (
        <Paper withBorder p="lg" mt="md">
          <Stack gap="md">
            <Title order={4}>Cache created</Title>
            <Text size="sm">Metadata: {result.cache_metadata_path}</Text>
            <Text size="sm">Tensors: {result.cache_tensors_path}</Text>
            <SimpleGrid cols={{ base: 1, xs: 2 }}>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">Audited layers</Text>
                <Text fw={500}>{result.layer_count}</Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">Candidate methods</Text>
                <Text fw={500}>{result.method_count}</Text>
              </Paper>
            </SimpleGrid>
            <Alert color="yellow">{result.warning}</Alert>
          </Stack>
        </Paper>
      )}
    </Card>
  )
}

function ReviewCachePanel() {
  const cachesQuery = useQuery({
    queryKey: ['activation-audit-caches'],
    queryFn: () => getJson<ArtifactsResponse>('/api/activation-audit-caches'),
  })
  const calibrationsQuery = useQuery({
    queryKey: ['activation-calibrations'],
    queryFn: () => getJson<ArtifactsResponse>('/api/activation-calibrations'),
  })
  const metricsQuery = useQuery({
    queryKey: ['activation-metrics'],
    queryFn: () => getJson<MetricsResponse>('/api/activation-metrics'),
  })
  const form = useForm({
    initialValues: {
      audit_cache: null as string | null,
      activation_calibration: null as string | null,
      metric: '',
      top_n: '5',
    },
    validate: {
      audit_cache: (value: string | null) => (value ? null : 'Cache is required'),
      activation_calibration: (value: string | null) => (value ? null : 'Calibration is required'),
      metric: (value: string) => (value ? null : 'Metric is required'),
      top_n: (value: string) =>
        Number.isInteger(Number(value)) && Number(value) >= 1 && Number(value) <= 100
          ? null
          : 'Top N must be an integer from 1 to 100',
    },
  })
  const scoreMutation = useMutation({
    mutationFn: (body: object) => postJson<{ result: ScoreReport }>('/api/activation/score', body),
  })
  const inspectMutation = useMutation({
    mutationFn: (body: object) => postJson<{ result: InspectReport }>('/api/activation/inspect', body),
  })

  useEffect(() => {
    if (!form.values.metric && metricsQuery.data) {
      form.setFieldValue('metric', metricsQuery.data.default_metric)
    }
  }, [form, metricsQuery.data])

  const caches = cachesQuery.data?.caches ?? []
  const calibrations = calibrationsQuery.data?.calibrations ?? []
  const metrics = metricsQuery.data?.metrics ?? []
  const disabled = scoreMutation.isPending || inspectMutation.isPending
  const cacheOptions = groupedArtifactOptions(caches)
  const calibrationOptions = groupedArtifactOptions(calibrations)
  const submitScore = () => {
    if (!form.validate().hasErrors) {
      scoreMutation.mutate({
        audit_cache: form.values.audit_cache,
        activation_calibration: form.values.activation_calibration,
        metric: form.values.metric,
      })
    }
  }
  const submitInspect = () => {
    if (!form.validate().hasErrors) {
      inspectMutation.mutate({
        audit_cache: form.values.audit_cache,
        activation_calibration: form.values.activation_calibration,
        top_n: Number(form.values.top_n),
      })
    }
  }
  const error = scoreMutation.error ?? inspectMutation.error
  const score = scoreMutation.data?.result
  const inspection = inspectMutation.data?.result

  return (
    <Card withBorder padding="lg" radius="md">
      <Stack>
        <Title order={3}>Review cache</Title>
        <Text c="dimmed" size="sm">
          Score and inspect a completed cache without rerunning candidate quantization.
        </Text>
        <Select
          label="Audit cache"
          placeholder="Choose a paired cache"
          data={cacheOptions}
          searchable
          clearable
          withAsterisk
          disabled={disabled}
          {...form.getInputProps('audit_cache')}
        />
        <Select
          label="Calibration"
          placeholder="Choose the matching calibration"
          data={calibrationOptions}
          searchable
          clearable
          withAsterisk
          disabled={disabled}
          {...form.getInputProps('activation_calibration')}
        />
        <Select
          label="Metric"
          data={metrics.map((metric) => ({ label: metric.label, value: metric.value }))}
          disabled={disabled || metricsQuery.isPending}
          withAsterisk
          {...form.getInputProps('metric')}
        />
        <Group>
          <Button onClick={submitScore} loading={scoreMutation.isPending} disabled={disabled}>
            Score
          </Button>
          <TextInput
            label="Worst evaluations"
            type="number"
            min={1}
            max={100}
            style={{ flex: '1 1 120px' }}
            disabled={disabled}
            {...form.getInputProps('top_n')}
          />
          <Button onClick={submitInspect} loading={inspectMutation.isPending} disabled={disabled}>
            Inspect
          </Button>
        </Group>
        {error && (
          <Alert color="red">
            {error instanceof Error ? error.message : 'Could not review activation cache'}
          </Alert>
        )}
        {score && (
          <Paper withBorder p="md">
            <Stack>
              <Title order={4}>Score</Title>
              <Text size="sm">
                {score.summary.available_candidate_count} of {score.summary.candidate_count} candidates available across {score.summary.layer_count} layers.
              </Text>
              <ScrollArea>
                <Table>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Tensor</Table.Th>
                      <Table.Th>Method</Table.Th>
                      <Table.Th>Storage</Table.Th>
                      <Table.Th>Cost</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {score.results.slice(0, 20).map((item) => (
                      <Table.Tr key={item.tensor_name + item.method}>
                        <Table.Td>{item.tensor_name}</Table.Td>
                        <Table.Td>{item.method}</Table.Td>
                        <Table.Td>{item.storage_bytes === null ? '—' : formatBytes(item.storage_bytes)}</Table.Td>
                        <Table.Td>{formatMetric(item.objective_cost)}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Stack>
          </Paper>
        )}
        {inspection && (
          <Paper withBorder p="md">
            <Stack>
              <Title order={4}>Inspection</Title>
              <Text size="sm">
                {inspection.summary.layer_count} layers; {inspection.summary.available_candidate_count} available candidates.
              </Text>
              {Object.entries(inspection.rankings).map(([name, entries]) => (
                <Stack gap={2} key={name}>
                  <Text fw={500} size="sm">{name.replaceAll('_', ' ')}</Text>
                  {entries.map((entry) => (
                    <Text size="sm" key={entry.tensor_name + entry.metric}>
                      {entry.tensor_name}{entry.method ? ' — ' + entry.method : ''}: {formatMetric(entry.value)}
                    </Text>
                  ))}
                </Stack>
              ))}
            </Stack>
          </Paper>
        )}
      </Stack>
    </Card>
  )
}

function OptimizeActivationPanel() {
  const [allowedMethods, setAllowedMethods] = useState<string[]>([])
  const [excludePrefixes, setExcludePrefixes] = useState('')
  const [excludeSuffixes, setExcludeSuffixes] = useState('')
  const [profileIdEdited, setProfileIdEdited] = useState(false)
  const lastDetailsCache = useRef<string | null>(null)
  const cachesQuery = useQuery({
    queryKey: ['activation-audit-caches'],
    queryFn: () => getJson<ArtifactsResponse>('/api/activation-audit-caches'),
  })
  const calibrationsQuery = useQuery({
    queryKey: ['activation-calibrations'],
    queryFn: () => getJson<ArtifactsResponse>('/api/activation-calibrations'),
  })
  const metricsQuery = useQuery({
    queryKey: ['activation-metrics'],
    queryFn: () => getJson<MetricsResponse>('/api/activation-metrics'),
  })
  const foldersQuery = useQuery({
    queryKey: ['profile-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/profile-folders'),
  })
  const form = useForm<OptimizeFormValues>({
    initialValues: {
      audit_cache: null,
      activation_calibration: null,
      metric: '',
      budget_mode: 'target',
      budget_value: '',
      output_folder: null,
      output_name: '',
      profile_id: '',
      summary_name: '',
      baseline_method: 'bf16',
      overwrite: false,
      dry_run: false,
    },
    validate: {
      audit_cache: (value) => (value ? null : 'Cache is required'),
      activation_calibration: (value) => (value ? null : 'Calibration is required'),
      metric: (value) => (value ? null : 'Metric is required'),
      budget_value: (value, values) => {
        const number = Number(value)
        return Number.isFinite(number) &&
          (values.budget_mode === 'promotion' ? number >= 0 : number > 0)
          ? null
          : 'Enter a valid budget'
      },
      output_folder: (value) => (value ? null : 'Profile folder is required'),
      output_name: (value) => {
        if (!value.trim()) return 'Profile filename is required'
        return value.includes('/') || value.includes('\\') ? 'Enter a filename, not a path' : null
      },
      profile_id: (value) => (value.trim() ? null : 'Profile ID is required'),
      summary_name: (value) =>
        value.includes('/') || value.includes('\\') ? 'Enter a filename, not a path' : null,
    },
  })
  const detailsQuery = useQuery({
    queryKey: ['activation-cache-details', form.values.audit_cache],
    queryFn: () =>
      getJson<{ methods: MethodOption[]; default_methods: string[]; baseline_methods: string[] }>(
        '/api/activation-cache-details?path=' +
          encodeURIComponent(form.values.audit_cache ?? ''),
      ),
    enabled: Boolean(form.values.audit_cache),
  })
  const optimizeMutation = useMutation({
    mutationFn: (body: object) => postJson<{ result: OptimizeResult }>('/api/activation/optimize', body),
  })

  useEffect(() => {
    if (form.values.metric === '' && metricsQuery.data) {
      form.setFieldValue('metric', metricsQuery.data.default_metric)
    }
  }, [form, metricsQuery.data])
  useEffect(() => {
    if (
      form.values.audit_cache &&
      detailsQuery.data &&
      lastDetailsCache.current !== form.values.audit_cache
    ) {
      setAllowedMethods(detailsQuery.data.default_methods)
      form.setFieldValue('baseline_method', 'bf16')
      lastDetailsCache.current = form.values.audit_cache
    }
  }, [detailsQuery.data, form, form.values.audit_cache])

  const caches = cachesQuery.data?.caches ?? []
  const calibrations = calibrationsQuery.data?.calibrations ?? []
  const metrics = metricsQuery.data?.metrics ?? []
  const folders = foldersQuery.data?.folders ?? []
  const methodOptions = detailsQuery.data?.methods ?? []
  const formDisabled = optimizeMutation.isPending
  const submit = (values: OptimizeFormValues) => {
    const parsedBudget = Number(values.budget_value)
    optimizeMutation.reset()
    optimizeMutation.mutate({
      audit_cache: values.audit_cache,
      activation_calibration: values.activation_calibration,
      output_path: joinOutputPath(values.output_folder ?? '', values.output_name.trim()),
      summary_output: values.summary_name.trim()
        ? joinOutputPath(values.output_folder ?? '', values.summary_name.trim())
        : null,
      target_size_gib: values.budget_mode === 'target' ? parsedBudget : null,
      promotion_budget_mib: values.budget_mode === 'promotion' ? parsedBudget : null,
      metric: values.metric,
      methods: allowedMethods,
      baseline_method: values.baseline_method,
      exclude_prefixes: excludePrefixes.split(',').map((value) => value.trim()).filter(Boolean),
      exclude_suffixes: excludeSuffixes.split(',').map((value) => value.trim()).filter(Boolean),
      profile_id: values.profile_id.trim(),
      overwrite: values.overwrite,
      dry_run: values.dry_run,
    })
  }
  const result = optimizeMutation.data?.result
  const error = optimizeMutation.error
  const updateFilename = (outputName: string) => {
    const previousDerived = form.values.output_name.trim().replace(/\.json$/i, '')
    const nextProfileId = outputName.trim().replace(/\.json$/i, '')
    form.setFieldValue('output_name', outputName)
    if (!profileIdEdited && (!form.values.profile_id || form.values.profile_id === previousDerived)) {
      form.setFieldValue('profile_id', nextProfileId)
    }
  }

  return (
    <Card withBorder padding="lg" radius="md">
      <form onSubmit={form.onSubmit(submit)}>
        <Stack>
          <Title order={3}>Optimize profile</Title>
          <Text c="dimmed" size="sm">
            Generate or estimate a profile from cached activation measurements.
          </Text>
          <Select
            label="Audit cache"
            placeholder="Choose a paired cache"
            data={groupedArtifactOptions(caches)}
            searchable
            clearable
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('audit_cache')}
          />
          <Select
            label="Calibration"
            placeholder="Choose the matching calibration"
            data={groupedArtifactOptions(calibrations)}
            searchable
            clearable
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('activation_calibration')}
          />
          <Select
            label="Metric"
            data={metrics.map((metric) => ({ label: metric.label, value: metric.value }))}
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('metric')}
          />
          <Select
            label="Budget mode"
            data={[
              { label: 'Target size (GiB)', value: 'target' },
              { label: 'Promotion budget (MiB)', value: 'promotion' },
            ]}
            allowDeselect={false}
            disabled={formDisabled}
            {...form.getInputProps('budget_mode')}
          />
          <TextInput
            label={form.values.budget_mode === 'target' ? 'Target size (GiB)' : 'Promotion budget (MiB)'}
            type="number"
            min={form.values.budget_mode === 'target' ? 0 : 0}
            step="any"
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('budget_value')}
          />
          <Select
            label="Profile folder"
            placeholder="Choose a configured profile folder"
            data={folders.map((folder) => ({ label: folder, value: folder }))}
            searchable
            clearable
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('output_folder')}
          />
          <TextInput
            label="Profile filename"
            description=".json is added automatically by the server."
            placeholder="activation-profile.json"
            disabled={formDisabled}
            withAsterisk
            value={form.values.output_name}
            onChange={(event) => updateFilename(event.currentTarget.value)}
            error={form.errors.output_name}
          />
          <TextInput
            label="Profile ID"
            placeholder="activation-profile"
            disabled={formDisabled}
            withAsterisk
            {...form.getInputProps('profile_id')}
            onChange={(event) => {
              setProfileIdEdited(true)
              form.setFieldValue('profile_id', event.currentTarget.value)
            }}
          />
          <TextInput
            label="Summary filename"
            description="Leave blank to derive a -summary.json file beside the profile."
            placeholder="activation-profile-summary.json"
            disabled={formDisabled}
            {...form.getInputProps('summary_name')}
          />

          <Stack gap="xs">
            <Text fw={500} size="sm">Allowed methods</Text>
            <Text c="dimmed" size="xs">Only methods present in this cache are available.</Text>
            <Group gap="md" wrap="wrap">
              {methodOptions.map((method) => (
                <Checkbox
                  key={method.value}
                  label={method.label}
                  checked={allowedMethods.includes(method.value)}
                  disabled={formDisabled}
                  onChange={(event) => {
                    const checked = event.currentTarget.checked
                    setAllowedMethods((previous) =>
                      checked
                        ? Array.from(new Set([...previous, method.value]))
                        : previous.filter((value) => value !== method.value),
                    )
                  }}
                />
              ))}
            </Group>
          </Stack>
          <Select
            label="Baseline method"
            data={(detailsQuery.data?.baseline_methods ?? ['bf16']).map((method) => ({
              label: method === 'bf16' ? 'BF16' : method,
              value: method,
            }))}
            disabled={formDisabled || detailsQuery.isPending}
            allowDeselect={false}
            {...form.getInputProps('baseline_method')}
          />

          <SimpleGrid cols={{ base: 1, sm: 2 }}>
            <TextInput
              label="Exclude prefixes"
              description="Comma-separated prefixes to keep in BF16."
              placeholder="blocks.0., blocks.1."
              value={excludePrefixes}
              disabled={formDisabled}
              onChange={(event) => setExcludePrefixes(event.currentTarget.value)}
            />
            <TextInput
              label="Exclude suffixes"
              description="Comma-separated suffixes to keep in BF16."
              placeholder=".attn.wq.weight, .attn.wk.weight"
              value={excludeSuffixes}
              disabled={formDisabled}
              onChange={(event) => setExcludeSuffixes(event.currentTarget.value)}
            />
          </SimpleGrid>

          <Checkbox
            label="Replace existing outputs"
            disabled={formDisabled}
            {...form.getInputProps('overwrite', { type: 'checkbox' })}
          />
          <Checkbox
            label="Estimate only"
            description="Validate and calculate the profile without writing either output."
            disabled={formDisabled}
            {...form.getInputProps('dry_run', { type: 'checkbox' })}
          />
          <Button type="submit" loading={formDisabled} disabled={formDisabled}>
            {form.values.dry_run ? 'Estimate profile' : 'Optimize profile'}
          </Button>
        </Stack>
      </form>

      {optimizeMutation.isPending && (
        <Group gap="sm" mt="md" aria-live="polite">
          <Loader size="sm" />
          <Text>
            {form.values.dry_run ? 'Estimating activation profile' : 'Optimizing activation profile'}
          </Text>
        </Group>
      )}

      {error && (
        <Alert color="red" mt="md">
          {error instanceof Error ? error.message : 'Activation optimization failed'}
          {error instanceof ApiError && error.status === 409 && (
            <Text mt="xs">An output already exists. Choose another name or enable replacement.</Text>
          )}
        </Alert>
      )}
      {result && (
        <Paper withBorder p="lg" mt="md">
          <Stack gap="md">
            <Title order={4}>{result.dry_run ? 'Profile estimate' : 'Profile created'}</Title>
            <SimpleGrid cols={{ base: 1, xs: 3 }}>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">Estimated size</Text>
                <Text fw={500}>{formatBytes(result.estimated_output_bytes)}</Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">Profile ID</Text>
                <Text fw={500} style={{ overflowWrap: 'anywhere' }}>{result.profile_id}</Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">Metric</Text>
                <Text fw={500} style={{ overflowWrap: 'anywhere' }}>{result.metric}</Text>
              </Paper>
            </SimpleGrid>
            <Text size="sm">
              Constraint:{' '}
              {result.target_bytes !== null
                ? 'target ' + formatBytes(result.target_bytes)
                : 'promotion budget ' + formatBytes(result.promotion_budget_bytes ?? 0)}
            </Text>
            {result.dry_run ? (
              <Text c="dimmed">No profile or summary was written.</Text>
            ) : (
              <>
                <Text size="sm">Profile: {result.profile_path}</Text>
                <Text size="sm">Summary: {result.summary_path}</Text>
              </>
            )}
            <Alert color="yellow">{result.warning}</Alert>
          </Stack>
        </Paper>
      )}
    </Card>
  )
}

function ActivationPage() {
  return (
    <Stack gap="xl">
      <PageHeader
        title="Activation"
        description="Create, review, and optimize version-1 activation artifacts."
      />
      <ActivationAuditPanel />
      <Divider label="Review an existing cache" labelPosition="center" />
      <ReviewCachePanel />
      <Divider label="Optimize from an existing cache" labelPosition="center" />
      <OptimizeActivationPanel />
      <Text size="sm" c="dimmed">
        Activation collection remains a ComfyUI or CLI workflow because it requires a live model execution context. Configure paired artifacts in{' '}
        <Anchor component={Link} to="/settings">Settings</Anchor>.
      </Text>
    </Stack>
  )
}

export const Route = createFileRoute('/activation')({ component: ActivationPage })
