import { Cube } from '@phosphor-icons/react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Box,
  Group,
  Loader,
  Modal,
  Paper,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'

import { ApiError, deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { getModels } from '../models.ts'
import { formatBytes } from '../utils/format-bytes.ts'
import { joinOutputPath } from '../utils/output-path.ts'
import { parentPath } from '../utils/parent-path.ts'

type QuantizeFormValues = {
  source_path: string | null
  profile: string | null
  output_folder: string | null
  output_name: string
  loras: Array<{ path: string | null; strength: string }>
  device: 'cpu' | 'cuda'
  dry_run: boolean
}

type QuantizeResult =
  | {
      estimated_output_bytes: number
      dry_run: true
    }
  | {
      output_path: string
      output_bytes: number
      adapter_count: number
      dry_run?: false
    }

type QuantizeProgress = {
  current: number
  total: number
  tensor_name: string
}

type QuantizeJobSnapshot = {
  id: string
  status: 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
  started_at: string
  completed_at: string | null
  progress: QuantizeProgress | null
  result: QuantizeResult | null
  error: string | null
}

type QuantizeOutputEvent = {
  sequence: number
  line: string
  progress: QuantizeProgress
}

type CurrentQuantizeJobResponse = {
  job: QuantizeJobSnapshot | null
}

type StartQuantizeJobResponse = {
  job: QuantizeJobSnapshot
}

type ProfilesResponse = {
  profiles: {
    folder: string
    name: string
    path: string
  }[]
}

type LorasResponse = {
  loras: {
    name: string
    path: string
  }[]
}

type FoldersResponse = {
  folders: string[]
}

type ProfileDetails = {
  profile_id: string
  default: string
  rules: Array<{
    action: string
    fallback?: string
    prefix: string
    suffixes: string[]
  }>
}

const DEVICE_OPTIONS = [
  { label: 'CPU', value: 'cpu' },
  { label: 'CUDA', value: 'cuda' },
]

function QuantizeResultSummary({ result }: { result: QuantizeResult }) {
  if (result.dry_run === true) {
    return (
      <Stack gap="lg">
        <SimpleGrid cols={{ base: 1, xs: 3 }}>
          <Paper withBorder p="sm">
            <Text c="dimmed" size="sm">Formatted size</Text>
            <Text fw={500}>{formatBytes(result.estimated_output_bytes)}</Text>
          </Paper>
          <Paper withBorder p="sm">
            <Text c="dimmed" size="sm">MiB</Text>
            <Text fw={500}>{(result.estimated_output_bytes / 1024 ** 2).toFixed(2)} MiB</Text>
          </Paper>
          <Paper withBorder p="sm">
            <Text c="dimmed" size="sm">GiB</Text>
            <Text fw={500}>{(result.estimated_output_bytes / 1024 ** 3).toFixed(2)} GiB</Text>
          </Paper>
        </SimpleGrid>
        <Text c="dimmed">No checkpoint was written.</Text>
      </Stack>
    )
  }

  return (
    <Stack gap="lg">
      <Stack gap={4} style={{ minWidth: 0 }}>
        <Text c="dimmed" size="sm">Output checkpoint</Text>
        <Text component="div" title={result.output_path} style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', userSelect: 'text' }}>
          {result.output_path}
        </Text>
      </Stack>
      <SimpleGrid cols={{ base: 1, xs: 2 }}>
        <Paper withBorder p="sm">
          <Text c="dimmed" size="sm">Output size</Text>
          <Text fw={500}>{formatBytes(result.output_bytes)}</Text>
        </Paper>
        <Paper withBorder p="sm">
          <Text c="dimmed" size="sm">Adapters</Text>
          <Text fw={500}>{result.adapter_count}</Text>
        </Paper>
      </SimpleGrid>
    </Stack>
  )
}

function QuantizePage() {
  const [job, setJob] = useState<QuantizeJobSnapshot | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [completedJobId, setCompletedJobId] = useState<string | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const [startPending, setStartPending] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [restoring, setRestoring] = useState(true)
  const [requestError, setRequestError] = useState<Error | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const [showOutput, setShowOutput] = useState(false)
  const [profileDetailsOpened, setProfileDetailsOpened] = useState(false)
  const [resultOpened, setResultOpened] = useState(false)
  const receivedSequences = useRef(new Set<number>())
  const terminalViewportRef = useRef<HTMLDivElement>(null)
  const terminalAtBottom = useRef(true)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })
  const profilesQuery = useQuery({
    queryKey: ['profiles'],
    queryFn: () => getJson<ProfilesResponse>('/api/profiles'),
  })
  const outputFoldersQuery = useQuery({
    queryKey: ['output-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/output-folders'),
  })
  const lorasQuery = useQuery({
    queryKey: ['loras'],
    queryFn: () => getJson<LorasResponse>('/api/loras'),
  })
  const form = useForm<QuantizeFormValues>({
    initialValues: {
      source_path: null,
      profile: null,
      output_folder: null,
      output_name: '',
      loras: [],
      device: 'cuda',
      dry_run: false,
    },
    validate: {
      source_path: (value) => (value ? null : 'Source checkpoint is required'),
      profile: (value) => (value ? null : 'Quantization profile is required'),
      output_folder: (value, values) =>
        values.dry_run || value
          ? null
          : 'Output folder is required unless estimate only is enabled',
      output_name: (value, values) =>
        values.dry_run
          ? null
          : !value.trim()
            ? 'Output file name is required unless estimate only is enabled'
            : value.includes('/') || value.includes('\\')
              ? 'Enter a file name, not a path'
              : null,
    },
  })
  const profileDetailsQuery = useQuery({
    queryKey: ['profile-details', form.values.profile],
    queryFn: () =>
      getJson<ProfileDetails>(
        `/api/profile-details?path=${encodeURIComponent(form.values.profile ?? '')}`,
      ),
    enabled: profileDetailsOpened && form.values.profile !== null,
  })

  useEffect(() => {
    let cancelled = false
    getJson<CurrentQuantizeJobResponse>('/api/quantize/jobs/current')
      .then(({ job: currentJob }) => {
        if (cancelled || currentJob === null) return
        setJob(currentJob)
        if (
          currentJob.status === 'starting' ||
          currentJob.status === 'running' ||
          currentJob.status === 'stopping'
        ) {
          setActiveJobId(currentJob.id)
          setShowOutput(true)
        } else {
          setCompletedJobId(currentJob.id)
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRequestError(
            error instanceof Error ? error : new Error('Could not restore Quantize job'),
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
      `/api/quantize/jobs/${encodeURIComponent(activeJobId)}/events`,
    )
    let completeReceived = false
    let terminalStateReceived = false
    receivedSequences.current.clear()

    const handleState = (event: MessageEvent<string>) => {
      const nextJob = JSON.parse(event.data) as QuantizeJobSnapshot
      terminalStateReceived =
        nextJob.status === 'succeeded' ||
        nextJob.status === 'failed' ||
        nextJob.status === 'cancelled'
      setStreamError(null)
      setJob(nextJob)
    }
    const handleOutput = (event: MessageEvent<string>) => {
      const output = JSON.parse(event.data) as QuantizeOutputEvent
      if (receivedSequences.current.has(output.sequence)) return
      receivedSequences.current.add(output.sequence)
      setStreamError(null)
      setLines((previous) => [...previous, output.line])
      setJob((previous) =>
        previous === null ? previous : { ...previous, progress: output.progress },
      )
    }
    const handleComplete = (event: MessageEvent<string>) => {
      const completedJob = JSON.parse(event.data) as QuantizeJobSnapshot
      completeReceived = true
      setStreamError(null)
      setJob(completedJob)
      setCompletedJobId(completedJob.id)
      setActiveJobId(null)
      setResultOpened(completedJob.status === 'succeeded')
      source.close()
    }

    source.addEventListener('state', handleState)
    source.addEventListener('output', handleOutput)
    source.addEventListener('complete', handleComplete)
    source.onerror = () => {
      if (!completeReceived && !terminalStateReceived) {
        setStreamError('The Quantize progress connection was interrupted.')
      }
    }

    return () => {
      completeReceived = true
      source.close()
    }
  }, [activeJobId])

  useEffect(() => {
    const viewport = terminalViewportRef.current
    if (viewport !== null && terminalAtBottom.current) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [lines])

  const models = modelsQuery.data?.models ?? []
  const profiles = profilesQuery.data?.profiles ?? []
  const loras = lorasQuery.data?.loras ?? []
  const outputFolders = outputFoldersQuery.data?.folders ?? []
  const selectedModel = models.find((model) => model.path === form.values.source_path)
  const nameCounts = new Map<string, number>()
  for (const model of models) {
    const key = model.name.toLowerCase()
    nameCounts.set(key, (nameCounts.get(key) ?? 0) + 1)
  }
  const modelOptions = [
    { group: 'Quantizable', items: models.filter((model) => model.quantizable) },
    {
      group: 'Not quantizable — every tensor must be BF16, F16, or F32',
      items: models.filter((model) => !model.quantizable),
    },
  ].flatMap(({ group, items }) =>
    items.length === 0
      ? []
      : [{
          group,
          items: items.map((model) => ({
            label:
              (nameCounts.get(model.name.toLowerCase()) ?? 0) > 1
                ? `${model.name} — ${parentPath(model.path)}`
                : model.name,
            value: model.path,
          })),
        }],
  )
  const loraNameCounts = new Map<string, number>()
  for (const lora of loras) {
    const key = lora.name.toLowerCase()
    loraNameCounts.set(key, (loraNameCounts.get(key) ?? 0) + 1)
  }
  const loraOptions = loras.map((lora) => ({
    label:
      (loraNameCounts.get(lora.name.toLowerCase()) ?? 0) > 1
        ? `${lora.name} — ${parentPath(lora.path)}`
        : lora.name,
    value: lora.path,
  }))
  const profileGroups = new Map<string, { label: string; value: string }[]>()
  for (const profile of profiles) {
    const options = profileGroups.get(profile.folder) ?? []
    options.push({ label: profile.name, value: profile.path })
    profileGroups.set(profile.folder, options)
  }
  const profileOptions = Array.from(profileGroups, ([group, items]) => ({
    group,
    items,
  }))
  const rulesByAction = new Map<
    string,
    Array<ProfileDetails['rules'][number] & { index: number }>
  >()
  for (const [index, rule] of (profileDetailsQuery.data?.rules ?? []).entries()) {
    const rules = rulesByAction.get(rule.action) ?? []
    rules.push({ ...rule, index: index + 1 })
    rulesByAction.set(rule.action, rules)
  }
  const outputFolderOptions = outputFolders.map((folder) => ({
    label: folder,
    value: folder,
  }))
  const isJobActive =
    job !== null &&
    activeJobId !== null &&
    (job.status === 'starting' || job.status === 'running' || job.status === 'stopping')
  const formDisabled = restoring || startPending || isJobActive
  const result =
    completedJobId === job?.id && job.status === 'succeeded' ? job.result : null
  const jobError =
    completedJobId === job?.id && job.status === 'failed' ? job.error : null
  const jobCancelled = completedJobId === job?.id && job.status === 'cancelled'
  const error = jobError ?? requestError?.message ?? streamError
  const dryRunJob = form.values.dry_run || result?.dry_run === true
  const showStatus =
    showOutput &&
    (isJobActive ||
      (!dryRunJob &&
        (job?.status === 'succeeded' ||
          job?.status === 'failed' ||
          job?.status === 'cancelled')))
  const showTerminal = showOutput && !dryRunJob

  const submit = async (values: QuantizeFormValues) => {
    if (
      values.loras.some(
        (lora) =>
          lora.path === null ||
          !lora.strength.trim() ||
          !Number.isFinite(Number(lora.strength)),
      )
    ) {
      setRequestError(new Error('Each LoRA needs a file and a finite strength.'))
      return
    }
    setStartPending(true)
    try {
      const response = await postJson<StartQuantizeJobResponse>('/api/quantize/jobs', {
        source_path: values.source_path,
        output_path:
          values.dry_run || !values.output_folder
            ? null
            : joinOutputPath(values.output_folder, values.output_name.trim()),
        profile: values.profile,
        adapter_path: values.loras.map((lora) => lora.path),
        adapter_strength: values.loras.map((lora) => Number(lora.strength)),
        device: values.device,
        dry_run: values.dry_run,
      })
      setJob(response.job)
      setActiveJobId(response.job.id)
      setCompletedJobId(null)
      setLines([])
      receivedSequences.current.clear()
      setRequestError(null)
      setStreamError(null)
      setShowOutput(true)
    } catch (error: unknown) {
      setRequestError(
        error instanceof Error ? error : new Error('Could not start Quantize job'),
      )
    } finally {
      setStartPending(false)
    }
  }

  const stop = async () => {
    if (activeJobId === null) return
    setStopPending(true)
    try {
      const response = await deleteJson<CurrentQuantizeJobResponse>(
        `/api/quantize/jobs/${encodeURIComponent(activeJobId)}`,
        {},
      )
      setJob(response.job)
      setRequestError(null)
    } catch (error: unknown) {
      setRequestError(error instanceof Error ? error : new Error('Could not stop Quantize job'))
    } finally {
      setStopPending(false)
    }
  }

  return (
    <Stack gap="xl">
      <PageHeader
        title="Quantize checkpoint"
        description="Convert a configured checkpoint with a saved JSON profile."
      />

      <Card withBorder padding="lg" radius="md">
        <form onSubmit={form.onSubmit(submit)}>
          <Stack>
            <Group align="center" gap="sm">
              <Cube size={22} />
              <Stack gap={2}>
                <Title order={3}>Quantize checkpoint</Title>
                <Text c="dimmed" size="sm">
                  Estimate the result or write a new checkpoint.
                </Text>
              </Stack>
            </Group>

            <Select
              label="Source checkpoint"
              description="Models with other tensor types are shown with a warning."
              placeholder="Choose a configured checkpoint"
              data={modelOptions}
              disabled={formDisabled || modelsQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No safetensors checkpoints found"
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
            {selectedModel && !selectedModel.quantizable && (
              <Alert color="yellow">
                This checkpoint contains tensors outside BF16, F16, or F32. Quantization may fail.
              </Alert>
            )}

            <Stack gap="xs">
              <Group justify="space-between" align="center">
                <Stack gap={0}>
                  <Text fw={500}>LoRA files</Text>
                  <Text size="sm" c="dimmed">
                    Optional; selected files are merged before quantization.
                  </Text>
                </Stack>
                <Button
                  type="button"
                  variant="default"
                  disabled={formDisabled || lorasQuery.isPending}
                  onClick={() => form.insertListItem('loras', { path: null, strength: '1' })}
                >
                  Add LoRA
                </Button>
              </Group>
              {form.values.loras.map((_, index) => (
                <Group key={index} align="flex-end" gap="sm" wrap="wrap">
                  <Select
                    label={`LoRA file ${index + 1}`}
                    placeholder="Choose a configured LoRA file"
                    data={loraOptions}
                    disabled={formDisabled || lorasQuery.isPending}
                    searchable
                    clearable
                    nothingFoundMessage="No LoRA files found"
                    style={{ flex: '1 1 280px' }}
                    {...form.getInputProps(`loras.${index}.path`)}
                  />
                  <TextInput
                    label="Strength"
                    type="number"
                    step="any"
                    disabled={formDisabled}
                    style={{ flex: '0 1 120px' }}
                    {...form.getInputProps(`loras.${index}.strength`)}
                  />
                  <Button
                    type="button"
                    color="red"
                    variant="subtle"
                    disabled={formDisabled}
                    onClick={() => form.removeListItem('loras', index)}
                  >
                    Remove
                  </Button>
                </Group>
              ))}
              {lorasQuery.error ? (
                <Alert color="red">
                  {lorasQuery.error instanceof Error
                    ? lorasQuery.error.message
                    : 'Could not load LoRA files'}
                </Alert>
              ) : !lorasQuery.isPending && loras.length === 0 ? (
                <Text size="sm" c="dimmed">
                  No LoRA files found. Configure a LoRA folder in{' '}
                  <Anchor component={Link} to="/settings">
                    Settings
                  </Anchor>
                  .
                </Text>
              ) : null}
            </Stack>

            <Group align="flex-end" gap="sm" wrap="nowrap">
              <Select
                label="Quantization profile"
                description="Profiles are grouped by their folder."
                placeholder="Choose a configured profile"
                data={profileOptions}
                disabled={formDisabled || profilesQuery.isPending}
                searchable
                clearable
                nothingFoundMessage="No JSON profiles found"
                withAsterisk
                style={{ flex: 1 }}
                {...form.getInputProps('profile')}
              />
              <Button
                variant="default"
                disabled={!form.values.profile || formDisabled}
                onClick={() => setProfileDetailsOpened(true)}
              >
                View layers
              </Button>
            </Group>
            {profilesQuery.error ? (
              <Alert color="red">
                {profilesQuery.error instanceof Error
                  ? profilesQuery.error.message
                  : 'Could not load profiles'}
              </Alert>
            ) : !profilesQuery.isPending && profiles.length === 0 ? (
              <Text size="sm" c="dimmed">
                No JSON profiles found. Configure a profile folder in{' '}
                <Anchor component={Link} to="/settings">
                  Settings
                </Anchor>
                .
              </Text>
            ) : null}
            <Select
              label="Output folder"
              description={
                form.values.dry_run
                  ? 'Destination folder (optional for estimate only)'
                  : 'Choose where the checkpoint will be written'
              }
              placeholder="Choose a configured output folder"
              data={outputFolderOptions}
              disabled={formDisabled || outputFoldersQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No output folders found"
              withAsterisk={!form.values.dry_run}
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
              label="Output file name"
              description=".safetensors is added automatically when omitted"
              placeholder="quantized.safetensors"
              disabled={formDisabled}
              withAsterisk={!form.values.dry_run}
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
              CUDA accelerates supported ConvRot INT5/INT6 and W4A4 work but must be available locally.
            </Text>
            <Switch
              label="Estimate output size only"
              description="Do not write a checkpoint"
              disabled={formDisabled}
              {...form.getInputProps('dry_run', { type: 'checkbox' })}
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
                loading={startPending}
                disabled={formDisabled}
                w={{ base: '100%', md: 'auto' }}
              >
                {formDisabled && !restoring ? 'Quantizing' : form.values.dry_run ? 'Estimate size' : 'Quantize'}
              </Button>
            </Group>
          </Stack>
        </form>

        {showStatus && (
          <Paper withBorder p="md" mt="md" aria-live="polite">
            {dryRunJob ? (
              <Group gap="sm">
                <Loader size="sm" />
                <Text>Estimating output size</Text>
              </Group>
            ) : job?.status === 'stopping' ? (
              <Text>Stopping conversion</Text>
            ) : job?.progress === null ? (
              <Text>Preparing conversion</Text>
            ) : job?.progress ? (
              <Stack gap="xs">
                <Progress
                  value={
                    job.progress.total > 0
                      ? (job.progress.current / job.progress.total) * 100
                      : 0
                  }
                />
                <Text size="sm">
                  {job.progress.current} / {job.progress.total} tensors
                </Text>
                <Text
                  size="sm"
                  title={job.progress.tensor_name}
                  style={{
                    minWidth: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {job.progress.tensor_name}
                </Text>
              </Stack>
            ) : null}
          </Paper>
        )}

        {showTerminal && (
          <Paper withBorder p="sm" mt="md" bg="dark.9">
            <Text size="sm" fw={500} c="white" mb="xs">
              Quantize output
            </Text>
            <ScrollArea
              h={260}
              viewportRef={terminalViewportRef}
              onScrollPositionChange={({ y }) => {
                const viewport = terminalViewportRef.current
                terminalAtBottom.current =
                  viewport !== null && y + viewport.clientHeight >= viewport.scrollHeight - 8
              }}
              aria-label="Quantize output"
            >
              <Box
                component="pre"
                m={0}
                c="white"
                style={{
                  whiteSpace: 'pre-wrap',
                  overflowWrap: 'anywhere',
                }}
              >
                {lines.length > 0 ? lines.join('\n') : 'Waiting for conversion output'}
              </Box>
            </ScrollArea>
          </Paper>
        )}

        {error && (
          <Alert color="red" mt="md">
            {error}
            {requestError instanceof ApiError && requestError.status === 409 && (
              <Text mt="xs">
                That destination already exists. Choose a different output path.
              </Text>
            )}
          </Alert>
        )}
        {jobCancelled && (
          <Alert color="yellow" mt="md">
            Quantization stopped. No checkpoint was written.
          </Alert>
        )}
      </Card>

      <Modal
        opened={profileDetailsOpened}
        onClose={() => setProfileDetailsOpened(false)}
        title="Profile layers"
        size="xl"
      >
        {profileDetailsQuery.isPending ? (
          <Group gap="sm">
            <Loader size="sm" />
            <Text>Loading profile rules</Text>
          </Group>
        ) : profileDetailsQuery.error ? (
          <Alert color="red">
            {profileDetailsQuery.error instanceof Error
              ? profileDetailsQuery.error.message
              : 'Could not load profile rules'}
          </Alert>
        ) : profileDetailsQuery.data ? (
          <Stack gap="md">
            <Text size="sm" c="dimmed">
              Default: {profileDetailsQuery.data.default}. Rules are applied in order; the
              first matching rule wins.
            </Text>
            {Array.from(rulesByAction, ([action, rules]) => (
              <Stack key={action} gap="xs">
                <Text fw={500}>{action}</Text>
                {rules.map((rule) => (
                  <Paper key={rule.index} withBorder p="sm">
                    <Text size="sm" fw={500}>Rule {rule.index}</Text>
                    <Stack gap={2} mt="xs">
                      <Text size="xs" c="dimmed">Starts with</Text>
                      <Text
                        component="code"
                        size="sm"
                        style={{ display: 'block', overflowWrap: 'anywhere' }}
                      >
                        {rule.prefix || '(any layer)'}
                      </Text>
                      <Text size="xs" c="dimmed" mt="xs">Ends with</Text>
                      {rule.suffixes.map((suffix) => (
                        <Text
                          key={suffix}
                          component="code"
                          size="sm"
                          style={{ display: 'block', overflowWrap: 'anywhere' }}
                        >
                          {suffix || '(any ending)'}
                        </Text>
                      ))}
                    </Stack>
                    {rule.fallback && (
                      <Text size="sm" c="dimmed">Fallback: {rule.fallback}</Text>
                    )}
                  </Paper>
                ))}
              </Stack>
            ))}
            {rulesByAction.size === 0 && (
              <Text c="dimmed">This profile has no explicit layer rules.</Text>
            )}
          </Stack>
        ) : null}
      </Modal>

      <Modal
        opened={resultOpened && result !== null}
        onClose={() => setResultOpened(false)}
        title={result?.dry_run ? 'Estimated output' : 'Quantization complete'}
        centered
      >
        {result && <QuantizeResultSummary result={result} />}
      </Modal>

      {result && (
        <Paper withBorder p="lg" radius="md">
          <Stack gap="lg">
            <Title order={3}>{result.dry_run ? 'Estimated output' : 'Quantization complete'}</Title>
            <QuantizeResultSummary result={result} />
          </Stack>
        </Paper>
      )}
    </Stack>
  )
}

export const Route = createFileRoute('/quantize')({ component: QuantizePage })
