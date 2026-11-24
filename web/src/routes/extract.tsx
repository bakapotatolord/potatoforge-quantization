import {
  Alert,
  Button,
  Card,
  Chip,
  Collapse,
  Divider,
  Group,
  NumberInput,
  Paper,
  Progress,
  Select,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'

import { ApiError, deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { getModels } from '../models.ts'
import { formatBytes } from '../utils/format-bytes.ts'
import { joinOutputPath } from '../utils/output-path.ts'
import { parentPath } from '../utils/parent-path.ts'
import { useJobLifecycle } from '../use-job-lifecycle.ts'

const cpuOnly = import.meta.env.VITE_CPU_ONLY === 'true'
const DEFAULT_DEVICE = cpuOnly ? 'cpu' : 'cuda'
const DEVICE_OPTIONS = cpuOnly
  ? [{ label: 'CPU', value: 'cpu' }]
  : [{ label: 'CPU', value: 'cpu' }, { label: 'CUDA', value: 'cuda' }]
const RANK_CHOICES = ['8', '16', '32', '64', '128', '256']
const SPLIT_QKV_RULE = 'qkv:q,k,v'

type ExtractFormValues = {
  base_path: string | null
  tuned_path: string | null
  ranks: string[]
  adaptive: boolean
  energy: number
  max_rank: string
  split_qkv: boolean
  output_folder: string | null
  output_name: string
  include: string
  exclude: string
  include_diffs: boolean
  dtype: 'fp16' | 'bf16' | 'fp32'
  device: 'cpu' | 'cuda'
}

type ExtractOutput = {
  label: string
  path: string
  bytes: number
  tensors: number
  energy_captured: number
}

type ExtractResult = {
  outputs: ExtractOutput[]
  tensor_count: number
  skipped_tensors: number
  skipped_reasons: Record<string, number>
  unmatched_base: number
  unmatched_tuned: number
}

type ExtractProgress = { current: number; total: number; tensor_name: string }

type ExtractJobSnapshot = {
  id: string
  status: 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
  started_at: string
  completed_at: string | null
  progress: ExtractProgress | null
  result: ExtractResult | null
  error: string | null
}

type ExtractOutputEvent = {
  sequence: number
  line: string
  progress: ExtractProgress
}

type FoldersResponse = { folders: string[] }

const stripSuffix = (name: string) => name.replace(/\.safetensors$/i, '')

function ExtractResultSummary({ result }: { result: ExtractResult }) {
  return (
    <Stack gap="lg">
      <Table withTableBorder>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{result.outputs.some((o) => o.label.startsWith('e')) ? 'Energy' : 'Rank'}</Table.Th>
            <Table.Th>Size</Table.Th>
            <Table.Th>Energy captured</Table.Th>
            <Table.Th>Tensors</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {result.outputs.map((output) => (
            <Table.Tr key={output.path}>
              <Table.Td>{output.label.startsWith('e') ? `${output.label.slice(1)}%` : output.label}</Table.Td>
              <Table.Td>{formatBytes(output.bytes)}</Table.Td>
              <Table.Td>{(output.energy_captured * 100).toFixed(1)}%</Table.Td>
              <Table.Td>{output.tensors}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Stack gap={4}>
        {result.outputs.map((output) => (
          <Text
            key={output.path}
            size="sm"
            title={output.path}
            style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', userSelect: 'text' }}
          >
            {output.path}
          </Text>
        ))}
      </Stack>
      <Text size="sm" c="dimmed">
        Energy captured is weight-space error against the fine-tune, not image quality. Check generated
        images before relying on a low rank.
      </Text>
      <Text size="sm" c="dimmed">
        {result.tensor_count} tensors extracted
        {result.skipped_tensors > 0 &&
          `; ${result.skipped_tensors} skipped (${Object.entries(result.skipped_reasons)
            .map(([reason, count]) => `${count} ${reason}`)
            .join(', ')})`}
        {result.unmatched_base + result.unmatched_tuned > 0 &&
          `; ${result.unmatched_base} only in base, ${result.unmatched_tuned} only in tuned`}
        .
      </Text>
      <Button
        variant="default"
        style={{ alignSelf: 'flex-start' }}
        renderRoot={(props) => <Link {...props} to="/quantize" />}
      >
        Merge into a checkpoint with Quantize
      </Button>
    </Stack>
  )
}

function ExtractPage() {
  const [completedJobId, setCompletedJobId] = useState<string | null>(null)
  const [startPending, setStartPending] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [requestError, setRequestError] = useState<Error | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const {
    job,
    setJob,
    activeJobId,
    setActiveJobId,
    restoring,
    restoreError,
    streamError,
    setStreamError,
  } = useJobLifecycle<ExtractJobSnapshot, ExtractOutputEvent>('/api/lora/extract/jobs', 'LoRA extraction', {
    onRestore: (currentJob) => {
      if (
        currentJob !== null &&
        currentJob.status !== 'starting' &&
        currentJob.status !== 'running' &&
        currentJob.status !== 'stopping'
      ) {
        setCompletedJobId(currentJob.id)
      }
    },
    onOutput: (output) => {
      setJob((previous) =>
        previous === null ? previous : { ...previous, progress: output.progress },
      )
    },
    onComplete: (completedJob) => setCompletedJobId(completedJob.id),
  })

  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: getModels })
  const outputFoldersQuery = useQuery({
    queryKey: ['output-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/output-folders'),
  })
  const loraFoldersQuery = useQuery({
    queryKey: ['lora-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/lora-folders'),
  })

  const form = useForm<ExtractFormValues>({
    initialValues: {
      base_path: null,
      tuned_path: null,
      ranks: ['64'],
      adaptive: false,
      energy: 90,
      max_rank: '128',
      split_qkv: false,
      output_folder: null,
      output_name: '',
      include: '',
      exclude: '',
      include_diffs: false,
      dtype: 'fp16',
      device: DEFAULT_DEVICE,
    },
    validate: {
      base_path: (value) => (value ? null : 'Base checkpoint is required'),
      tuned_path: (value, values) =>
        !value
          ? 'Fine-tuned checkpoint is required'
          : value === values.base_path
            ? 'Choose a different checkpoint from the base'
            : null,
      ranks: (value, values) =>
        values.adaptive || value.length > 0 ? null : 'Choose at least one rank',
      energy: (value, values) =>
        !values.adaptive || (value > 0 && value <= 100) ? null : 'Energy must be between 0 and 100',
      output_folder: (value) => (value ? null : 'Output folder is required'),
      output_name: (value) =>
        !value.trim()
          ? 'Output file name is required'
          : value.includes('/') || value.includes('\\')
            ? 'Enter a file name, not a path'
            : null,
    },
  })

  const models = modelsQuery.data?.models ?? []
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
  const outputFolders = outputFoldersQuery.data?.folders ?? []
  const loraFolders = loraFoldersQuery.data?.folders ?? []
  const folderOptions = [
    { group: 'LoRA folders', items: loraFolders },
    { group: 'Output folders', items: outputFolders.filter((f) => !loraFolders.includes(f)) },
  ]
    .filter(({ items }) => items.length > 0)
    .map(({ group, items }) => ({ group, items: items.map((f) => ({ label: f, value: f })) }))

  const suggestedName = (tunedPath: string | null) => {
    const name = models.find((model) => model.path === tunedPath)?.name
    return name ? `${stripSuffix(name)}-delta.safetensors` : ''
  }

  const isJobActive =
    job !== null &&
    activeJobId !== null &&
    (job.status === 'starting' || job.status === 'running' || job.status === 'stopping')
  const formDisabled = restoring || startPending || isJobActive
  const result = completedJobId === job?.id && job.status === 'succeeded' ? job.result : null
  const jobError = completedJobId === job?.id && job.status === 'failed' ? job.error : null
  const jobCancelled = completedJobId === job?.id && job.status === 'cancelled'
  const error = jobError ?? requestError?.message ?? restoreError?.message ?? streamError
  const multipleRanks = !form.values.adaptive && form.values.ranks.length > 1

  const submit = async (values: ExtractFormValues) => {
    if (values.output_folder === null) return
    setStartPending(true)
    try {
      const ranks = values.adaptive
        ? [Number(values.max_rank)]
        : [...values.ranks].map(Number).sort((a, b) => a - b)
      const response = await postJson<{ job: ExtractJobSnapshot }>('/api/lora/extract/jobs', {
        base_path: values.base_path,
        tuned_path: values.tuned_path,
        output_path: joinOutputPath(values.output_folder, values.output_name.trim()),
        ranks,
        energy: values.adaptive ? values.energy / 100 : null,
        split_fused: values.split_qkv ? [SPLIT_QKV_RULE] : [],
        include_diffs: values.include_diffs,
        include: values.include.trim() || null,
        exclude: values.exclude.trim() || null,
        dtype: values.dtype,
        device: values.device,
      })
      setJob(response.job)
      setActiveJobId(response.job.id)
      setCompletedJobId(null)
      setRequestError(null)
      setStreamError(null)
    } catch (caught: unknown) {
      setRequestError(caught instanceof Error ? caught : new Error('Could not start LoRA extraction'))
    } finally {
      setStartPending(false)
    }
  }

  const stop = async () => {
    if (activeJobId === null) return
    setStopPending(true)
    try {
      const response = await deleteJson<{ job: ExtractJobSnapshot | null }>(
        `/api/lora/extract/jobs/${encodeURIComponent(activeJobId)}`,
        {},
      )
      setJob(response.job)
      setRequestError(null)
    } catch (caught: unknown) {
      setRequestError(caught instanceof Error ? caught : new Error('Could not stop LoRA extraction'))
    } finally {
      setStopPending(false)
    }
  }

  return (
    <Stack gap="xl" maw={820} w="100%" mx="auto">
      <PageHeader
        title="Extract LoRA"
        description="Compress the difference between a fine-tune and its base checkpoint into a LoRA."
      />

      <Card withBorder padding="lg" radius="md">
        <form onSubmit={form.onSubmit(submit)}>
          <Stack>
            <Divider label="Checkpoints" labelPosition="left" />
            <Select
              label="Base checkpoint"
              description="The original model the fine-tune started from. INT8 ConvRot or float."
              placeholder="Choose a configured checkpoint"
              data={modelOptions}
              disabled={formDisabled || modelsQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No safetensors checkpoints found"
              withAsterisk
              {...form.getInputProps('base_path')}
            />
            <Select
              label="Fine-tuned checkpoint"
              description="Tensor names are matched automatically, including a differing key prefix."
              placeholder="Choose a configured checkpoint"
              data={modelOptions}
              disabled={formDisabled || modelsQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No safetensors checkpoints found"
              withAsterisk
              {...form.getInputProps('tuned_path')}
              onChange={(path) => {
                const previous = suggestedName(form.values.tuned_path)
                if (!form.values.output_name || form.values.output_name === previous) {
                  form.setFieldValue('output_name', suggestedName(path))
                }
                form.setFieldValue('tuned_path', path)
              }}
            />
            {modelsQuery.error ? (
              <Alert color="red">
                {modelsQuery.error instanceof Error ? modelsQuery.error.message : 'Could not load models'}
              </Alert>
            ) : !modelsQuery.isPending && models.length === 0 ? (
              <Text size="sm" c="dimmed">Add a model folder in Settings first.</Text>
            ) : null}

            <Divider label="Rank" labelPosition="left" />
            {form.values.adaptive ? (
              <Group align="flex-start" gap="md" wrap="wrap">
                <NumberInput
                  label="Energy to capture (%)"
                  description="Each tensor gets the smallest rank reaching this share of its change."
                  min={1}
                  max={100}
                  disabled={formDisabled}
                  style={{ flex: '1 1 260px' }}
                  {...form.getInputProps('energy')}
                />
                <Select
                  label="Maximum rank"
                  data={RANK_CHOICES}
                  allowDeselect={false}
                  disabled={formDisabled}
                  style={{ flex: '0 1 160px' }}
                  {...form.getInputProps('max_rank')}
                />
              </Group>
            ) : (
              <Stack gap="xs">
                <Text size="sm" fw={500}>Ranks</Text>
                <Text size="sm" c="dimmed">
                  Pick one or more. Each rank writes its own file from one pass; size grows with rank.
                </Text>
                <Chip.Group multiple {...form.getInputProps('ranks')}>
                  <Group gap="xs">
                    {RANK_CHOICES.map((rank) => (
                      <Chip key={rank} value={rank} disabled={formDisabled}>
                        {rank}
                      </Chip>
                    ))}
                  </Group>
                </Chip.Group>
                {form.errors.ranks && <Text size="sm" c="red">{form.errors.ranks}</Text>}
              </Stack>
            )}
            <Switch
              label="Adaptive rank per tensor"
              description="Spend rank where the fine-tune changed the most instead of one rank everywhere."
              disabled={formDisabled}
              {...form.getInputProps('adaptive', { type: 'checkbox' })}
            />
            <Switch
              label="Split fused qkv into q, k and v"
              description="Needed for ComfyUI-style loaders. Leave off to merge into a checkpoint that stores a fused qkv weight."
              disabled={formDisabled}
              {...form.getInputProps('split_qkv', { type: 'checkbox' })}
            />

            <Divider label="Output" labelPosition="left" />
            <Select
              label="Output folder"
              description="LoRA folders let the result appear in Quantize's LoRA list."
              placeholder="Choose a configured folder"
              data={folderOptions}
              disabled={formDisabled || outputFoldersQuery.isPending || loraFoldersQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No folders found"
              withAsterisk
              {...form.getInputProps('output_folder')}
            />
            {folderOptions.length === 0 && !outputFoldersQuery.isPending && !loraFoldersQuery.isPending && (
              <Text size="sm" c="dimmed">Add an output or LoRA folder in Settings first.</Text>
            )}
            <TextInput
              label="Output file name"
              description={
                multipleRanks
                  ? 'A rank suffix such as _r32 is added to each file name.'
                  : '.safetensors is added automatically when omitted'
              }
              placeholder="tuned-delta.safetensors"
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

            <Button
              type="button"
              variant="subtle"
              size="compact-sm"
              style={{ alignSelf: 'flex-start' }}
              onClick={() => setShowAdvanced((open) => !open)}
            >
              {showAdvanced ? 'Hide advanced options' : 'Show advanced options'}
            </Button>
            <Collapse expanded={showAdvanced}>
              <Stack>
                <TextInput
                  label="Only include tensors matching (regex)"
                  placeholder="layers\.(1[0-9]|2[0-9])\."
                  disabled={formDisabled}
                  {...form.getInputProps('include')}
                />
                <TextInput
                  label="Skip tensors matching (regex)"
                  disabled={formDisabled}
                  {...form.getInputProps('exclude')}
                />
                <Select
                  label="Stored LoRA precision"
                  data={[
                    { label: 'FP16', value: 'fp16' },
                    { label: 'BF16', value: 'bf16' },
                    { label: 'FP32 (largest)', value: 'fp32' },
                  ]}
                  allowDeselect={false}
                  disabled={formDisabled}
                  {...form.getInputProps('dtype')}
                />
                <Switch
                  label="Also store changed norms and biases"
                  description="Written as .diff tensors, which LoRA merge supports."
                  disabled={formDisabled}
                  {...form.getInputProps('include_diffs', { type: 'checkbox' })}
                />
              </Stack>
            </Collapse>

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
              <Button type="submit" loading={startPending} disabled={formDisabled} w={{ base: '100%', md: 'auto' }}>
                {formDisabled && !restoring ? 'Extracting' : 'Extract LoRA'}
              </Button>
            </Group>
          </Stack>
        </form>

        {isJobActive && (
          <Paper withBorder p="md" mt="md" aria-live="polite">
            {job?.status === 'stopping' ? (
              <Text>Stopping extraction</Text>
            ) : job?.progress ? (
              <Stack gap="xs">
                <Progress
                  value={job.progress.total > 0 ? (job.progress.current / job.progress.total) * 100 : 0}
                />
                <Text size="sm">
                  {job.progress.current} / {job.progress.total} tensors
                </Text>
                <Text
                  size="sm"
                  title={job.progress.tensor_name}
                  style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                >
                  {job.progress.tensor_name}
                </Text>
              </Stack>
            ) : (
              <Text>Preparing extraction</Text>
            )}
          </Paper>
        )}

        {error && (
          <Alert color="red" mt="md">
            {error}
            {requestError instanceof ApiError && requestError.status === 409 && (
              <Text mt="xs">That destination already exists. Choose a different output name.</Text>
            )}
          </Alert>
        )}
        {jobCancelled && (
          <Alert color="yellow" mt="md">
            Extraction stopped. No LoRA files were written.
          </Alert>
        )}
      </Card>

      {result && (
        <Paper withBorder p="lg" radius="md">
          <Stack gap="lg">
            <Title order={3}>Extraction complete</Title>
            <ExtractResultSummary result={result} />
          </Stack>
        </Paper>
      )}
    </Stack>
  )
}

export const Route = createFileRoute('/extract')({
  component: ExtractPage,
})
