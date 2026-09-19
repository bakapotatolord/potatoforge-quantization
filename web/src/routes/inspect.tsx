import { useEffect, useMemo, useState } from 'react'
import {
  ChartBar,
  MagnifyingGlass,
} from '@phosphor-icons/react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Checkbox,
  Card,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { DataTable } from 'mantine-datatable'
import { useMutation, useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'

import { getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { getModels } from '../models.ts'
import { formatBytes } from '../utils/format-bytes.ts'
import { joinOutputPath } from '../utils/output-path.ts'
import { parentPath } from '../utils/parent-path.ts'

type InspectFormValues = {
  model_path: string | null
  output_folder: string | null
  output_name: string
  overwrite: boolean
}

type FoldersResponse = {
  folders: string[]
}

type TensorInfo = {
  dtype: string
  shape: number[]
  data_offsets: number[]
}

type InspectReport = {
  source_path: string
  metadata: Record<string, unknown>
  model_info: Record<string, TensorInfo>
  quantization: {
    summary: string
    layers: Record<string, string>
  }
  summary: Record<string, number>
}

type InspectResponse = {
  result: InspectReport
}

type TensorRecord = {
  name: string
  dtype: string
  shape: string
  size: string
  method: string
}

const RECORDS_PER_PAGE = 50

function InspectPage() {
  const [tensorSearch, setTensorSearch] = useState('')
  const [page, setPage] = useState(1)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })
  const outputFoldersQuery = useQuery({
    queryKey: ['output-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/output-folders'),
  })

  const form = useForm<InspectFormValues>({
    initialValues: {
      model_path: null,
      output_folder: null,
      output_name: '',
      overwrite: false,
    },
    validate: {
      model_path: (value) => (value ? null : 'Model path is required'),
      output_folder: (value, values) =>
        values.output_name.trim() && !value
          ? 'Report folder is required to save a report'
          : null,
      output_name: (value) => {
        const name = value.trim()
        if (!name) return null
        if (name.includes('/') || name.includes('\\')) {
          return 'Enter a filename, not a path'
        }
        return null
      },
    },
  })

  const inspectMutation = useMutation({
    mutationFn: (values: InspectFormValues) =>
      postJson<InspectResponse>('/api/inspect', {
        model_path: values.model_path,
        output:
          values.output_folder && values.output_name.trim()
            ? joinOutputPath(values.output_folder, values.output_name.trim())
            : null,
        overwrite: values.overwrite,
      }),
  })
  const inspect = (values: InspectFormValues) => {
    inspectMutation.reset()
    setPage(1)
    inspectMutation.mutate(values)
  }

  const models = modelsQuery.data?.models ?? []
  const outputFolders = outputFoldersQuery.data?.folders ?? []
  const modelNameCounts = new Map<string, number>()
  for (const model of models) {
    const name = model.name.toLowerCase()
    modelNameCounts.set(name, (modelNameCounts.get(name) ?? 0) + 1)
  }
  const modelOptions = models.map((model) => ({
    label:
      (modelNameCounts.get(model.name.toLowerCase()) ?? 0) > 1
        ? `${model.name} — ${parentPath(model.path)}`
        : model.name,
    value: model.path,
  }))
  useEffect(() => {
    if (form.values.output_folder === null && outputFolders.length > 0) {
      form.setFieldValue('output_folder', outputFolders[0])
    }
  }, [form, outputFolders])
  const error = inspectMutation.error
  const report = inspectMutation.data?.result
  const tensorRecords = useMemo<TensorRecord[]>(() => {
    if (!report) return []

    return Object.entries(report.model_info).map(([name, tensor]) => ({
      name,
      dtype: tensor.dtype,
      shape: `[${tensor.shape.join(', ')}]`,
      size: formatBytes(tensor.data_offsets[1] - tensor.data_offsets[0]),
      method: report.quantization.layers[name] ?? 'unquantized',
    }))
  }, [report])
  const filteredTensorRecords = useMemo(
    () => tensorRecords.filter((tensor) =>
      tensor.name.toLowerCase().includes(tensorSearch.trim().toLowerCase()),
    ),
    [tensorRecords, tensorSearch],
  )
  const pagedTensorRecords = filteredTensorRecords.slice(
    (page - 1) * RECORDS_PER_PAGE,
    page * RECORDS_PER_PAGE,
  )

  return (
    <Stack gap="xl">
      <PageHeader
        title="Inspect checkpoint"
        description="Inspect tensors and optionally save a JSON report."
      />

      <Card withBorder padding="lg" radius="md">
        <form onSubmit={form.onSubmit(inspect)} style={{ height: '100%' }}>
          <Stack h="100%">
            <Group align="center" gap="sm">
              <MagnifyingGlass size={22} />
              <Stack gap={2}>
                <Title order={3}>Inspect checkpoint</Title>
                <Text c="dimmed" size="sm">Choose a discovered model and inspect its tensors.</Text>
              </Stack>
            </Group>

            <Select
              label="Model path"
              placeholder="Choose a configured model"
              data={modelOptions}
              disabled={modelsQuery.isPending}
              searchable
              clearable
              {...form.getInputProps('model_path')}
            />
            <Text size="sm" c="dimmed">
              Model library paths must be configured in{' '}
              <Anchor component={Link} to="/settings">Settings</Anchor>.
            </Text>
            {modelsQuery.error ? (
              <Alert color="red">
                {modelsQuery.error instanceof Error ? modelsQuery.error.message : 'Could not load models'}
              </Alert>
            ) : !modelsQuery.isPending && models.length === 0 ? (
              <Text size="sm" c="dimmed">No safetensors checkpoints found in the configured folders.</Text>
            ) : null}
            <Select
              label="Report folder"
              description="Optional; defaults to the first configured output folder."
              placeholder="Choose a configured output folder"
              data={outputFolders.map((folder) => ({ label: folder, value: folder }))}
              disabled={outputFoldersQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No output folders found"
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
              label="Report filename"
              description="Optional; .json is added automatically when omitted"
              placeholder="checkpoint.json"
              {...form.getInputProps('output_name')}
            />
            <Checkbox
              label="Replace an existing report"
              {...form.getInputProps('overwrite', { type: 'checkbox' })}
            />
            <Group justify="flex-end" mt="auto" w="100%">
              <Button
                type="submit"
                loading={inspectMutation.isPending}
                w={{ base: '100%', md: 'auto' }}
              >
                Inspect
              </Button>
            </Group>
          </Stack>
        </form>

        {error && (
          <Alert color="red" mt="md">
            {error instanceof Error ? error.message : 'Inspection failed'}
          </Alert>
        )}
      </Card>

      {report && (
        <Paper withBorder p="lg" radius="md">
          <Stack gap="lg">
            <Group gap="sm" wrap="nowrap">
              <ChartBar size={22} />
              <Stack gap={2} style={{ minWidth: 0 }}>
                <Title order={3}>Inspection results</Title>
                <Text
                  c="dimmed"
                  size="sm"
                  title={report.source_path}
                  style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                >
                  {report.source_path}
                </Text>
              </Stack>
            </Group>

            <SimpleGrid cols={{ base: 1, xs: 2, sm: 3, lg: 5 }}>
              <Summary label="Tensors" value={String(report.summary.tensor_count)} />
              <Summary label="Groups" value={String(report.summary.group_count)} />
              <Summary label="Metadata" value={String(report.summary.metadata_count)} />
              <Summary label="Payload" value={formatBytes(report.summary.total_raw_payload_bytes)} />
              <Summary label="Quantization" value={report.quantization.summary} />
            </SimpleGrid>

            <TextInput
              label="Find tensor"
              placeholder="blocks.0"
              value={tensorSearch}
              onChange={(event) => {
                setTensorSearch(event.currentTarget.value)
                setPage(1)
              }}
            />
            <DataTable<TensorRecord>
              records={pagedTensorRecords}
              idAccessor="name"
              scrollAreaProps={{ type: 'auto' }}
              page={page}
              onPageChange={setPage}
              totalRecords={filteredTensorRecords.length}
              recordsPerPage={RECORDS_PER_PAGE}
              noRecordsText={tensorSearch ? 'No tensors match this search.' : 'No tensors found.'}
              columns={[
                { accessor: 'name', title: 'Tensor' },
                { accessor: 'dtype', title: 'DType' },
                { accessor: 'shape', title: 'Shape' },
                { accessor: 'size', title: 'Size' },
                {
                  accessor: 'method',
                  title: 'Method',
                  render: ({ method }) => (
                    <Badge size="sm" color={METHOD_BADGE_COLORS[method] ?? 'gray'} variant="light">
                      {method}
                    </Badge>
                  ),
                },
              ]}
              withTableBorder
              withColumnBorders
              striped
              highlightOnHover
            />
          </Stack>
        </Paper>
      )}
    </Stack>
  )
}

const METHOD_BADGE_COLORS: Record<string, string> = {
  unquantized: 'gray',
  int5_convrot: 'grape',
  int8_convrot: 'blue',
  int6_convrot: 'cyan',
  int6_rowwise: 'violet',
  convrot_w4a4: 'orange',
  convrot_w4a4_mse: 'yellow',
}

const Summary = ({ label, value }: { label: string; value: string }) => {
  return (
    <Paper withBorder p="sm">
      <Text c="dimmed" size="sm">{label}</Text>
      <Text fw={500}>{value}</Text>
    </Paper>
  )
}

export const Route = createFileRoute('/inspect')({ component: InspectPage })
