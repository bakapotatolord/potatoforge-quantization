import { Alert, Badge, Button, Card, Group, NumberInput, Select, Stack, Table, Text, TextInput, Title } from '@mantine/core'
import { DataTable } from 'mantine-datatable'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'

import { deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'

type Input = { name: string; value: string | number | boolean | null; type: string }
type Node = { id: string; class_type: string; title: string; inputs: Input[] }
type Workflow = { path: string; nodes: Node[] }
type Dataset = { path: string; name: string | null; run_count: number; fields: { name: string; count: number; types: string[] }[] }
type Binding = { dataset_field: string; node_id: string; class_type: string; input_name: string }
type RunRecord = { index: number | string; seed?: string; status: string; prompt_id: string | null; node_id: string | null; error: string | null; cached_nodes: string[]; executed_nodes: string[]; progress?: { value: number; max: number } }
type Job = { id: string; status: string; progress: { current: number; total: number; rows: RunRecord[] } | null; result: { rows: RunRecord[] } | null; error: string | null }

function saved(key: string, fallback = ''): string {
  try { return localStorage.getItem(`comfy-runner-${key}`) ?? fallback } catch { return fallback }
}

function ComfyRunnerPage() {
  const queryClient = useQueryClient()
  const [baseUrl, setBaseUrl] = useState(() => saved('url', 'http://127.0.0.1:8188'))
  const [command, setCommand] = useState(() => saved('command'))
  const [workingDirectory, setWorkingDirectory] = useState(() => saved('directory'))
  const [jsonFolder, setJsonFolder] = useState(() => saved('json-folder'))
  const [jsonFiles, setJsonFiles] = useState<{ name: string; path: string }[]>([])
  const [timeout, setTimeoutValue] = useState<number | string>(120)
  const [workflowPath, setWorkflowPath] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [finalWorkflowPath, setFinalWorkflowPath] = useState('')
  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [bindings, setBindings] = useState<Binding[]>([])
  const [connection, setConnection] = useState<{ system: { system?: { comfyui_version?: string }; devices?: { name: string }[] }; capabilities: { jobs_api: boolean; targeted_cancel: boolean } } | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [preflight, setPreflight] = useState<string | null>(null)
  const jobQuery = useQuery({ queryKey: ['comfy-run'], queryFn: () => getJson<{ job: Job | null }>('/api/comfy/runs/current') })
  const job = jobQuery.data?.job
  const active = job && ['starting', 'running', 'stopping'].includes(job.status)

  useEffect(() => {
    try {
      localStorage.setItem('comfy-runner-url', baseUrl)
      localStorage.setItem('comfy-runner-command', command)
      localStorage.setItem('comfy-runner-directory', workingDirectory)
      localStorage.setItem('comfy-runner-json-folder', jsonFolder)
    } catch { /* Private browsing may disable storage. */ }
  }, [baseUrl, command, workingDirectory, jsonFolder])

  useEffect(() => {
    if (!active) return
    const events = new EventSource(`/api/comfy/runs/${job.id}/events`)
    const refresh = () => { void queryClient.invalidateQueries({ queryKey: ['comfy-run'] }) }
    events.addEventListener('state', refresh)
    events.addEventListener('output', refresh)
    events.addEventListener('complete', refresh)
    events.onerror = refresh
    return () => events.close()
  }, [active, job?.id, queryClient])

  async function act(work: () => Promise<void>) {
    setBusy(true)
    setError('')
    try { await work() } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)) }
    finally { setBusy(false) }
  }

  const choices = workflow?.nodes.flatMap((node) => node.inputs.map((input) => ({
    value: `${node.id}\u0000${input.name}`, label: `${node.title || node.class_type} (${node.id}) · ${input.name} (${input.type})`,
  }))) ?? []
  const rows = job?.progress?.rows ?? job?.result?.rows ?? []
  const runRequest = { base_url: baseUrl, workflow_path: workflowPath, dataset_path: datasetPath, bindings, final_workflow_path: finalWorkflowPath || null }
  const preflightKey = JSON.stringify(runRequest)

  return <Stack gap="lg">
    <PageHeader title="Comfy Runner" description="Run an API workflow sequentially with values from a JSON dataset." />
    {error && <Alert color="red" title="Comfy Runner error">{error}</Alert>}
    <Card withBorder><Stack>
      <Title order={3}>Connection</Title>
      <TextInput label="ComfyUI server URL" value={baseUrl} onChange={(event) => { setBaseUrl(event.currentTarget.value); setConnection(null) }} />
      <Group><Button loading={busy} onClick={() => void act(async () => { setConnection(await postJson('/api/comfy/connect', { base_url: baseUrl })) })}>Connect</Button>
        {connection && <Badge color="green">Connected · {connection.system.system?.comfyui_version || connection.system.devices?.[0]?.name || 'ComfyUI'}</Badge>}</Group>
      <TextInput label="Optional ComfyUI .bat path" value={command} onChange={(event) => setCommand(event.currentTarget.value)} placeholder="C:\\ComfyUI\\run_nvidia_gpu.bat" />
      <TextInput label="Working directory" value={workingDirectory} onChange={(event) => setWorkingDirectory(event.currentTarget.value)} />
      <Group><NumberInput label="Startup timeout (seconds)" value={timeout} min={1} max={600} onChange={setTimeoutValue} />
        <Button variant="light" disabled={!command || !workingDirectory} loading={busy} onClick={() => void act(async () => {
          await postJson('/api/comfy/launch', { base_url: baseUrl, command, working_directory: workingDirectory, timeout_seconds: Number(timeout) })
          setConnection(await postJson('/api/comfy/connect', { base_url: baseUrl }))
        })}>Launch and connect</Button></Group>
    </Stack></Card>
    <Card withBorder><Stack>
      <Title order={3}>JSON files</Title>
      <TextInput label="Folder containing workflow and dataset JSON files" value={jsonFolder} onChange={(event) => { setJsonFolder(event.currentTarget.value); setJsonFiles([]) }} />
      <Button variant="light" disabled={!jsonFolder} loading={busy} onClick={() => void act(async () => {
        const result = await getJson<{ files: { name: string; path: string }[] }>(`/api/comfy/json-files?folder=${encodeURIComponent(jsonFolder)}`)
        setJsonFiles(result.files)
      })}>List JSON files</Button>
      {!!jsonFiles.length && <Text size="sm">{jsonFiles.length} JSON files found. Choose them below or enter full paths.</Text>}
    </Stack></Card>
    <Card withBorder><Stack>
      <Title order={3}>Workflow</Title>
      {!!jsonFiles.length && <Select label="Choose API workflow" searchable data={jsonFiles.map((file) => ({ value: file.path, label: file.name }))} value={jsonFiles.some((file) => file.path === workflowPath) ? workflowPath : null} onChange={(value) => { setWorkflowPath(value || ''); setWorkflow(null) }} />}
      <TextInput label="API workflow JSON path on this computer" value={workflowPath} onChange={(event) => { setWorkflowPath(event.currentTarget.value); setWorkflow(null) }} />
      <Button variant="light" disabled={!workflowPath} loading={busy} onClick={() => void act(async () => {
        setWorkflow(await postJson('/api/comfy/workflows/inspect', { path: workflowPath }))
        setBindings([])
      })}>Inspect workflow</Button>
      {workflow && <Text>{workflow.nodes.length} nodes. Literal inputs are available for binding.</Text>}
      {workflow && <DataTable records={workflow.nodes} idAccessor="id" columns={[
        { accessor: 'id', title: 'Node ID' }, { accessor: 'class_type', title: 'Class' },
        { accessor: 'title' }, { accessor: 'inputs', title: 'Literal inputs', render: (node) => node.inputs.map((input) => `${input.name}: ${String(input.value)}`).join(', ') },
      ]} />}
      {!!jsonFiles.length && <Select label="Choose optional final workflow" clearable searchable data={jsonFiles.map((file) => ({ value: file.path, label: file.name }))} value={jsonFiles.some((file) => file.path === finalWorkflowPath) ? finalWorkflowPath : null} onChange={(value) => setFinalWorkflowPath(value || '')} />}
      <TextInput label="Optional final workflow JSON path" description="Use a separate API workflow to finalize a calibration after all dataset runs succeed." value={finalWorkflowPath} onChange={(event) => setFinalWorkflowPath(event.currentTarget.value)} />
    </Stack></Card>
    <Card withBorder><Stack>
      <Title order={3}>Dataset</Title>
      {!!jsonFiles.length && <Select label="Choose dataset" searchable data={jsonFiles.map((file) => ({ value: file.path, label: file.name }))} value={jsonFiles.some((file) => file.path === datasetPath) ? datasetPath : null} onChange={(value) => { setDatasetPath(value || ''); setDataset(null) }} />}
      <TextInput label="Dataset JSON path on this computer" value={datasetPath} onChange={(event) => { setDatasetPath(event.currentTarget.value); setDataset(null) }} />
      <Button variant="light" disabled={!datasetPath} loading={busy} onClick={() => void act(async () => {
        setDataset(await postJson('/api/comfy/datasets/inspect', { path: datasetPath }))
        setBindings([])
      })}>Inspect dataset</Button>
      {dataset && <><Text>{dataset.name || 'Dataset'} · {dataset.run_count} runs</Text><Table striped><Table.Thead><Table.Tr><Table.Th>Field</Table.Th><Table.Th>Present</Table.Th><Table.Th>Type</Table.Th></Table.Tr></Table.Thead><Table.Tbody>
        {dataset.fields.map((field) => <Table.Tr key={field.name}><Table.Td>{field.name}</Table.Td><Table.Td>{field.count}/{dataset.run_count}</Table.Td><Table.Td>{field.types.join(', ')}</Table.Td></Table.Tr>)}
      </Table.Tbody></Table></>}
    </Stack></Card>
    {workflow && dataset && <Card withBorder><Stack>
      <Title order={3}>Bindings</Title>
      <Text size="sm">Choose which literal workflow input each dataset field changes. Add another row to use a field more than once.</Text>
      {bindings.map((binding, index) => <Group key={index} align="end">
        <Select label="Dataset field" data={dataset.fields.map((field) => field.name)} value={binding.dataset_field || null} onChange={(value) => setBindings((old) => old.map((item, at) => at === index ? { ...item, dataset_field: value || '' } : item))} searchable />
        <Select label="Workflow input" data={choices} value={binding.node_id ? `${binding.node_id}\u0000${binding.input_name}` : null} searchable onChange={(value) => {
          const [nodeId, inputName] = (value || '').split('\u0000')
          const node = workflow.nodes.find((entry) => entry.id === nodeId)
          setBindings((old) => old.map((item, at) => at === index ? { ...item, node_id: nodeId || '', class_type: node?.class_type || '', input_name: inputName || '' } : item))
        }} />
        <Button color="red" variant="subtle" onClick={() => setBindings((old) => old.filter((_, at) => at !== index))}>Remove</Button>
      </Group>)}
      <Button variant="light" onClick={() => setBindings((old) => [...old, { dataset_field: '', node_id: '', class_type: '', input_name: '' }])}>Add binding</Button>
    </Stack></Card>}
    <Card withBorder><Stack>
      <Title order={3}>Run</Title>
      <Group><Button variant="light" disabled={!workflow || !dataset || !bindings.length || !!active} loading={busy} onClick={() => void act(async () => {
        await postJson('/api/comfy/preflight', runRequest)
        setPreflight(preflightKey)
      })}>Check setup</Button>
      {preflight === preflightKey && <Badge color="green">Preflight passed</Badge>}</Group>
      <Group><Button disabled={!workflow || !dataset || !bindings.length || !connection || !!active} loading={busy} onClick={() => void act(async () => {
        await postJson('/api/comfy/runs', runRequest)
        await queryClient.invalidateQueries({ queryKey: ['comfy-run'] })
      })}>Start sequential run</Button>
      {active && <Button color="red" variant="light" disabled={job.status === 'stopping'} onClick={() => void act(async () => {
        await deleteJson(`/api/comfy/runs/${job.id}`, {})
        await queryClient.invalidateQueries({ queryKey: ['comfy-run'] })
      })}>Cancel run</Button>}
      {job && <Badge color={job.status === 'succeeded' ? 'green' : job.status === 'failed' ? 'red' : 'blue'}>{job.status}</Badge>}</Group>
      {job?.error && <Alert color="red">{job.error}</Alert>}
      {job && <Text size="sm">{rows.filter((row) => row.status === 'completed').length}/{rows.length} completed</Text>}
      {!!rows.length && <DataTable records={rows} idAccessor="index" columns={[
        { accessor: 'index', title: 'Run' }, { accessor: 'seed' }, { accessor: 'status' },
        { accessor: 'prompt_id', title: 'Comfy prompt ID' }, { accessor: 'node_id', title: 'Current node' },
        { accessor: 'executed_nodes', title: 'Executed nodes', render: (row) => row.executed_nodes.join(', ') },
        { accessor: 'cached_nodes', title: 'Cached nodes', render: (row) => row.cached_nodes.join(', ') },
        { accessor: 'error', title: 'Error' },
      ]} />}
    </Stack></Card>
  </Stack>
}

export const Route = createFileRoute('/comfy')({ component: ComfyRunnerPage })
