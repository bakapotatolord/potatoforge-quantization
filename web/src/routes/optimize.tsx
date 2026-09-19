import { SlidersHorizontal } from '@phosphor-icons/react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Checkbox,
  Group,
  Loader,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useMutation, useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'

import { ApiError, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'
import { formatBytes } from '../utils/format-bytes.ts'
import { joinOutputPath } from '../utils/output-path.ts'

type OptimizeFormValues = {
  audit_path: string | null
  output_folder: string | null
  output_name: string
  profile_id: string
  target_size_gib: string
  overwrite: boolean
  dry_run: boolean
}

type AuditReportsResponse = {
  reports: Array<{ folder: string; name: string; path: string }>
}

type FoldersResponse = {
  folders: string[]
}

type OptimizeResponse = {
  result: {
    audit_path: string
    profile_path: string | null
    profile_id: string
    target_bytes: number
    estimated_output_bytes: number
    dry_run: boolean
    warning: string
  }
}

function OptimizePage() {
  const auditReportsQuery = useQuery({
    queryKey: ['audit-reports'],
    queryFn: () => getJson<AuditReportsResponse>('/api/audit-reports'),
  })
  const profileFoldersQuery = useQuery({
    queryKey: ['profile-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/profile-folders'),
  })
  const optimizeMutation = useMutation({
    mutationFn: (body: {
      audit_path: string | null
      output_path: string
      profile_id: string
      target_size_gib: number
      overwrite: boolean
      dry_run: boolean
    }) => postJson<OptimizeResponse>('/api/optimize', body),
  })

  const form = useForm<OptimizeFormValues>({
    initialValues: {
      audit_path: null,
      output_folder: null,
      output_name: '',
      profile_id: '',
      target_size_gib: '',
      overwrite: false,
      dry_run: false,
    },
    validate: {
      audit_path: (value) => (value ? null : 'Weight-audit report is required'),
      output_folder: (value) => (value ? null : 'Profile folder is required'),
      output_name: (value) => {
        const name = value.trim()
        if (!name) return 'Profile filename is required'
        if (name.includes('/') || name.includes('\\')) {
          return 'Enter a filename, not a path'
        }
        return null
      },
      profile_id: (value) => (value.trim() ? null : 'Profile ID is required'),
      target_size_gib: (value) => {
        if (!value.trim()) return 'Target size is required'
        const number = Number(value)
        return Number.isFinite(number) && number > 0
          ? null
          : 'Target size must be a finite positive number'
      },
    },
  })

  const reports = auditReportsQuery.data?.reports ?? []
  const profileFolders = profileFoldersQuery.data?.folders ?? []
  const reportGroups = new Map<string, { label: string; value: string }[]>()
  for (const report of reports) {
    const options = reportGroups.get(report.folder) ?? []
    options.push({ label: report.name, value: report.path })
    reportGroups.set(report.folder, options)
  }
  const reportOptions = Array.from(reportGroups, ([group, items]) => ({ group, items }))
  const profileFolderOptions = profileFolders.map((folder) => ({
    label: folder,
    value: folder,
  }))
  const formDisabled = optimizeMutation.isPending
  const result = optimizeMutation.data?.result
  const error = optimizeMutation.error

  const submit = (values: OptimizeFormValues) => {
    if (values.output_folder === null) return
    optimizeMutation.reset()
    optimizeMutation.mutate({
      audit_path: values.audit_path,
      output_path: joinOutputPath(values.output_folder, values.output_name.trim()),
      profile_id: values.profile_id.trim(),
      target_size_gib: Number(values.target_size_gib),
      overwrite: values.overwrite,
      dry_run: values.dry_run,
    })
  }

  return (
    <Stack gap="xl">
      <PageHeader
        title="Optimize profile"
        description="Generate a target-size quantization profile from a weight-audit report."
      />

      <Card withBorder padding="lg" radius="md">
        <form onSubmit={form.onSubmit(submit)}>
          <Stack>
            <Group align="center" gap="sm">
              <SlidersHorizontal size={22} />
              <Stack gap={2}>
                <Title order={3}>Optimize profile</Title>
                <Text c="dimmed" size="sm">
                  Estimate or create a profile from a current format-version-5 weight audit.
                </Text>
              </Stack>
            </Group>

            <Select
              label="Weight-audit report"
              description="Select a current format-version-5 weight audit."
              placeholder="Choose a configured audit report"
              data={reportOptions}
              disabled={formDisabled || auditReportsQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No JSON audit reports found"
              withAsterisk
              {...form.getInputProps('audit_path')}
            />
            {auditReportsQuery.error ? (
              <Alert color="red">
                {auditReportsQuery.error instanceof Error
                  ? auditReportsQuery.error.message
                  : 'Could not load audit reports'}
              </Alert>
            ) : !auditReportsQuery.isPending && reports.length === 0 ? (
              <Text size="sm" c="dimmed">
                No JSON audit reports found in configured output folders. Create one in{' '}
                <Anchor component={Link} to="/audit">
                  Audit
                </Anchor>{' '}
                or configure folders in{' '}
                <Anchor component={Link} to="/settings">
                  Settings
                </Anchor>
                .
              </Text>
            ) : null}

            <Select
              label="Profile folder"
              placeholder="Choose a configured profile folder"
              data={profileFolderOptions}
              disabled={formDisabled || profileFoldersQuery.isPending}
              searchable
              clearable
              nothingFoundMessage="No profile folders found"
              withAsterisk
              {...form.getInputProps('output_folder')}
            />
            {profileFoldersQuery.error ? (
              <Alert color="red">
                {profileFoldersQuery.error instanceof Error
                  ? profileFoldersQuery.error.message
                  : 'Could not load profile folders'}
              </Alert>
            ) : !profileFoldersQuery.isPending && profileFolders.length === 0 ? (
              <Text size="sm" c="dimmed">
                No profile folders found. Configure one in{' '}
                <Anchor component={Link} to="/settings">
                  Settings
                </Anchor>
                .
              </Text>
            ) : null}

            <TextInput
              label="Profile filename"
              description=".json is added automatically when omitted"
              placeholder="source-8.5gib.json"
              disabled={formDisabled}
              withAsterisk
              {...form.getInputProps('output_name')}
              onChange={(event) => {
                const outputName = event.currentTarget.value
                const currentProfileId = form.values.output_name.trim().replace(/\.json$/i, '')
                form.setFieldValue('output_name', outputName)
                if (!form.values.profile_id || form.values.profile_id === currentProfileId) {
                  form.setFieldValue('profile_id', outputName.trim().replace(/\.json$/i, ''))
                }
              }}
            />

            <TextInput
              label="Profile ID"
              placeholder="source-8.5gib"
              disabled={formDisabled}
              withAsterisk
              {...form.getInputProps('profile_id')}
            />

            <TextInput
              label="Target size (GiB)"
              type="number"
              step="any"
              min="0"
              placeholder="8.5"
              disabled={formDisabled}
              withAsterisk
              {...form.getInputProps('target_size_gib')}
            />

            <Checkbox
              label="Replace existing profile"
              disabled={formDisabled}
              {...form.getInputProps('overwrite', { type: 'checkbox' })}
            />

            <Checkbox
              label="Estimate only"
              description="Validate the destination but do not write a profile."
              disabled={formDisabled}
              {...form.getInputProps('dry_run', { type: 'checkbox' })}
            />

            <Group justify="flex-end" mt="sm" w="100%">
              <Button
                type="submit"
                loading={optimizeMutation.isPending}
                disabled={formDisabled}
                w={{ base: '100%', md: 'auto' }}
              >
                {optimizeMutation.isPending
                  ? form.values.dry_run
                    ? 'Estimating profile'
                    : 'Optimizing'
                  : 'Optimize'}
              </Button>
            </Group>
          </Stack>
        </form>

        {optimizeMutation.isPending && (
          <Group gap="sm" mt="md" aria-live="polite">
            <Loader size="sm" />
            <Text>
              {form.values.dry_run ? 'Estimating profile' : 'Optimizing profile'}
            </Text>
          </Group>
        )}

        {error && (
          <Alert color="red" mt="md">
            {error instanceof Error ? error.message : 'Optimization failed'}
            {error instanceof ApiError && error.status === 409 && (
              <Text mt="xs">
                That profile already exists. Choose a different filename or enable replacement.
              </Text>
            )}
          </Alert>
        )}
      </Card>

      {result && (
        <Paper withBorder p="lg" radius="md">
          <Stack gap="lg">
            <Title order={3}>{result.dry_run ? 'Profile estimate' : 'Profile created'}</Title>
            <SimpleGrid cols={{ base: 1, xs: 3 }}>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">
                  Target size
                </Text>
                <Text fw={500} title={String(result.target_bytes)}>
                  {formatBytes(result.target_bytes)}
                </Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">
                  Estimated checkpoint size
                </Text>
                <Text fw={500} title={String(result.estimated_output_bytes)}>
                  {formatBytes(result.estimated_output_bytes)}
                </Text>
              </Paper>
              <Paper withBorder p="sm">
                <Text c="dimmed" size="sm">
                  Profile ID
                </Text>
                <Text fw={500} style={{ overflowWrap: 'anywhere' }}>
                  {result.profile_id}
                </Text>
              </Paper>
            </SimpleGrid>

            {result.dry_run ? (
              <Text c="dimmed">No profile was written.</Text>
            ) : (
              <>
                <Stack gap={4} style={{ minWidth: 0 }}>
                  <Text c="dimmed" size="sm">
                    Profile path
                  </Text>
                  <Text
                    component="div"
                    title={result.profile_path ?? ''}
                    style={{
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      userSelect: 'text',
                    }}
                  >
                    {result.profile_path}
                  </Text>
                </Stack>
                <Alert color="yellow">{result.warning}</Alert>
              </>
            )}
          </Stack>
        </Paper>
      )}
    </Stack>
  )
}

export const Route = createFileRoute('/optimize')({ component: OptimizePage })
