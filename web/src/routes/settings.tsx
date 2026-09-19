import { useState } from 'react'
import { Folder, Trash } from '@phosphor-icons/react'
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'

import { deleteJson, getJson, postJson } from '../api.ts'
import { PageHeader } from '../components/PageHeader.tsx'

type FoldersResponse = {
  folders: string[]
}

type FolderLibraryProps = {
  title: string
  description: string
  label: string
  placeholder: string
  emptyMessage: string
  loadingMessage: string
  folders: string[]
  path: string
  isLoading: boolean
  isAdding: boolean
  error: unknown
  onPathChange: (path: string) => void
  onAdd: () => void
  onRemove: (path: string) => void
}

function FolderLibrary({
  title,
  description,
  label,
  placeholder,
  emptyMessage,
  loadingMessage,
  folders,
  path,
  isLoading,
  isAdding,
  error,
  onPathChange,
  onAdd,
  onRemove,
}: FolderLibraryProps) {
  return (
    <Card withBorder padding="lg" radius="md" h="100%">
      <Stack>
        <Group align="center" justify="space-between" gap="sm">
          <Folder size={22} />
          <Group gap="sm" style={{ flex: 1, minWidth: 0 }}>
            <Stack gap={2} style={{ minWidth: 0 }}>
              <Title order={3}>{title}</Title>
              <Text c="dimmed" size="sm">{description}</Text>
            </Stack>
            <Badge variant="light" color="gray">
              {folders.length} {folders.length === 1 ? 'folder' : 'folders'}
            </Badge>
          </Group>
        </Group>

        <Group align="end" wrap="wrap">
          <TextInput
            label={label}
            placeholder={placeholder}
            value={path}
            onChange={(event) => onPathChange(event.currentTarget.value)}
            style={{ flex: '1 1 220px' }}
          />
          <Button onClick={onAdd} loading={isAdding} disabled={!path.trim()}>
            Add folder
          </Button>
        </Group>

        {isLoading ? (
          <Text size="sm" c="dimmed">{loadingMessage}</Text>
        ) : folders.length > 0 ? (
          <Stack gap="xs">
            <Text size="sm" fw={500}>Configured folders</Text>
            {folders.map((folder) => (
              <Paper key={folder} withBorder p="xs">
                <Group justify="space-between" gap="xs" wrap="nowrap">
                  <Group gap="xs" wrap="nowrap" style={{ minWidth: 0, flex: 1 }}>
                    <Folder size={16} />
                    <Text
                      size="sm"
                      title={folder}
                      style={{
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {folder}
                    </Text>
                  </Group>
                  <ActionIcon
                    color="red"
                    variant="subtle"
                    size="sm"
                    aria-label={'Remove folder ' + folder}
                    onClick={() => onRemove(folder)}
                  >
                    <Trash size={16} />
                  </ActionIcon>
                </Group>
              </Paper>
            ))}
          </Stack>
        ) : (
          <Text size="sm" c="dimmed">{emptyMessage}</Text>
        )}

        {Boolean(error) && (
          <Alert color="red">
            {error instanceof Error ? error.message : 'Could not load ' + title.toLowerCase()}
          </Alert>
        )}
      </Stack>
    </Card>
  )
}

function SettingsPage() {
  const [modelFolderPath, setModelFolderPath] = useState('')
  const [loraFolderPath, setLoraFolderPath] = useState('')
  const [profileFolderPath, setProfileFolderPath] = useState('')
  const [outputFolderPath, setOutputFolderPath] = useState('')
  const queryClient = useQueryClient()

  const modelFoldersQuery = useQuery({
    queryKey: ['model-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/model-folders'),
  })
  const profileFoldersQuery = useQuery({
    queryKey: ['profile-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/profile-folders'),
  })
  const loraFoldersQuery = useQuery({
    queryKey: ['lora-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/lora-folders'),
  })
  const outputFoldersQuery = useQuery({
    queryKey: ['output-folders'],
    queryFn: () => getJson<FoldersResponse>('/api/output-folders'),
  })
  const addModelFolderMutation = useMutation({
    mutationFn: (path: string) => postJson<FoldersResponse>('/api/model-folders', { path }),
    onSuccess: async () => {
      setModelFolderPath('')
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['model-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['models'] }),
      ])
    },
  })
  const removeModelFolderMutation = useMutation({
    mutationFn: (path: string) => deleteJson<FoldersResponse>('/api/model-folders', { path }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['model-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['models'] }),
      ])
    },
  })
  const addLoraFolderMutation = useMutation({
    mutationFn: (path: string) => postJson<FoldersResponse>('/api/lora-folders', { path }),
    onSuccess: async () => {
      setLoraFolderPath('')
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['lora-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['loras'] }),
      ])
    },
  })
  const removeLoraFolderMutation = useMutation({
    mutationFn: (path: string) => deleteJson<FoldersResponse>('/api/lora-folders', { path }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['lora-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['loras'] }),
      ])
    },
  })
  const addProfileFolderMutation = useMutation({
    mutationFn: (path: string) => postJson<FoldersResponse>('/api/profile-folders', { path }),
    onSuccess: async () => {
      setProfileFolderPath('')
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['profile-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['profiles'] }),
      ])
    },
  })
  const removeProfileFolderMutation = useMutation({
    mutationFn: (path: string) => deleteJson<FoldersResponse>('/api/profile-folders', { path }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['profile-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['profiles'] }),
      ])
    },
  })
  const addOutputFolderMutation = useMutation({
    mutationFn: (path: string) => postJson<FoldersResponse>('/api/output-folders', { path }),
    onSuccess: async () => {
      setOutputFolderPath('')
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['output-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['audit-reports'] }),
      ])
    },
  })
  const removeOutputFolderMutation = useMutation({
    mutationFn: (path: string) => deleteJson<FoldersResponse>('/api/output-folders', { path }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['output-folders'] }),
        queryClient.invalidateQueries({ queryKey: ['audit-reports'] }),
      ])
    },
  })

  return (
    <Stack gap="xl">
      <PageHeader
        title="Settings"
        description="Configure the local folders used for checkpoints, LoRAs, profiles, and outputs."
      />

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg" verticalSpacing="lg">
        <FolderLibrary
          title="Model library"
          description="Add folders containing safetensors checkpoints."
          label="Model folder"
          placeholder="C:\models"
          emptyMessage="Add a folder to make checkpoints available in Inspect."
          loadingMessage="Loading model library…"
          folders={modelFoldersQuery.data?.folders ?? []}
          path={modelFolderPath}
          isLoading={modelFoldersQuery.isPending}
          isAdding={addModelFolderMutation.isPending}
          error={
            modelFoldersQuery.error ??
            addModelFolderMutation.error ??
            removeModelFolderMutation.error
          }
          onPathChange={setModelFolderPath}
          onAdd={() => addModelFolderMutation.mutate(modelFolderPath.trim())}
          onRemove={(path) => removeModelFolderMutation.mutate(path)}
        />

        <FolderLibrary
          title="LoRA library"
          description="Add folders containing LoRA safetensors files."
          label="LoRA folder"
          placeholder="C:\loras"
          emptyMessage="Add a folder to make LoRA files available in Quantize."
          loadingMessage="Loading LoRA library…"
          folders={loraFoldersQuery.data?.folders ?? []}
          path={loraFolderPath}
          isLoading={loraFoldersQuery.isPending}
          isAdding={addLoraFolderMutation.isPending}
          error={
            loraFoldersQuery.error ??
            addLoraFolderMutation.error ??
            removeLoraFolderMutation.error
          }
          onPathChange={setLoraFolderPath}
          onAdd={() => addLoraFolderMutation.mutate(loraFolderPath.trim())}
          onRemove={(path) => removeLoraFolderMutation.mutate(path)}
        />

        <FolderLibrary
          title="Profile library"
          description="Add folders containing JSON quantization profiles."
          label="Profile folder"
          placeholder="C:\profiles"
          emptyMessage="Add a folder to make profiles available in Quantize."
          loadingMessage="Loading profile library…"
          folders={profileFoldersQuery.data?.folders ?? []}
          path={profileFolderPath}
          isLoading={profileFoldersQuery.isPending}
          isAdding={addProfileFolderMutation.isPending}
          error={
            profileFoldersQuery.error ??
            addProfileFolderMutation.error ??
            removeProfileFolderMutation.error
          }
          onPathChange={setProfileFolderPath}
          onAdd={() => addProfileFolderMutation.mutate(profileFolderPath.trim())}
          onRemove={(path) => removeProfileFolderMutation.mutate(path)}
        />

        <FolderLibrary
          title="Output folders"
          description="Add folders where checkpoints and audit reports may be written."
          label="Output folder"
          placeholder="C:\output"
          emptyMessage="Add a folder to make it available for Audit and Quantize."
          loadingMessage="Loading output folders…"
          folders={outputFoldersQuery.data?.folders ?? []}
          path={outputFolderPath}
          isLoading={outputFoldersQuery.isPending}
          isAdding={addOutputFolderMutation.isPending}
          error={
            outputFoldersQuery.error ??
            addOutputFolderMutation.error ??
            removeOutputFolderMutation.error
          }
          onPathChange={setOutputFolderPath}
          onAdd={() => addOutputFolderMutation.mutate(outputFolderPath.trim())}
          onRemove={(path) => removeOutputFolderMutation.mutate(path)}
        />
      </SimpleGrid>
    </Stack>
  )
}

export const Route = createFileRoute('/settings')({ component: SettingsPage })
