import { Stack, Text, Title } from '@mantine/core'

type PageHeaderProps = {
  title: string
  description: string
}

export const PageHeader = ({ title, description }: PageHeaderProps) => (
  <Stack gap={4}>
    <Title order={2}>{title}</Title>
    <Text c="dimmed">{description}</Text>
  </Stack>
)
