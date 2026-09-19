import { ChartBar, Cube, Gear, MagnifyingGlass, SlidersHorizontal } from '@phosphor-icons/react'
import { AppShell, Burger, Container, Group, NavLink, Title } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'

function RootLayout() {
  const [opened, { toggle }] = useDisclosure(false)
  const pathname = useRouterState({ select: (state) => state.location.pathname })

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 224, breakpoint: 'sm', collapsed: { mobile: !opened } }}
    >
      <AppShell.Header px="lg">
        <Group h="100%" gap="xs">
          <Burger
            opened={opened}
            onClick={toggle}
            hiddenFrom="sm"
            size="sm"
            aria-label="Toggle navigation"
          />
          <Cube size={20} weight="duotone" />
          <Title order={3}>PotatoForge</Title>
        </Group>
      </AppShell.Header>
      <AppShell.Navbar p="sm">
        <NavLink
          component={Link}
          to="/inspect"
          label="Inspect"
          leftSection={<MagnifyingGlass size={18} />}
          active={pathname === '/inspect'}
        />
        <NavLink
          component={Link}
          to="/audit"
          label="Audit"
          leftSection={<ChartBar size={18} />}
          active={pathname === '/audit'}
        />
        <NavLink
          component={Link}
          to="/optimize"
          label="Optimize"
          leftSection={<SlidersHorizontal size={18} />}
          active={pathname === '/optimize'}
        />
        <NavLink
          component={Link}
          to="/quantize"
          label="Quantize"
          leftSection={<Cube size={18} />}
          active={pathname === '/quantize'}
        />
        <NavLink
          component={Link}
          to="/settings"
          label="Settings"
          leftSection={<Gear size={18} />}
          active={pathname === '/settings'}
        />
      </AppShell.Navbar>
      <AppShell.Main>
        <Container size="xl" py="xl">
          <Outlet />
        </Container>
      </AppShell.Main>
    </AppShell>
  )
}

export const Route = createRootRoute({ component: RootLayout })
