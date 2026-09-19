export const joinOutputPath = (folder: string, name: string): string =>
  folder.endsWith('\\') || folder.endsWith('/')
    ? folder + name
    : folder + '\\' + name
