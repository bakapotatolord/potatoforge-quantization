export const parentPath = (path: string): string => {
  const separator = Math.max(path.lastIndexOf('\\'), path.lastIndexOf('/'))
  return separator > 0 ? path.slice(0, separator) : path
}
