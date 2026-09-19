export async function getJson<TResponse>(path: string): Promise<TResponse> {
  return request<TResponse>(path, { method: 'GET' })
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

export async function postJson<TResponse>(
  path: string,
  body: unknown,
): Promise<TResponse> {
  return request<TResponse>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function deleteJson<TResponse>(
  path: string,
  body: unknown,
): Promise<TResponse> {
  return request<TResponse>(path, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function request<TResponse>(
  path: string,
  init: RequestInit,
): Promise<TResponse> {
  const response = await fetch(path, init)
  const text = await response.text()
  let data: unknown
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    throw new ApiError(
      `API returned a non-JSON response (${response.status})`,
      response.status,
    )
  }

  if (!response.ok) {
    const message =
      typeof data === 'object' && data !== null && 'detail' in data
        ? String(data.detail)
        : `Request failed (${response.status})`
    throw new ApiError(message, response.status)
  }

  return data as TResponse
}
