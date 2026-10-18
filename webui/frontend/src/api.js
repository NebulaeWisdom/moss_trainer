export async function request(path, options = {}) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 20000)
  try {
    const response = await fetch(`/api${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal,
    })
    const contentType = response.headers.get('content-type') || ''
    const data = contentType.includes('application/json') ? await response.json() : await response.text()
    if (!response.ok) {
      const detail = data?.detail ?? data?.error ?? data
      throw new Error(`${response.status} · ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`)
    }
    return data
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('请求超时（20 秒）。请检查本地服务；任务不会因此停止。')
    if (error instanceof TypeError) throw new Error('无法连接本地 API。请检查服务是否运行；后台任务可能仍在继续。')
    throw error
  } finally {
    clearTimeout(timeout)
  }
}

export const clone = value => JSON.parse(JSON.stringify(value))
export const active = job => ['queued', 'running', 'stopping'].includes(job?.status)
export const stageName = stage => ({ cache: 'Cache 缓存', train: 'Train 训练', sample: 'Sample 采样' }[stage] || stage)
export const statusName = job => ({
  queued: 'Queued 等待启动', running: { cache: 'Caching 缓存中', train: 'Training 训练中', sample: 'Sampling 采样中' }[job?.stage] || 'Running 运行中',
  stopping: 'Stopping 停止中', completed: 'Completed 已完成', failed: 'Failed 失败', stopped: 'Stopped 已停止', interrupted: 'Interrupted 已中断',
}[job?.status] || 'Idle 空闲')
export const statusType = job => ({ completed: 'success', failed: 'danger', stopping: 'danger', running: 'warning', queued: 'primary' }[job?.status] || 'info')
export function bytes(value) {
  if (value == null || !Number.isFinite(value)) return '—'
  if (value < 1024) return `${value} B`
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), 4)
  return `${(value / 1024 ** unit).toFixed(1)} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][unit]}`
}
export function numeric(value, digits = 3) {
  if (value == null || !Number.isFinite(value)) return '—'
  return value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toFixed(digits)).toLocaleString('en-US')
}
export function dateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}
export function elapsed(job, now) {
  if (!job?.started_at) return '尚未开始'
  const seconds = Math.max(0, Math.floor(((job.finished_at ? new Date(job.finished_at).valueOf() : now) - new Date(job.started_at).valueOf()) / 1000))
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds / 60) % 60}m ${seconds % 60}s`
}
export function downloadText(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
