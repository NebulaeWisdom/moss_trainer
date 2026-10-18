import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { active, clone, request } from './api'

export function useWorkbench() {
  const stored = key => { try { return localStorage.getItem(`moss.${key}`) } catch { return null } }
  const persist = (key, value) => { try { localStorage.setItem(`moss.${key}`, value) } catch { /* Storage can be disabled. */ } }
  const page = ref(['overview', 'configuration', 'samples', 'runs'].includes(stored('page')) ? stored('page') : 'overview')
  const theme = ref(stored('theme') === 'light' ? 'light' : 'dark')
  const info = ref(null)
  const schema = ref({ sections: [], fields: {} })
  const configs = ref([])
  const configRecord = ref(null)
  const config = ref(null)
  const baseline = ref('')
  const jobs = ref([])
  const selectedId = ref(stored('selectedJob') || '')
  const system = ref(null)
  const metrics = ref({ series: [], columns: [], source: null })
  const images = ref([])
  const logs = ref('')
  const logTrimmed = ref(false)
  const logOffset = ref(0)
  const now = ref(Date.now())
  const lastSuccess = ref(null)
  const initializing = ref(true)
  const loadingConfig = ref(false)
  const launching = ref('')
  const stopping = ref(false)
  const saving = ref(false)
  const errors = reactive({})
  const saveDialog = ref(false)
  const revisionName = ref('')
  const stopTarget = ref(null)
  const snapshot = ref(null)
  const snapshotDialog = ref(false)
  const snapshotLoading = ref(false)
  const options = reactive({ cacheParent: '', cacheDir: '', outputParent: '', resume: '', adapter: '', sampleMode: 'base', cacheName: '', trainName: '', sampleName: '' })
  const selectedJob = computed(() => jobs.value.find(job => job.id === selectedId.value) || null)
  const activeJobs = computed(() => jobs.value.filter(active))
  const dirty = computed(() => config.value !== null && JSON.stringify(config.value) !== baseline.value)
  const newestMetric = computed(() => metrics.value.series.at(-1) || {})
  const errorList = computed(() => Object.entries(errors).map(([key, message]) => ({ key, message })))
  const configurationName = computed(() => configRecord.value?.name || '未加载配置')
  let timer
  let polling = false
  let disposed = false
  let detailsVersion = 0
  let detailsPending = ''

  watch(page, async value => {
    persist('page', value)
    refreshDetails()
    await nextTick()
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
  })
  watch(theme, value => { document.documentElement.classList.toggle('dark', value === 'dark'); document.documentElement.dataset.theme = value; persist('theme', value) }, { immediate: true })
  watch(selectedId, value => {
    persist('selectedJob', value)
    detailsVersion += 1
    metrics.value = { series: [], columns: [], source: null }
    images.value = []
    logs.value = ''
    logOffset.value = 0
    logTrimmed.value = false
    for (const key of ['metrics', 'images', 'logs', 'job']) delete errors[key]
    refreshDetails()
  })

  async function attempt(key, fn) {
    try { const value = await fn(); delete errors[key]; return value }
    catch (error) {
      errors[key] = error.message
      if (['launch', 'save', 'stop', 'config', 'snapshot'].includes(key)) ElMessage.error({ message: error.message, duration: 7000, showClose: true })
      return null
    }
  }
  async function loadConfig(id, confirm = true) {
    if (!id || loadingConfig.value) return
    if (confirm && dirty.value) {
      try { await ElMessageBox.confirm('加载另一份配置会替换尚未保存的表单编辑；原文件不会改变。继续加载？', '未保存的编辑', { confirmButtonText: '加载配置', cancelButtonText: '保留编辑', type: 'warning' }) }
      catch { return }
    }
    loadingConfig.value = true
    try {
      await attempt('config', async () => {
        const record = await request(`/config?id=${encodeURIComponent(id)}`)
        configRecord.value = record
        config.value = clone(record.config)
        baseline.value = JSON.stringify(record.config)
        persist('selectedConfig', record.id)
      })
    } finally { loadingConfig.value = false }
  }
  async function refreshConfigs() {
    return attempt('configs', async () => { const data = await request('/configs'); configs.value = data.configs; return data })
  }
  async function initialize() {
    initializing.value = true
    try {
      const [, , listing] = await Promise.all([
        attempt('info', async () => { info.value = await request('/info'); if (!options.outputParent) options.outputParent = info.value.default_output_root; if (!options.cacheParent) options.cacheParent = `${info.value.workspace_dir.replace(/[/\\]$/, '')}/temp/webui-cache` }),
        attempt('schema', async () => { schema.value = await request('/schema') }),
        refreshConfigs(),
        poll(),
      ])
      if (!config.value && listing?.configs?.length) {
        const remembered = listing.configs.find(item => item.id === stored('selectedConfig'))
        await loadConfig(remembered?.id || listing.default_id || listing.configs[0].id, false)
      }
    } finally { initializing.value = false }
  }
  async function refreshDetails() {
    const id = selectedId.value
    const version = detailsVersion
    if (!id || disposed || document.hidden || detailsPending === `${id}:${version}`) return
    const marker = `${id}:${version}`
    detailsPending = marker
    const current = () => selectedId.value === id && detailsVersion === version && !disposed
    const fetchPart = async (key, suffix, apply) => {
      try { const result = await request(`/jobs/${encodeURIComponent(id)}/${suffix}`); if (current()) { apply(result); delete errors[key] } }
      catch (error) { if (current()) errors[key] = error.message }
    }
    try {
      const work = []
      if (page.value === 'overview' || page.value === 'runs') work.push(fetchPart('metrics', 'metrics', value => { metrics.value = value }))
      if (page.value !== 'configuration') work.push(fetchPart('images', 'images', value => { images.value = value.images }))
      if (page.value === 'runs') {
        const offset = logOffset.value
        work.push(fetchPart('logs', `logs?offset=${offset}`, value => {
          if (value.offset < offset) logs.value = ''
          const next = logs.value + value.text
          if (next.length > 1500000) logTrimmed.value = true
          logs.value = next.slice(-1500000)
          logOffset.value = value.offset
        }))
      }
      await Promise.all(work)
    } finally { if (detailsPending === marker) detailsPending = '' }
  }
  async function poll() {
    if (polling || disposed || document.hidden) return
    polling = true
    now.value = Date.now()
    try {
      await Promise.all([
        attempt('jobs', async () => {
          const data = await request('/jobs')
          jobs.value = data.jobs
          if (!selectedId.value && jobs.value.length) selectedId.value = jobs.value[0].id
          lastSuccess.value = new Date().toISOString()
        }),
        attempt('system', async () => { system.value = await request('/system') }),
      ])
      await refreshDetails()
    } finally { polling = false }
  }
  function stageError(stage) {
    if (!config.value) return '请先加载配置。'
    if (activeJobs.value.length) return '已有活动 GPU 任务；完成或显式停止后再启动。'
    if (stage === 'train' && options.resume.trim()) return ''
    if (typeof config.value.model_dir !== 'string' || !config.value.model_dir.trim()) return '请在配置中填写模型目录。'
    if (stage === 'cache' && !config.value.dataset?.image_dir) return '请填写 dataset.image_dir。'
    if (stage === 'train' && !options.cacheDir.trim()) return '请填写已完成的缓存目录。'
    if (stage === 'sample') {
      if (options.sampleMode !== 'base' && !options.adapter.trim()) return '适配器 / 对照模式需要检查点路径。'
      const sample = config.value.sample
      const prompts = typeof sample?.prompt === 'string' ? [sample.prompt] : sample?.prompt
      if (!Array.isArray(prompts) || !prompts.length || prompts.some(prompt => typeof prompt !== 'string' || !prompt.trim())) return '请填写每一行提示词；空行不会被静默丢弃。'
      if (typeof sample.negative_prompt !== 'string') return '共享负向提示词必须为文本，可为空。'
      if (!Number.isInteger(sample.width) || sample.width <= 0 || sample.width % 16 || !Number.isInteger(sample.height) || sample.height <= 0 || sample.height % 16) return '宽高必须为正整数且为 16 的倍数。'
      if (!Number.isInteger(sample.num_inference_steps) || sample.num_inference_steps <= 0) return '采样步数必须为正整数。'
      if (!Number.isFinite(sample.guidance_scale) || !Number.isInteger(sample.seed)) return '请填写有效的 CFG 与整数种子。'
    }
    return ''
  }
  async function launch(stage) {
    const reason = stageError(stage)
    if (reason) { errors.launch = reason; return }
    if (launching.value) return
    launching.value = stage
    try {
      await attempt('launch', async () => {
        const payload = { stage, config: clone(config.value), name: options[`${stage}Name`] || undefined, output_dir: options.outputParent.trim() || undefined }
        if (stage === 'cache') payload.cache_dir = options.cacheParent.trim() || undefined
        if (stage === 'train') { payload.cache_dir = options.cacheDir.trim() || undefined; payload.resume = options.resume.trim() || undefined }
        if (stage === 'sample') { payload.sample_mode = options.sampleMode; payload.adapter = options.sampleMode === 'base' ? undefined : options.adapter.trim() }
        const job = await request('/jobs', { method: 'POST', body: JSON.stringify(payload) })
        jobs.value = [job, ...jobs.value.filter(item => item.id !== job.id)]
        selectedId.value = job.id
        page.value = 'runs'
        ElMessage.success('任务已提交；使用独立配置快照与新输出目录。')
        await refreshDetails()
      })
    } finally { launching.value = '' }
  }
  async function confirmStop() {
    if (!stopTarget.value || stopping.value) return
    const target = stopTarget.value
    stopping.value = true
    try {
      const result = await attempt('stop', async () => {
        const job = await request(`/jobs/${encodeURIComponent(target.id)}/stop`, { method: 'POST' })
        jobs.value = jobs.value.map(item => item.id === job.id ? job : item)
        return job
      })
      if (result) { stopTarget.value = null; ElMessage.warning('已请求停止指定任务；最终状态将在轮询中更新。') }
    } finally { stopping.value = false }
  }
  async function saveRevision() {
    if (!revisionName.value.trim() || !config.value || saving.value) return
    const savedSnapshot = clone(config.value)
    saving.value = true
    try {
      const record = await attempt('save', () => request('/configs', { method: 'POST', body: JSON.stringify({ name: revisionName.value.trim(), config: savedSnapshot }) }))
      if (record) {
        configRecord.value = record
        baseline.value = JSON.stringify(record.config)
        configs.value = [{ id: record.id, name: record.name, path: record.path }, ...configs.value.filter(item => item.id !== record.id)]
        persist('selectedConfig', record.id)
        saveDialog.value = false
        ElMessage.success('已保存为新版本；原文件未覆盖。')
      }
    } finally { saving.value = false }
  }
  async function viewSnapshot(job) {
    snapshotDialog.value = true
    snapshotLoading.value = true
    snapshot.value = null
    try { snapshot.value = await attempt('snapshot', () => request(`/jobs/${encodeURIComponent(job.id)}/config`)) }
    finally { snapshotLoading.value = false }
  }
  async function loadSnapshot() {
    if (!snapshot.value?.config) return
    if (dirty.value) {
      try { await ElMessageBox.confirm('将任务快照载入表单，会替换尚未保存的编辑。继续？', '加载独立快照', { confirmButtonText: '加载快照', cancelButtonText: '保留编辑', type: 'warning' }) }
      catch { return }
    }
    config.value = clone(snapshot.value.config)
    configRecord.value = { name: '任务快照', path: snapshot.value.path }
    baseline.value = JSON.stringify(config.value)
    snapshotDialog.value = false
    page.value = 'configuration'
  }
  function useCache(job) {
    options.cacheDir = job.cache_dir || ''
    page.value = 'runs'
    ElMessage.success('已填入训练缓存路径。')
  }
  function beforeUnload(event) { if (dirty.value) { event.preventDefault(); event.returnValue = '' } }
  function onVisible() { if (!document.hidden) poll() }
  onMounted(() => { initialize(); timer = setInterval(poll, 2000); document.addEventListener('visibilitychange', onVisible); window.addEventListener('beforeunload', beforeUnload) })
  onBeforeUnmount(() => { disposed = true; clearInterval(timer); document.removeEventListener('visibilitychange', onVisible); window.removeEventListener('beforeunload', beforeUnload) })
  return { page, theme, info, schema, configs, configRecord, config, jobs, selectedId, selectedJob, activeJobs, system, metrics, images, logs, logTrimmed, now, lastSuccess, initializing, loadingConfig, launching, stopping, saving, errors, errorList, saveDialog, revisionName, stopTarget, snapshot, snapshotDialog, snapshotLoading, options, dirty, newestMetric, configurationName, initialize, loadConfig, refreshConfigs, poll, stageError, launch, confirmStop, saveRevision, viewSnapshot, loadSnapshot, useCache }
}
