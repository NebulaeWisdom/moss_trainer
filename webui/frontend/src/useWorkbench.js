import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { active, clone, request } from './api'

export function useWorkbench() {
  const stored = key => { try { return localStorage.getItem(`moss.${key}`) } catch { return null } }
  const persist = (key, value) => { try { localStorage.setItem(`moss.${key}`, value) } catch { /* Storage can be disabled. */ } }
  const rememberedPage = stored('page') === 'samples' ? 'evaluations' : stored('page')
  const page = ref(['overview', 'configuration', 'evaluations', 'runs'].includes(rememberedPage) ? rememberedPage : 'overview')
  const theme = ref(stored('theme') === 'light' ? 'light' : 'dark')
  const info = ref(null)
  const schema = ref({ sections: [], fields: {} })
  const configs = ref([])
  const configRecord = ref(null)
  const config = ref(null)
  const configDraftToken = shallowRef({})
  const evaluationConfig = ref(null)
  const evaluationDefaults = ref(null)
  const evaluationOrigin = ref('独立推理草稿 · 正在读取默认参数')
  const evaluationBaseline = ref('')
  const evaluationLoading = ref(false)
  const checkpoints = ref([])
  const checkpointsLoading = ref(false)
  const jobConfig = ref(null)
  const pipelineJobId = ref(stored('pipelineJob') || '')
  const evaluationJobId = ref(stored('evaluationJob') || '')
  const baseline = ref('')
  const jobs = ref([])
  const selectedId = ref(stored('selectedJob') || '')
  const system = ref(null)
  const metrics = ref({ series: [], columns: [], source: null })
  const images = ref([])
  const imagesTruncated = ref(false)
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
  const options = reactive({ cacheParent: '', cacheDir: '', outputParent: '', resume: '', cacheName: '', trainName: '' })
  const evaluationOptions = reactive({ adapters: [], includeBase: true, outputParent: '', name: '' })
  const selectedJob = computed(() => jobs.value.find(job => job.id === selectedId.value) || null)
  const activeJobs = computed(() => jobs.value.filter(active))
  const dirty = computed(() => config.value !== null && JSON.stringify(config.value) !== baseline.value)
  const evaluationDirty = computed(() => evaluationConfig.value !== null && JSON.stringify(evaluationConfig.value) !== evaluationBaseline.value)
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
    if (value === 'evaluations') {
      refreshCheckpoints()
      if (selectedJob.value?.stage !== 'evaluation') selectedId.value = jobs.value.find(job => job.id === evaluationJobId.value && job.stage === 'evaluation')?.id || jobs.value.find(job => job.stage === 'evaluation')?.id || ''
    } else if (value === 'runs' && selectedJob.value?.stage === 'evaluation') {
      selectedId.value = jobs.value.find(job => job.id === pipelineJobId.value && job.stage !== 'evaluation')?.id || jobs.value.find(job => job.stage !== 'evaluation')?.id || ''
    }
    refreshDetails()
    await nextTick()
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
  })
  watch(theme, value => { document.documentElement.classList.toggle('dark', value === 'dark'); document.documentElement.dataset.theme = value; persist('theme', value) }, { immediate: true })
  watch(selectedId, value => {
    persist('selectedJob', value)
    if (selectedJob.value?.stage === 'evaluation') { evaluationJobId.value = value; persist('evaluationJob', value) }
    else if (selectedJob.value) { pipelineJobId.value = value; persist('pipelineJob', value) }
    jobConfig.value = null
    detailsVersion += 1
    metrics.value = { series: [], columns: [], source: null }
    images.value = []
    imagesTruncated.value = false
    logs.value = ''
    logOffset.value = 0
    logTrimmed.value = false
    for (const key of ['metrics', 'images', 'logs', 'job', 'jobConfig']) delete errors[key]
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
        configDraftToken.value = {}
        baseline.value = JSON.stringify(record.config)
        persist('selectedConfig', record.id)
      })
    } finally { loadingConfig.value = false }
  }
  async function refreshConfigs() {
    return attempt('configs', async () => { const data = await request('/configs'); configs.value = data.configs; return data })
  }
  async function refreshCheckpoints() {
    if (checkpointsLoading.value) return
    checkpointsLoading.value = true
    try { return await attempt('checkpoints', async () => { const result = await request('/checkpoints'); checkpoints.value = result.checkpoints; if (result.truncated) ElMessage.warning('检查点扫描窗口已截断；可显式添加未列出的工作区检查点路径。'); return result }) }
    finally { checkpointsLoading.value = false }
  }
  function inferenceDraft(source = {}) {
    const defaults = evaluationDefaults.value
    const pick = (value, keys) => Object.fromEntries(keys.filter(key => value && Object.hasOwn(value, key)).map(key => [key, value[key]]))
    return clone({
      model_dir: Object.hasOwn(source, 'model_dir') ? source.model_dir : defaults.model_dir,
      model_paths: { ...defaults.model_paths, ...pick(source.model_paths, ['transformer', 'vae', 'text_encoder', 'tokenizer']) },
      text: { ...defaults.text, ...pick(source.text, ['max_sequence_length', 'gpu_memory_gib', 'cpu_memory_gib']) },
      adapter: { ...defaults.adapter, ...pick(source.adapter, ['algo', 'rank', 'alpha', 'dropout', 'target_modules', 'kwargs']) },
      sample: {
        ...defaults.sample,
        ...pick(source.sample, ['prompt', 'negative_prompt', 'width', 'height', 'num_inference_steps', 'guidance_scale', 'seed', 'uncond_skip_layers', 'slg_layers', 'slg_scale', 'slg_start', 'slg_stop']),
        scheduler: { ...defaults.sample.scheduler, ...pick(source.sample?.scheduler, ['num_train_timesteps', 'shift', 'training_shift']) },
      },
    })
  }
  async function loadEvaluationDefaults() {
    if (!evaluationDefaults.value) {
      const result = await attempt('evaluationDefaults', () => request('/evaluation-defaults'))
      if (!result) return false
      evaluationDefaults.value = result.config
    }
    if (!evaluationConfig.value) {
      evaluationConfig.value = inferenceDraft()
      evaluationBaseline.value = JSON.stringify(evaluationConfig.value)
      evaluationOrigin.value = '独立推理草稿 · 直接填写本页模型路径即可使用'
    }
    return true
  }
  async function initialize() {
    initializing.value = true
    try {
      const [, , listing] = await Promise.all([
        attempt('info', async () => { info.value = await request('/info'); if (!options.outputParent) options.outputParent = info.value.default_output_root; if (!evaluationOptions.outputParent) evaluationOptions.outputParent = info.value.default_output_root; if (!options.cacheParent) options.cacheParent = `${info.value.workspace_dir.replace(/[/\\]$/, '')}/temp/webui-cache` }),
        attempt('schema', async () => { schema.value = await request('/schema') }),
        refreshConfigs(),
        poll(),
        refreshCheckpoints(),
        loadEvaluationDefaults(),
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
      const stage = selectedJob.value?.stage
      if (!jobConfig.value && stage && stage !== 'cache' && page.value !== 'configuration') work.push(fetchPart('jobConfig', 'config', value => { jobConfig.value = value.config }))
      if (stage === 'train' && (page.value === 'overview' || page.value === 'runs')) work.push(fetchPart('metrics', 'metrics', value => { metrics.value = value }))
      if (stage && stage !== 'cache' && page.value !== 'configuration') work.push(fetchPart('images', 'images', value => { images.value = value.images; imagesTruncated.value = value.truncated === true }))
      if (page.value === 'runs' || page.value === 'evaluations') {
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
          const selected = jobs.value.find(job => job.id === selectedId.value)
          if (!selected || page.value === 'evaluations' && selected.stage !== 'evaluation' || page.value === 'runs' && selected.stage === 'evaluation') selectedId.value = jobs.value.find(job => page.value === 'evaluations' ? job.stage === 'evaluation' : page.value === 'runs' ? job.stage !== 'evaluation' : true)?.id || ''
          lastSuccess.value = new Date().toISOString()
        }),
        attempt('system', async () => { system.value = await request('/system') }),
      ])
      await refreshDetails()
    } finally { polling = false }
  }
  function stageError(stage) {
    const draft = stage === 'evaluation' ? evaluationConfig.value : config.value
    if (!draft) return stage === 'evaluation' ? '独立推理默认参数尚未读取，请刷新本地 API 连接。' : '请先加载配置。'
    if (activeJobs.value.length) return '已有活动 GPU 任务；完成或显式停止后再启动。'
    if (stage === 'train' && options.resume.trim()) return ''
    if (stage !== 'evaluation' && (typeof draft.model_dir !== 'string' || !draft.model_dir.trim())) return '请填写模型目录。'
    if (stage === 'cache' && !draft.dataset?.image_dir) return '请填写 dataset.image_dir。'
    if (stage === 'train' && !options.cacheDir.trim()) return '请填写已完成的缓存目录。'
    if (stage === 'evaluation') {
      const components = ['transformer', 'vae', 'text_encoder', 'tokenizer']
      if ((!draft.model_dir || !draft.model_dir.trim()) && components.some(key => !draft.model_paths?.[key]?.trim())) return '请在本页填写基座模型根目录，或填写全部四个组件目录。'
      if (!Number.isInteger(draft.text?.max_sequence_length) || draft.text.max_sequence_length <= 0 || !Number.isFinite(draft.text.gpu_memory_gib) || draft.text.gpu_memory_gib <= 0 || !Number.isFinite(draft.text.cpu_memory_gib) || draft.text.cpu_memory_gib <= 0) return '文本长度须为正整数，GPU / CPU 加载预算须为正数。'
      const schedule = draft.sample?.scheduler
      if (!Number.isInteger(schedule?.num_train_timesteps) || schedule.num_train_timesteps <= 0 || !Number.isFinite(schedule.shift) || schedule.shift <= 0 || !Number.isFinite(schedule.training_shift) || schedule.training_shift <= 0) return '调度器时间网格须为正整数，shift 与参考网格 shift 须为正数。'
      if (!evaluationOptions.includeBase && !evaluationOptions.adapters.length) return '请至少选择基座或一个 LoRA 检查点。'
      const sample = draft.sample
      const prompts = typeof sample?.prompt === 'string' ? [sample.prompt] : sample?.prompt
      if (!Array.isArray(prompts) || !prompts.length || prompts.some(prompt => typeof prompt !== 'string' || !prompt.trim())) return '请填写每一行提示词；空行不会被静默丢弃。'
      if (typeof sample.negative_prompt !== 'string') return '共享负向提示词必须为文本，可为空。'
      if (!Number.isInteger(sample.width) || sample.width <= 0 || sample.width % 16 || !Number.isInteger(sample.height) || sample.height <= 0 || sample.height % 16) return '宽高必须为正整数且为 16 的倍数。'
      if (!Number.isInteger(sample.num_inference_steps) || sample.num_inference_steps <= 0) return '采样步数必须为正整数。'
      if (!Number.isFinite(sample.guidance_scale) || sample.guidance_scale < 0 || !Number.isInteger(sample.seed)) return '请填写非负 CFG 与整数种子。'
      if (['uncond_skip_layers', 'slg_layers'].some(key => !Array.isArray(sample[key]) || sample[key].some(value => !Number.isInteger(value) || value < 0))) return '跳层列表必须是非负整数数组；空列表表示不启用对应跳层。'
      if (!Number.isFinite(sample.slg_scale) || sample.slg_scale < 0 || !Number.isFinite(sample.slg_start) || !Number.isFinite(sample.slg_stop) || sample.slg_start < 0 || sample.slg_stop > 1 || sample.slg_start > sample.slg_stop) return 'SLG 强度须非负；跳层作用区间须满足 0 ≤ start ≤ stop ≤ 1。'
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
        const evaluation = stage === 'evaluation'
        const payload = { stage, config: clone(evaluation ? evaluationConfig.value : config.value), name: (evaluation ? evaluationOptions.name : options[`${stage}Name`]) || undefined, output_dir: (evaluation ? evaluationOptions.outputParent : options.outputParent).trim() || undefined }
        if (stage === 'cache') payload.cache_dir = options.cacheParent.trim() || undefined
        if (stage === 'train') { payload.cache_dir = options.cacheDir.trim() || undefined; payload.resume = options.resume.trim() || undefined }
        if (evaluation) { payload.adapters = [...evaluationOptions.adapters]; payload.include_base = evaluationOptions.includeBase }
        const job = await request('/jobs', { method: 'POST', body: JSON.stringify(payload) })
        jobs.value = [job, ...jobs.value.filter(item => item.id !== job.id)]
        selectedId.value = job.id
        page.value = evaluation ? 'evaluations' : 'runs'
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
    configDraftToken.value = {}
    configRecord.value = { name: '任务快照', path: snapshot.value.path }
    baseline.value = JSON.stringify(config.value)
    snapshotDialog.value = false
    page.value = 'configuration'
  }
  async function initializeEvaluation(job = null, checkpointPath = null) {
    if (evaluationLoading.value) return
    if (evaluationDirty.value) {
      try { await ElMessageBox.confirm('导入会替换当前独立推理草稿；仅提取模型、文本加载、适配器与采样设置。训练配置和历史文件保持不变。继续？', '导入推理参数', { confirmButtonText: '导入独立草稿', cancelButtonText: '保留草稿', type: 'warning' }) }
      catch { return }
    }
    evaluationLoading.value = true
    try {
      if (!await loadEvaluationDefaults()) return
      let source = config.value
      let origin = `仅导入训练表单中的推理参数 · ${configurationName.value}`
      if (job || checkpointPath) {
        const result = await attempt('evaluationConfig', () => request(checkpointPath ? `/checkpoints/config?path=${encodeURIComponent(checkpointPath)}` : `/jobs/${encodeURIComponent(job.id)}/config`))
        if (!result) return
        source = result.config
        origin = checkpointPath ? `仅导入检查点推理参数 · ${checkpointPath}` : `仅导入训练任务推理参数 · ${job.name || job.id}`
      }
      if (!source) { errors.evaluationConfig = '请先加载训练配置，或选择已完成训练任务。'; return }
      evaluationConfig.value = inferenceDraft(source)
      evaluationBaseline.value = JSON.stringify(evaluationConfig.value)
      evaluationOrigin.value = origin
      if (job) {
        await refreshCheckpoints()
        const root = `${job.output_dir.replace(/\\/g, '/').replace(/\/$/, '')}/`.toLowerCase()
        evaluationOptions.adapters = checkpoints.value.filter(item => item.path.replace(/\\/g, '/').toLowerCase().startsWith(root)).map(item => item.path)
        if (!evaluationOptions.adapters.length) ElMessage.warning('未发现该训练输出中的已保存检查点；请从目录选择或显式添加实际路径。不会重建未保存状态。')
        evaluationOptions.name = `${job.name || 'train'}-evaluation`
      }
      page.value = 'evaluations'
    } finally { evaluationLoading.value = false }
  }
  function useCache(job) {
    options.cacheDir = job.cache_dir || ''
    page.value = 'runs'
    ElMessage.success('已填入训练缓存路径。')
  }
  function beforeUnload(event) { if (dirty.value || evaluationDirty.value) { event.preventDefault(); event.returnValue = '' } }
  function onVisible() { if (!document.hidden) poll() }
  onMounted(() => { initialize(); timer = setInterval(poll, 2000); document.addEventListener('visibilitychange', onVisible); window.addEventListener('beforeunload', beforeUnload) })
  onBeforeUnmount(() => { disposed = true; clearInterval(timer); document.removeEventListener('visibilitychange', onVisible); window.removeEventListener('beforeunload', beforeUnload) })
  return { page, theme, info, schema, configs, configRecord, config, configDraftToken, evaluationConfig, evaluationOrigin, evaluationLoading, evaluationOptions, checkpoints, checkpointsLoading, jobConfig, jobs, selectedId, selectedJob, activeJobs, system, metrics, images, imagesTruncated, logs, logTrimmed, now, lastSuccess, initializing, loadingConfig, launching, stopping, saving, errors, errorList, saveDialog, revisionName, stopTarget, snapshot, snapshotDialog, snapshotLoading, options, dirty, newestMetric, configurationName, initialize, loadConfig, refreshConfigs, refreshCheckpoints, initializeEvaluation, poll, stageError, launch, confirmStop, saveRevision, viewSnapshot, loadSnapshot, useCache }
}
