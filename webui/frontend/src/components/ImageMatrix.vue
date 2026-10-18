<script setup>
import { computed, ref, watch } from 'vue'
import ImageGallery from './ImageGallery.vue'
import { active } from '../api'
const props = defineProps({ mode: { type: String, required: true }, images: { type: Array, default: () => [] }, job: Object, config: Object, truncated: Boolean })
const selected = ref(null)
const probe = ref('')
const imageA = ref('')
const imageB = ref('')
const meta = image => image.metadata || {}
const normalizedPath = path => String(path || '').replace(/\\/g, '/').toLowerCase()
const probeFields = ['prompt_index', 'prompt', 'negative_prompt', 'seed', 'width', 'height', 'num_inference_steps', 'guidance_scale']
const signature = data => JSON.stringify(probeFields.map(key => [key, data[key] ?? null]))
const classified = computed(() => {
  const grouped = []
  const ungrouped = []
  for (const image of props.images) {
    const data = meta(image)
    const valid = props.mode === 'training'
      ? data.kind === 'training_preview' && data.probe_complete === true && data.probe_id && Number.isFinite(data.step) && Number.isInteger(data.prompt_index)
      : data.kind === 'evaluation' && Number.isInteger(data.prompt_index) && Number.isFinite(data.seed) && (data.multiplier === 0 || typeof data.source_checkpoint === 'string')
    ;(valid ? grouped : ungrouped).push(image)
  }
  return { grouped, ungrouped }
})
const columns = computed(() => {
  if (props.mode === 'training') return [...new Set(classified.value.grouped.map(image => meta(image).step))].sort((a, b) => a - b).map(step => ({ key: String(step), label: `Step ${step}`, detail: '训练期间预览' }))
  const sources = []
  if (props.job?.include_base) sources.push({ key: 'base', label: 'Base 基座', detail: '无适配器 · multiplier 0' })
  for (const [index, path] of (props.job?.adapters || []).entries()) sources.push({ key: normalizedPath(path), label: `${index + 1}. ${path.replace(/\\/g, '/').split('/').at(-1)}`, detail: path })
  for (const image of classified.value.grouped) {
    const data = meta(image)
    const key = data.multiplier === 0 ? 'base' : normalizedPath(data.source_checkpoint)
    if (!sources.some(source => source.key === key)) sources.push({ key, label: key === 'base' ? 'Base 基座' : data.source_checkpoint.replace(/\\/g, '/').split('/').at(-1), detail: data.source_checkpoint || '无适配器' })
  }
  return sources
})
const rows = computed(() => {
  const result = new Map()
  if (props.mode === 'evaluation' && props.config?.sample) {
    const sample = props.config.sample
    const prompts = typeof sample.prompt === 'string' ? [sample.prompt] : sample.prompt
    if (Array.isArray(prompts) && Number.isFinite(sample.seed)) prompts.forEach((prompt, index) => {
      result.set(`${index}:${sample.seed}`, { key: `${index}:${sample.seed}`, index, prompt, seed: sample.seed, data: sample, cells: new Map(), images: [] })
    })
  }
  for (const image of classified.value.grouped) {
    const data = meta(image)
    const key = props.mode === 'training' ? `${data.probe_id}:${signature(data)}` : `${data.prompt_index}:${data.seed}`
    if (!result.has(key)) result.set(key, { key, index: data.prompt_index, prompt: data.prompt, seed: data.seed, data, cells: new Map(), images: [] })
    const row = result.get(key)
    const column = props.mode === 'training' ? String(data.step) : data.multiplier === 0 ? 'base' : normalizedPath(data.source_checkpoint)
    if (!row.cells.has(column)) row.cells.set(column, [])
    row.cells.get(column).push(image)
    row.images.push(image)
  }
  return [...result.values()].sort((a, b) => a.index - b.index || a.seed - b.seed)
})
const comparable = computed(() => rows.value.filter(row => new Set(row.images.map(image => meta(image).step)).size >= 2))
const chosenProbe = computed(() => comparable.value.find(row => row.key === probe.value))
const compareImages = computed(() => (chosenProbe.value?.images || []).slice().sort((a, b) => meta(a).step - meta(b).step))
const leftImage = computed(() => compareImages.value.find(image => image.id === imageA.value))
const rightImages = computed(() => compareImages.value.filter(image => meta(image).step !== meta(leftImage.value || {}).step))
const rightImage = computed(() => rightImages.value.find(image => image.id === imageB.value))
watch(comparable, value => { if (!value.some(row => row.key === probe.value)) probe.value = value[0]?.key || '' }, { immediate: true })
watch(compareImages, value => { if (!value.some(image => image.id === imageA.value)) imageA.value = value[0]?.id || '' }, { immediate: true })
watch(rightImages, value => { if (!value.some(image => image.id === imageB.value)) imageB.value = value.at(-1)?.id || '' }, { immediate: true })
const metadataEntries = computed(() => selected.value ? Object.entries({ name: selected.value.name, ...meta(selected.value) }) : [])
const text = value => typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value ?? '—')
</script>

<template>
  <section class="image-matrix" :aria-label="mode === 'training' ? '训练预览演化矩阵' : '独立评估 XY 矩阵'">
    <div class="matrix-heading"><div><h3>{{ mode === 'training' ? 'Preview evolution · 训练预览演化' : 'XY results · LoRA 评估矩阵' }}</h3><p class="muted">{{ mode === 'training' ? '每行一个固定探针，列按训练步数递增。仅显示训练期间实际生成的预览；设置变化单独分行。' : '每行一个提示词索引与种子；列为基座和所选检查点。所有结果属于独立评估，不写入训练历史。' }}</p></div><el-tag type="info">{{ classified.grouped.length }} 张 · {{ rows.length }} 行 × {{ columns.length }} 列</el-tag></div>
    <el-alert v-if="truncated" type="warning" show-icon :closable="false" title="服务返回的图片窗口已截断；以下不是完整训练历史 / 评估结果。旧文件未删除。" />
    <el-empty v-if="!columns.length || !rows.length" :image-size="70" :description="mode === 'training' ? '尚无可对齐的训练期预览；需在训练开始前开启 sample.preview' : '尚无评估图片；选择基座 / 检查点并启动独立评估'" />
    <div v-else class="matrix-scroll" tabindex="0" aria-label="可横向滚动的图片矩阵">
      <table class="matrix-table"><thead><tr><th class="matrix-probe-head">{{ mode === 'training' ? 'PROBE / 固定探针' : 'PROMPT × SEED' }}</th><th v-for="column in columns" :key="column.key"><strong>{{ column.label }}</strong><small :title="column.detail">{{ column.detail }}</small></th></tr></thead><tbody><tr v-for="row in rows" :key="row.key"><th scope="row"><span class="eyebrow">PROMPT {{ row.index + 1 }} · SEED {{ row.seed }}</span><p>{{ row.prompt }}</p><small>{{ row.data.width ?? '—' }} × {{ row.data.height ?? '—' }} · steps {{ row.data.num_inference_steps ?? '—' }} · CFG {{ row.data.guidance_scale ?? '—' }}</small><details><summary>探针设置</summary><pre>{{ text(row.data) }}</pre></details></th><td v-for="column in columns" :key="column.key"><template v-if="row.cells.get(column.key)?.length"><button v-for="image in row.cells.get(column.key)" :key="image.id" class="matrix-image-button" @click="selected = image" :aria-label="`放大提示词 ${row.index + 1}，${column.label}`"><img :src="image.url" loading="lazy" :alt="`${row.prompt || image.name} · ${column.label}`" /><span>{{ image.name }}</span></button></template><div v-else class="matrix-missing">{{ active(job) ? '尚未生成' : '无记录' }}<small>无图片文件</small></div></td></tr></tbody></table>
    </div>
    <section v-if="mode === 'training'" class="probe-comparison">
      <h3>A / B · 同探针跨步比较</h3><p class="field-hint">只允许同一提示词索引、文本、负向、种子与采样设置的两个不同训练步；不混入独立评估图片。</p>
      <p v-if="!comparable.length" class="muted">至少需要同一探针在两个不同训练步的真实预览。</p>
      <template v-else><div class="comparison-controls"><el-select id="compare-probe" v-model="probe" aria-label="选择固定探针"><el-option v-for="row in comparable" :key="row.key" :value="row.key" :label="`Prompt ${row.index + 1} · seed ${row.seed} · ${row.prompt}`" /></el-select><el-select id="compare-step-a" v-model="imageA" aria-label="A 侧训练步"><el-option v-for="image in compareImages" :key="image.id" :value="image.id" :label="`A · Step ${meta(image).step} · ${image.name}`" /></el-select><el-select id="compare-step-b" v-model="imageB" aria-label="B 侧训练步"><el-option v-for="image in rightImages" :key="image.id" :value="image.id" :label="`B · Step ${meta(image).step} · ${image.name}`" /></el-select></div><div class="comparison-pair"><article v-for="(image, index) in [leftImage, rightImage]" :key="index"><template v-if="image"><h3>{{ index === 0 ? 'A' : 'B' }} · Step {{ meta(image).step }}</h3><button class="matrix-image-button" @click="selected = image"><img :src="image.url" :alt="`${index === 0 ? 'A' : 'B'} 侧同探针 Step ${meta(image).step}`" /></button></template></article></div></template>
    </section>
    <el-collapse v-if="classified.ungrouped.length" class="unclassified-images"><el-collapse-item :title="`元数据不足 / 不兼容：${classified.ungrouped.length} 张原始图片（不参与矩阵比较）`" name="unclassified"><el-alert type="warning" :closable="false" title="这些实际文件的探针或来源信息不完整，不能保证可比性。保留查看入口，不推测或合并。" /><ImageGallery :images="classified.ungrouped" :job-name="job?.name" /></el-collapse-item></el-collapse>
    <el-dialog :model-value="!!selected" @update:model-value="value => { if (!value) selected = null }" title="Artifact details · 图片及来源" width="min(1120px, 94vw)" destroy-on-close><div v-if="selected" class="lightbox-layout"><div class="lightbox-image"><img :src="selected.url" :alt="meta(selected).prompt || selected.name" /></div><div class="image-metadata"><h3>{{ selected.name }}</h3><dl><div v-for="[key, value] in metadataEntries" :key="key"><dt>{{ key }}</dt><dd>{{ text(value) }}</dd></div></dl></div></div></el-dialog>
  </section>
</template>
