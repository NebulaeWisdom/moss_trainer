<script setup>
import { computed, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { CopyDocument, FullScreen } from '@element-plus/icons-vue'
const props = defineProps({ images: { type: Array, default: () => [] }, compact: Boolean, jobName: String })
const search = ref('')
const order = ref('newest')
const page = ref(1)
const selected = ref(null)
const meta = image => image.metadata || {}
const prompt = image => meta(image).prompt || meta(image).positive_prompt || ''
const filtered = computed(() => {
  const term = search.value.toLowerCase()
  const list = props.images.filter(image => `${image.name} ${JSON.stringify(meta(image))}`.toLowerCase().includes(term))
  return order.value === 'newest' ? list : [...list].reverse()
})
const shown = computed(() => props.compact ? filtered.value.slice(0, 4) : filtered.value.slice((page.value - 1) * 24, page.value * 24))
const entries = computed(() => selected.value ? Object.entries({ name: selected.value.name, ...(selected.value.width ? { width: selected.value.width, height: selected.value.height } : {}), ...meta(selected.value) }) : [])
watch([search, order], () => { page.value = 1 })
watch(() => props.images, () => { if (page.value > Math.max(1, Math.ceil(filtered.value.length / 24))) page.value = 1 })
function text(value) { return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value ?? '—') }
async function copy(value) {
  try { await navigator.clipboard.writeText(text(value)); ElMessage.success('已复制') }
  catch { ElMessage.error('剪贴板不可用，请在详情中选择并复制文字。') }
}
</script>

<template>
  <div class="gallery">
    <div v-if="!compact" class="gallery-toolbar">
      <el-input v-model="search" placeholder="搜索提示词、种子或文件名" clearable aria-label="搜索生成图片" />
      <el-select v-model="order" aria-label="图片排序"><el-option label="最新文件优先" value="newest" /><el-option label="最早文件优先" value="oldest" /></el-select>
      <span class="muted">{{ filtered.length }} 张 · {{ jobName || '未选择任务' }}</span>
    </div>
    <el-empty v-if="!shown.length" :image-size="90" :description="search ? '没有匹配的图片' : '暂无生成图片 · 选择采样或带预览的训练任务'" />
    <div v-else class="gallery-grid">
      <article v-for="image in shown" :key="image.id" class="image-tile">
        <button class="image-open" @click="selected = image" :aria-label="`放大 ${image.name}`">
          <img :src="image.url" loading="lazy" :alt="prompt(image).slice(0, 120) || image.name" />
          <span class="image-overlay"><FullScreen />查看原图</span>
        </button>
        <div class="image-caption"><p :title="prompt(image) || image.name">{{ prompt(image) || image.name }}</p><div><code>{{ meta(image).seed != null ? `seed ${meta(image).seed}` : image.name }}</code><el-button v-if="prompt(image)" text size="small" :icon="CopyDocument" @click="copy(prompt(image))" aria-label="复制提示词" /></div></div>
      </article>
    </div>
    <el-pagination v-if="!compact && filtered.length > 24" v-model:current-page="page" :total="filtered.length" :page-size="24" layout="prev, pager, next, total" />
    <el-dialog :model-value="!!selected" @update:model-value="value => { if (!value) selected = null }" title="Image details · 生成图片详情" width="min(1120px, 94vw)" class="image-dialog" destroy-on-close>
      <div v-if="selected" class="lightbox-layout">
        <div class="lightbox-image"><img :src="selected.url" :alt="prompt(selected) || selected.name" /></div>
        <div class="image-metadata"><h3>{{ selected.name }}</h3><p class="muted">元数据来自实际生成文件，不推测缺失参数。</p><dl><div v-for="[key, value] in entries" :key="key"><dt>{{ key }}<el-button text size="small" :icon="CopyDocument" @click="copy(value)" :aria-label="`复制 ${key}`" /></dt><dd>{{ text(value) }}</dd></div></dl></div>
      </div>
    </el-dialog>
  </div>
</template>
