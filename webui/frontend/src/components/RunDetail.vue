<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { CopyDocument, Download, VideoPause } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import MetricChart from './MetricChart.vue'
import ImageMatrix from './ImageMatrix.vue'
import { active, dateTime, downloadText, elapsed, numeric, stageName, statusName, statusType } from '../api'
const props = defineProps({ job: Object, jobConfig: Object, metrics: { type: Object, required: true }, images: Array, imagesTruncated: Boolean, logs: String, logTrimmed: Boolean, theme: String, now: Number })
const emit = defineEmits(['stop', 'snapshot', 'use-cache', 'evaluate'])
const tab = ref('logs')
const tablePage = ref(1)
const follow = ref(true)
const logSearch = ref('')
const logLevel = ref('all')
const logElement = ref(null)
const recentRows = computed(() => props.metrics.series.slice().reverse().slice((tablePage.value - 1) * 50, tablePage.value * 50))
const columns = computed(() => {
  const available = props.metrics.columns
  const priority = ['step', 'epoch', 'loss', 'lr', ...available.filter(key => key.startsWith('lr.')), 'seconds_per_step', 'elapsed_seconds', 'elapsed', 'grad_norm', ...available.filter(key => key.includes('vram'))]
  return [...new Set([...priority.filter(key => key === 'step' || available.includes(key)), ...available])]
})
const lrFields = computed(() => { const keys = props.metrics.columns.filter(key => key.startsWith('lr.')).slice(0, 3); return (keys.length ? keys : ['lr']).map((key, index) => ({ key, label: key, color: ['#9A8EF5', '#A57EAE', '#4C9989'][index] })) })
const columnLabel = key => ({ step: 'Step 步数', epoch: 'Epoch', loss: 'Loss 损失', lr: 'Learning rate', seconds_per_step: '秒 / 步', step_seconds: 'Step seconds', elapsed_seconds: '累计秒数' }[key] || key)
const visibleLogs = computed(() => {
  const term = logSearch.value.toLowerCase()
  return (props.logs || '').split('\n').filter(line => (!term || line.toLowerCase().includes(term)) && (logLevel.value === 'all' || (logLevel.value === 'error' ? /error|exception|traceback|failed/i.test(line) : logLevel.value === 'warning' ? /warn/i.test(line) : /info/i.test(line)))).join('\n')
})
const command = computed(() => (props.job?.command || []).map(part => /[\s"]/g.test(part) ? JSON.stringify(part) : part).join(' '))
async function scrollLog() { await nextTick(); if (follow.value && logElement.value) logElement.value.scrollTop = logElement.value.scrollHeight }
watch(() => props.logs, scrollLog)
watch([follow, tab], scrollLog)
watch(() => props.job?.id, () => { tablePage.value = 1; tab.value = props.job?.stage === 'evaluation' ? 'evaluation' : 'logs' }, { immediate: true })
async function copy(value) { try { await navigator.clipboard.writeText(value); ElMessage.success('已复制') } catch { ElMessage.error('剪贴板不可用，请手动选择复制。') } }
</script>

<template>
  <el-card class="run-detail" shadow="never">
    <el-empty v-if="!job" description="尚未选择任务 · 启动任务后在此查看真实状态与输出" />
    <template v-else>
      <div class="detail-heading"><div><div class="eyebrow">{{ stageName(job.stage) }} / {{ job.id }}</div><h2>{{ job.name || job.id }}</h2></div><div class="inline-actions"><el-tag :type="statusType(job)" aria-live="polite">{{ statusName(job) }}</el-tag><el-button id="stop-selected" type="danger" plain :icon="VideoPause" :disabled="!active(job) || job.status === 'stopping'" @click="emit('stop', job)">{{ job.status === 'stopping' ? '停止中' : '停止任务' }}</el-button></div></div>
      <el-alert v-if="job.error" :title="job.error" type="error" :closable="false" show-icon />
      <el-alert v-if="job.compatibility_note" :title="job.compatibility_note" type="info" :closable="false" show-icon />
      <div v-if="job.stage === 'train' && job.status === 'completed'" class="evaluate-training-action"><el-button id="evaluate-checkpoints" type="primary" plain @click="emit('evaluate', job)">Evaluate checkpoints · 评估已保存检查点</el-button><p class="field-hint">复制该任务配置并选择实际保存的检查点，打开独立 XY 评估。不会补写训练预览；未保存的训练状态无法重建。</p></div>
      <div class="job-facts"><span>PID <b>{{ job.pid ?? '—' }}</b></span><span>运行时间 <b>{{ elapsed(job, now) }}</b></span><span>退出码 <b>{{ job.exit_code ?? '—' }}</b></span><span>创建于 {{ dateTime(job.created_at) }}</span></div>
      <el-collapse class="job-paths"><el-collapse-item title="Command & paths · 实际命令、配置与输出路径" name="paths"><dl><div><dt>配置快照</dt><dd><code>{{ job.config_path }}</code><el-button id="view-job-config" text @click="emit('snapshot', job)">查看快照</el-button></dd></div><div v-if="job.config_origin"><dt>配置来源</dt><dd>{{ job.config_origin }}</dd></div><div><dt>输出目录</dt><dd><code>{{ job.output_dir }}</code><el-button text :icon="CopyDocument" @click="copy(job.output_dir)" aria-label="复制输出路径" /></dd></div><div v-if="job.cache_dir"><dt>缓存目录</dt><dd><code>{{ job.cache_dir }}</code><el-button v-if="job.stage === 'cache' && job.status === 'completed'" id="use-cache" text @click="emit('use-cache', job)">用于训练</el-button></dd></div><div><dt>真实命令</dt><dd><pre class="command-line">{{ command || '等待启动命令记录' }}</pre><el-button v-if="command" text :icon="CopyDocument" @click="copy(command)">复制命令</el-button></dd></div></dl></el-collapse-item></el-collapse>
      <el-tabs v-model="tab" class="detail-tabs">
        <el-tab-pane label="Logs 日志" name="logs">
          <div class="log-toolbar"><el-input id="log-search" v-model="logSearch" placeholder="搜索当前已读取日志" clearable /><el-select v-model="logLevel" aria-label="日志等级"><el-option label="All 全部" value="all" /><el-option label="Info" value="info" /><el-option label="Warning" value="warning" /><el-option label="Error" value="error" /></el-select><el-switch v-model="follow" active-text="跟随" /><el-button :icon="CopyDocument" @click="copy(logs || '')">复制</el-button><el-button :icon="Download" @click="downloadText(`${job.id}-loaded.log`, logs || '')">下载已读日志</el-button></div>
          <p v-if="logTrimmed" class="warning-text">浏览器仅保留最近 1.5 MB；磁盘完整日志保持不变。</p><pre ref="logElement" class="log-viewer" role="log" aria-live="off" aria-label="任务标准输出与错误日志" tabindex="0">{{ visibleLogs || (logs ? '没有匹配的日志行' : '尚无输出 · 后台进程产生 stdout / stderr 后显示') }}</pre>
        </el-tab-pane>
        <el-tab-pane v-if="job.stage === 'train'" label="Charts 曲线" name="charts"><div class="training-charts"><MetricChart :rows="metrics.series" :fields="[{ key: 'loss', label: 'Loss', color: '#D4920A' }]" title="Loss · 训练损失" :theme="theme" /><MetricChart :rows="metrics.series" :fields="lrFields" title="Learning rate · 学习率" :theme="theme" /><MetricChart :rows="metrics.series" :fields="[{ key: 'seconds_per_step', label: 'Seconds / step', color: '#5A8FE0' }]" title="Timing · 单步耗时" :theme="theme" /></div><p class="field-hint">只绘制真实数值。时间轴为训练步数；无数据时不推测损失、ETA 或吞吐。</p></el-tab-pane>
        <el-tab-pane v-if="job.stage === 'train'" label="Metrics 指标" name="metrics"><p class="field-hint">{{ metrics.source || '训练日志尚未产生' }}<span v-if="metrics.truncated"> · 服务仅返回日志中最近的指标窗口</span></p><el-table :data="recentRows" stripe max-height="520" empty-text="暂无真实训练指标" class="metrics-table"><el-table-column v-for="column in columns" :key="column" :prop="column" :label="columnLabel(column)" :fixed="column === 'step' ? 'left' : undefined" min-width="145" align="right"><template #default="{ row }"><span class="mono">{{ numeric(row[column], 6) }}</span></template></el-table-column></el-table><el-pagination v-if="metrics.series.length > 50" v-model:current-page="tablePage" :page-size="50" :total="metrics.series.length" layout="prev, pager, next, total" /></el-tab-pane>
        <el-tab-pane v-if="job.stage === 'train'" label="Preview evolution 预览演化" name="preview"><ImageMatrix mode="training" :images="images" :job="job" :config="jobConfig" :truncated="imagesTruncated" /></el-tab-pane>
        <el-tab-pane v-if="job.stage === 'evaluation'" label="XY results 评估矩阵" name="evaluation"><ImageMatrix mode="evaluation" :images="images" :job="job" :config="jobConfig" :truncated="imagesTruncated" /></el-tab-pane>
      </el-tabs>
    </template>
  </el-card>
</template>
