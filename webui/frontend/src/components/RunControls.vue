<script setup>
import { computed } from 'vue'
import { VideoPlay, VideoPause, Setting } from '@element-plus/icons-vue'
import { active, elapsed, stageName, statusName, statusType } from '../api'
const props = defineProps({ options: { type: Object, required: true }, jobs: { type: Array, default: () => [] }, configName: String, launching: String, stageError: Function, now: Number })
const emit = defineEmits(['launch', 'stop', 'samples', 'select'])
const latest = computed(() => Object.fromEntries(['cache', 'train', 'sample'].map(stage => [stage, props.jobs.find(job => job.stage === stage && active(job)) || props.jobs.find(job => job.stage === stage)])))
</script>

<template>
  <el-card class="launch-context" shadow="never"><div class="launch-context-title"><h2>Launch context · 启动设置</h2><span class="muted">当前表单：{{ configName }} · 启动时创建独立配置副本</span></div><el-form label-position="top"><el-form-item label="输出父目录 · 所有任务自动创建唯一子目录"><el-input id="output-parent" v-model="options.outputParent" placeholder="工作区内的输出父目录；留空使用服务默认值" /></el-form-item></el-form></el-card>
  <div class="control-grid">
    <el-card v-for="stage in ['cache', 'train', 'sample']" :key="stage" class="control-card" :class="stage" shadow="never">
      <template #header><h2>{{ stageName(stage) }}</h2><el-tag :type="statusType(latest[stage])" size="small">{{ statusName(latest[stage]) }}</el-tag></template>
      <el-form label-position="top">
        <el-form-item label="任务名称（可选）"><el-input :id="`${stage}-name`" v-model="options[`${stage}Name`]" :placeholder="`${stage} · 留空自动命名`" /></el-form-item>
        <template v-if="stage === 'cache'">
          <el-form-item label="缓存父目录"><el-input id="cache-parent" v-model="options.cacheParent" placeholder="工作区内缓存父目录" /></el-form-item>
          <p class="control-hint">只读训练图片，输出独立缓存。不会覆盖已有缓存；完成后将实际缓存路径用于训练。</p>
        </template>
        <template v-else-if="stage === 'train'">
          <el-form-item label="训练缓存目录"><el-input id="train-cache-dir" v-model="options.cacheDir" placeholder="已完成缓存的实际目录" /></el-form-item>
          <el-form-item label="恢复状态文件（可选）"><el-input id="train-resume" v-model="options.resume" placeholder="可信的本地 state.pt 完整路径" /></el-form-item>
          <p v-if="options.resume.trim()" class="warning-text">仅选择可信的本地 state.pt（反序列化可能执行代码）；恢复使用状态文件内的配置和缓存，不使用当前表单参数。需同目录 config.json。</p>
          <p v-else class="control-hint">使用当前编辑配置的独立快照。关闭页面不会中断训练；未保存到检查点的进度在停止后会丢失。</p>
        </template>
        <template v-else>
          <p class="control-hint">基座、适配器或同种子对照采样。提示词列表与共享参数在 Samples 工作区编辑。</p>
          <div class="sample-mode-summary"><el-tag type="primary">{{ { base: 'Base 基座', adapter: 'Adapter 适配器', compare: 'Compare 对照' }[options.sampleMode] }}</el-tag><el-button text :icon="Setting" @click="emit('samples')" id="open-samples">编辑采样参数</el-button></div>
          <p class="control-hint">独立采样不改变训练配置或进度，但与训练共享 GPU；不支持同时启动。</p>
        </template>
      </el-form>
      <div v-if="latest[stage]" class="process-facts"><span>PID <code>{{ latest[stage].pid ?? '—' }}</code></span><span>{{ elapsed(latest[stage], now) }}</span><el-button text size="small" @click="emit('select', latest[stage].id)">查看任务</el-button></div>
      <p class="launch-reason" v-if="stageError(stage)">{{ stageError(stage) }}</p>
      <div class="control-actions"><el-button :id="stage === 'sample' ? 'start-sample-runs' : `start-${stage}`" type="primary" :class="`start-${stage}`" :icon="VideoPlay" :loading="launching === stage" :disabled="!!launching || !!stageError(stage)" @click="emit('launch', stage)">开始{{ { cache: '缓存', train: '训练', sample: '采样' }[stage] }}</el-button><el-button :id="`stop-${stage}`" type="danger" plain :icon="VideoPause" :disabled="!active(latest[stage]) || latest[stage]?.status === 'stopping'" @click="emit('stop', latest[stage])">{{ latest[stage]?.status === 'stopping' ? '停止中' : '停止' }}</el-button></div>
    </el-card>
  </div>
</template>
