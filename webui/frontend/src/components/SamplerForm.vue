<script setup>
import { computed, ref } from 'vue'
import { Plus, VideoPlay } from '@element-plus/icons-vue'
const props = defineProps({ modelValue: Object, options: { type: Object, required: true }, configName: String, disabledReason: String, launching: String, checkpoints: { type: Array, default: () => [] }, checkpointsLoading: Boolean, initializing: Boolean, canInitialize: Boolean })
const emit = defineEmits(['update:modelValue', 'launch', 'initialize', 'checkpoint-config', 'refresh-checkpoints'])
const explicitPath = ref('')
const sourceError = ref('')
const sourceChoices = computed(() => [...props.checkpoints, ...props.options.adapters.filter(path => !props.checkpoints.some(item => item.path === path)).map(path => ({ path, name: path.replace(/\\/g, '/').split('/').at(-1) }))])
function addCheckpoint() {
  const path = explicitPath.value.trim()
  if (!path) { sourceError.value = '请填写检查点路径。'; return }
  const normalized = path.replace(/\\/g, '/').toLowerCase()
  if (props.options.adapters.some(item => item.replace(/\\/g, '/').toLowerCase() === normalized)) { sourceError.value = '该检查点已选择，不会重复添加。'; return }
  props.options.adapters.push(path)
  explicitPath.value = ''
  sourceError.value = ''
}
const sample = computed(() => props.modelValue?.sample || {})
const validPrompts = computed(() => typeof sample.value.prompt === 'string' || (Array.isArray(sample.value.prompt) && sample.value.prompt.every(value => typeof value === 'string')))
const prompts = computed(() => typeof sample.value.prompt === 'string' ? [sample.value.prompt] : Array.isArray(sample.value.prompt) ? sample.value.prompt : [])
function update(key, value) { emit('update:modelValue', { ...props.modelValue, sample: { ...sample.value, [key]: value === undefined ? null : value } }) }
function setPrompt(index, value) {
  if (typeof sample.value.prompt === 'string') update('prompt', value)
  else { const next = prompts.value.slice(); next[index] = value; update('prompt', next) }
}
function addPrompt() { update('prompt', [...prompts.value, '']) }
function removePrompt(index) { update('prompt', prompts.value.filter((_, position) => position !== index)) }
</script>

<template>
  <el-card class="sampler-card" shadow="never">
    <template #header><div><h2>XY Evaluation · 独立 LoRA 评估</h2><p class="muted">{{ configName }} · 编辑本页不会改变训练预览配置</p></div><el-button id="initialize-evaluation" :disabled="!canInitialize" :loading="initializing" @click="emit('initialize')">从训练表单初始化独立副本</el-button></template>
    <el-alert title="评估是独立任务，不是训练流水线第三阶段。不会补写训练期预览、改写训练输出或改变训练参数。" type="info" show-icon :closable="false" />
    <el-form label-position="top" class="evaluation-form">
      <div class="sampler-topline"><el-form-item label="XY 列 · 评估来源"><el-checkbox id="evaluation-include-base" v-model="options.includeBase">包含 Base 基座（无 LoRA）</el-checkbox></el-form-item><div class="scheduler-note"><span class="eyebrow">SCHEDULER</span><span>MOSS 官方调度器 · 固定</span></div></div>
      <el-form-item label="选择一个或多个 LoRA 检查点 · 按选择顺序排列矩阵列"><div class="checkpoint-select-row"><el-select id="evaluation-checkpoints" v-model="options.adapters" multiple filterable placeholder="选择实际保存的检查点" :loading="checkpointsLoading"><el-option v-for="checkpoint in sourceChoices" :key="checkpoint.path" :value="checkpoint.path" :label="`${checkpoint.run_name ? checkpoint.run_name + ' / ' : ''}${checkpoint.name}${checkpoint.step != null ? ' · step ' + checkpoint.step : ''}`" /></el-select><el-button id="refresh-checkpoints" @click="emit('refresh-checkpoints')" :loading="checkpointsLoading">刷新检查点</el-button></div></el-form-item>
      <div class="checkpoint-explicit"><el-input id="evaluation-checkpoint-path" v-model="explicitPath" placeholder="工作区内 .safetensors 检查点完整路径" @keyup.enter="addCheckpoint" /><el-button id="add-evaluation-checkpoint" @click="addCheckpoint">添加路径</el-button></div><p v-if="sourceError" class="error-text">{{ sourceError }}</p>
      <ol v-if="options.adapters.length" class="checkpoint-sources"><li v-for="(path, index) in options.adapters" :key="path"><code>{{ index + 1 }} · {{ path }}</code><el-button :id="`evaluation-source-config-${index}`" text :loading="initializing" @click="emit('checkpoint-config', path)">从该检查点配置初始化</el-button></li></ol>
      <p class="warning-text">所有检查点必须与独立评估配置的模型架构、适配器算法、rank / alpha / 目标模块匹配；不兼容会报告错误，不会跳过来源。一次 CLI 加载所有选中检查点，基座只生成一列。</p>
      <el-empty v-if="!modelValue" description="请显式从训练表单或所选检查点初始化独立评估配置" />
      <template v-else>
      <div class="sampler-shared-grid"><el-form-item v-for="field in [{ key: 'width', label: 'Width 宽度', min: 16, step: 16 }, { key: 'height', label: 'Height 高度', min: 16, step: 16 }, { key: 'num_inference_steps', label: 'Steps 步数', min: 1, step: 1 }, { key: 'guidance_scale', label: 'CFG 引导强度', step: 0.1 }, { key: 'seed', label: 'Seed 随机种子', step: 1 }]" :key="field.key" :label="field.label"><el-input-number :id="`evaluation-${field.key}`" :model-value="sample[field.key]" @update:model-value="update(field.key, $event)" :step="field.step" :min="field.min" controls-position="right" /><code class="field-path">evaluation.sample.{{ field.key }}</code></el-form-item></div>
      <el-form-item label="Shared negative prompt · 共享负向提示词"><el-input id="evaluation-negative" :model-value="sample.negative_prompt" @update:model-value="update('negative_prompt', $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 5 }" placeholder="所有评估提示词共享；可留空" /></el-form-item>
      <div class="prompt-heading"><h3>Prompt list · 提示词列表</h3><el-tag type="info">{{ prompts.length }} 条 · 不设数量上限</el-tag></div>
      <el-alert v-if="!validPrompts && sample.prompt !== undefined" title="当前 prompt 类型不兼容评估。请在下方独立评估配置中显式调整；原值未改变。" type="warning" :closable="false" show-icon />
      <div v-if="validPrompts" class="prompt-list"><div v-for="(prompt, index) in prompts" :key="index" class="prompt-row"><label :for="`evaluation-prompt-${index}`" class="row-number">{{ String(index + 1).padStart(2, '0') }}</label><el-input :id="`evaluation-prompt-${index}`" :model-value="prompt" @update:model-value="setPrompt(index, $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 10 }" :placeholder="`第 ${index + 1} 条评估提示词 · 描述主体、场景与风格`" /><el-button text type="danger" :disabled="prompts.length === 1" @click="removePrompt(index)" :aria-label="`移除第 ${index + 1} 条提示词`">移除</el-button></div></div>
      <el-button id="add-evaluation-prompt" :icon="Plus" @click="addPrompt" :disabled="!validPrompts && sample.prompt !== undefined">添加提示词</el-button>
      <div class="sampler-output-grid"><el-form-item label="评估任务名称（可选）"><el-input id="evaluation-task-name" v-model="options.name" placeholder="evaluation · 留空自动命名" /></el-form-item><el-form-item label="独立评估输出父目录"><el-input id="evaluation-output-parent" v-model="options.outputParent" placeholder="工作区内输出父目录；自动创建唯一子目录" /></el-form-item></div>
      </template>
      <div class="sampler-footer"><div><p>XY 行：提示词 + 种子；XY 列：基座 + 选中检查点。共享负向、步数、CFG 与尺寸。</p><p class="muted">单 GPU 同时运行一个任务。{{ prompts.length }} 行 × {{ Number(options.includeBase) + options.adapters.length }} 列，共 {{ prompts.length * (Number(options.includeBase) + options.adapters.length) }} 张。</p><p v-if="disabledReason" class="warning-text">{{ disabledReason }}</p></div><el-button id="start-evaluation" type="primary" size="large" :icon="VideoPlay" :loading="launching === 'evaluation'" :disabled="!!launching || !!disabledReason" @click="emit('launch')">开始独立评估</el-button></div>
    </el-form>
  </el-card>
</template>
