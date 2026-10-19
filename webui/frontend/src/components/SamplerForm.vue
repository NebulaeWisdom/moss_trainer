<script setup>
import { computed, ref } from 'vue'
import { Plus, VideoPlay } from '@element-plus/icons-vue'
import ConfigField from './ConfigField.vue'
const props = defineProps({ modelValue: Object, options: { type: Object, required: true }, configName: String, disabledReason: String, launching: String, schema: { type: Object, default: () => ({}) }, workspaceDir: String, checkpoints: { type: Array, default: () => [] }, checkpointsLoading: Boolean, initializing: Boolean, canInitialize: Boolean })
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
const components = [{ key: 'transformer', label: 'DiT / Transformer' }, { key: 'vae', label: 'VAE' }, { key: 'text_encoder', label: 'Text encoder 文本编码器' }, { key: 'tokenizer', label: 'Tokenizer 分词器' }]
const textFields = [{ key: 'max_sequence_length', label: '最大文本长度 · tokens', min: 1, step: 1 }, { key: 'gpu_memory_gib', label: '文本编码 GPU 加载预算 · GiB', min: 0, step: 0.5 }, { key: 'cpu_memory_gib', label: '文本编码 CPU 加载预算 · GiB', min: 0, step: 0.5 }]
const schedulerFields = [{ key: 'num_train_timesteps', label: '参考时间网格点数', min: 1, step: 1, hint: '官方调度器时间网格大小，非优化器训练步数。' }, { key: 'shift', label: '推理 Shift', min: 0, step: 0.1, hint: '采样时间步分布的 shift。' }, { key: 'training_shift', label: '参考网格 Shift', min: 0, step: 0.1, hint: '模型参考时间网格偏移，不是训练学习率调度器。' }]
function updateRoot(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function updateGroup(group, key, value) { updateRoot(group, { ...props.modelValue?.[group], [key]: value }) }
function effectivePath(key) {
  const override = props.modelValue?.model_paths?.[key]?.trim()
  const root = props.modelValue?.model_dir?.trim()
  if (!override && !root) return '尚未设置：填写模型根目录或此组件目录'
  const path = (override || `${root.replace(/[/\\]+$/, '')}/${key}`).replace(/\\/g, '/')
  return /^[a-zA-Z]:\//.test(path) || path.startsWith('/') || !props.workspaceDir ? path : `${props.workspaceDir.replace(/[/\\]+$/, '')}/${path}`
}
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
    <template #header><div><h2>XY Evaluation · 独立 LoRA 评估</h2><p class="muted">{{ configName }} · 无需训练配置，直接设置推理模型与参数</p></div><el-button id="initialize-evaluation" text :disabled="!canInitialize" :loading="initializing" @click="emit('initialize')">可选：导入训练表单推理参数</el-button></template>
    <el-alert title="独立推理工作区。模型路径、加载预算、调度器、LoRA 与提示词均在本页设置；不会改变训练预览或训练历史。" type="info" show-icon :closable="false" />
    <el-empty v-if="!modelValue" description="正在读取独立推理默认参数；若连接失败，请使用页面顶部刷新重试" />
    <el-form v-else label-position="top" class="evaluation-form">
      <div class="evaluation-source-grid">
        <section class="inference-panel base-source-panel" aria-label="基础模型与加载路径">
          <div class="inference-panel-heading"><h3>Base model · 基座与组件加载路径</h3><el-checkbox id="evaluation-include-base" v-model="options.includeBase">加入基座对照列</el-checkbox></div>
          <p class="field-hint">基座和全部 LoRA 共用以下模型组件。取消基座对照列也仍需加载模型。</p>
          <el-form-item label="模型包根目录 · Base model directory"><el-input id="evaluation-model-dir" :model-value="modelValue.model_dir" @update:model-value="updateRoot('model_dir', $event)" placeholder="包含 transformer / vae / text_encoder / tokenizer 的模型目录" clearable /><code class="field-path">model_dir · 相对路径以工作区为根</code></el-form-item>
          <div class="component-path-fields"><el-form-item v-for="component in components" :key="component.key" :label="`${component.label} 目录覆盖（可选）`"><el-input :id="`evaluation-model-path-${component.key}`" :model-value="modelValue.model_paths?.[component.key] || ''" @update:model-value="updateGroup('model_paths', component.key, $event.trim() ? $event : null)" :placeholder="`留空继承 <model_dir>/${component.key}`" clearable /><code class="effective-model-path">实际加载目录：{{ effectivePath(component.key) }}</code></el-form-item></div>
          <p class="field-hint">组件覆盖必须是目录，不是权重文件。四个组件均显式指定时可不填写模型包根目录；不复制或修改模型。</p>
        </section>
        <section class="inference-panel checkpoint-source-panel" aria-label="LoRA 评估来源">
          <div class="inference-panel-heading"><h3>LoRA sources · 检查点来源</h3><el-tag type="info">{{ options.adapters.length }} selected</el-tag></div>
          <p class="field-hint">与左侧基座使用相同组件与推理设置。选择顺序就是 XY 矩阵的检查点列顺序。</p>
          <el-form-item label="一个或多个已保存的 LoRA 检查点"><div class="checkpoint-select-row"><el-select id="evaluation-checkpoints" v-model="options.adapters" multiple filterable placeholder="选择实际保存的检查点；基座评估可不选" :loading="checkpointsLoading"><el-option v-for="checkpoint in sourceChoices" :key="checkpoint.path" :value="checkpoint.path" :label="`${checkpoint.run_name ? checkpoint.run_name + ' / ' : ''}${checkpoint.name}${checkpoint.step != null ? ' · step ' + checkpoint.step : ''}`" /></el-select><el-button id="refresh-checkpoints" @click="emit('refresh-checkpoints')" :loading="checkpointsLoading">刷新</el-button></div></el-form-item>
          <div class="checkpoint-explicit"><el-input id="evaluation-checkpoint-path" v-model="explicitPath" placeholder="工作区内 .safetensors 检查点完整路径" @keyup.enter="addCheckpoint" /><el-button id="add-evaluation-checkpoint" @click="addCheckpoint">添加路径</el-button></div><p v-if="sourceError" class="error-text">{{ sourceError }}</p>
          <ol v-if="options.adapters.length" class="checkpoint-sources"><li v-for="(path, index) in options.adapters" :key="path"><code>{{ index + 1 }} · {{ path }}</code><el-button :id="`evaluation-source-config-${index}`" text :loading="initializing" @click="emit('checkpoint-config', path)">可选：导入此检查点推理参数</el-button></li></ol>
          <p class="warning-text">所有检查点须与共用模型架构、适配器算法、rank / alpha / 目标模块匹配。不兼容会报告错误，不跳过来源；一个任务内依次加载全部选中 LoRA。</p>
        </section>
      </div>
      <section class="inference-panel" aria-label="文本编码加载设置"><div class="inference-panel-heading"><h3>Text encoding · 文本编码与加载预算</h3></div><p class="field-hint">实际用于冻结文本编码器的 token 长度与 GPU / CPU 加载预算，均须大于 0。</p><div class="inference-numeric-grid"><el-form-item v-for="field in textFields" :key="field.key" :label="field.label"><el-input-number :id="`evaluation-text-${field.key}`" :model-value="modelValue.text?.[field.key]" @update:model-value="updateGroup('text', field.key, $event)" :min="field.min" :step="field.step" :precision="field.key === 'max_sequence_length' ? 0 : undefined" controls-position="right" /><code class="field-path">text.{{ field.key }}</code></el-form-item></div></section>
      <section v-if="options.adapters.length" class="inference-panel" aria-label="LoRA 结构参数"><div class="inference-panel-heading"><h3>Adapter architecture · 共用适配器结构</h3></div><p class="field-hint">仅在选中 LoRA 时使用。下列参数必须与所有检查点结构一致；kwargs 仍按真实库参数传递。</p><ConfigField :model-value="modelValue.adapter" path="adapter" :schema="schema.fields || {}" @update:model-value="updateRoot('adapter', $event)" /></section>
      <section class="inference-panel" aria-label="官方推理调度器"><div class="inference-panel-heading"><h3>Scheduler · MOSS V2 nearest-grid Euler</h3><el-tag type="info">官方实现 · 固定算法</el-tag></div><p class="warning-text">以下为实际 MOSSImageScheduler 构造参数。修改时必须与模型训练的时间分布对齐；默认 1000 / 5 / 1 保持原官方行为。参考网格 Shift 不是训练优化器调度。</p><div class="inference-numeric-grid"><el-form-item v-for="field in schedulerFields" :key="field.key" :label="field.label"><el-input-number :id="`evaluation-scheduler-${field.key}`" :model-value="sample.scheduler?.[field.key]" @update:model-value="update('scheduler', { ...sample.scheduler, [field.key]: $event })" :min="field.min" :step="field.step" :precision="field.key === 'num_train_timesteps' ? 0 : undefined" controls-position="right" /><p class="field-hint">{{ field.hint }}</p><code class="field-path">sample.scheduler.{{ field.key }}</code></el-form-item></div></section>
      <div class="inference-panel-heading sampling-heading"><h3>Sampling · 共享生成参数</h3></div>
      <div class="sampler-shared-grid"><el-form-item v-for="field in [{ key: 'width', label: 'Width 宽度', min: 16, step: 16, integer: true }, { key: 'height', label: 'Height 高度', min: 16, step: 16, integer: true }, { key: 'num_inference_steps', label: 'Steps 步数', min: 1, step: 1, integer: true }, { key: 'guidance_scale', label: 'CFG 引导强度', min: 0, step: 0.1 }, { key: 'seed', label: 'Seed 随机种子', step: 1, integer: true }]" :key="field.key" :label="field.label"><el-input-number :id="`evaluation-${field.key}`" :model-value="sample[field.key]" @update:model-value="update(field.key, $event)" :step="field.step" :min="field.min" :precision="field.integer ? 0 : undefined" controls-position="right" /><code class="field-path">sample.{{ field.key }}</code></el-form-item></div>
      <el-form-item label="Shared negative prompt · 共享负向提示词"><el-input id="evaluation-negative" :model-value="sample.negative_prompt" @update:model-value="update('negative_prompt', $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 5 }" placeholder="所有评估提示词共享；可留空" /></el-form-item>
      <section class="inference-panel guidance-panel" aria-label="高级跳层引导"><div class="inference-panel-heading"><h3>Advanced guidance · 高级跳层引导</h3></div><p class="field-hint">无条件 CFG 跳层默认 [22]；空列表禁用。SLG 默认关闭（空层列表、强度 0）。作用区间同时约束无条件跳层与 SLG，须满足 0 ≤ start ≤ stop ≤ 1。</p><div class="guidance-fields"><ConfigField v-for="key in ['uncond_skip_layers', 'slg_layers', 'slg_scale', 'slg_start', 'slg_stop']" :key="key" :model-value="sample[key]" :path="`sample.${key}`" :schema="schema.fields || {}" @update:model-value="update(key, $event)" /></div></section>
      <div class="prompt-heading"><h3>Prompt list · 提示词列表</h3><el-tag type="info">{{ prompts.length }} 条 · 不设数量上限</el-tag></div>
      <el-alert v-if="!validPrompts && sample.prompt !== undefined" title="导入的 prompt 不是文本或文本列表。请从有效检查点导入推理参数，或重新加载独立默认配置；不修改原文件。" type="warning" :closable="false" show-icon />
      <div v-if="validPrompts" class="prompt-list"><div v-for="(prompt, index) in prompts" :key="index" class="prompt-row"><label :for="`evaluation-prompt-${index}`" class="row-number">{{ String(index + 1).padStart(2, '0') }}</label><el-input :id="`evaluation-prompt-${index}`" :model-value="prompt" @update:model-value="setPrompt(index, $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 10 }" :placeholder="`第 ${index + 1} 条评估提示词 · 描述主体、场景与风格`" /><el-button text type="danger" :disabled="prompts.length === 1" @click="removePrompt(index)" :aria-label="`移除第 ${index + 1} 条提示词`">移除</el-button></div></div>
      <el-button id="add-evaluation-prompt" :icon="Plus" @click="addPrompt" :disabled="!validPrompts && sample.prompt !== undefined">添加提示词</el-button>
      <div class="sampler-output-grid"><el-form-item label="评估任务名称（可选）"><el-input id="evaluation-task-name" v-model="options.name" placeholder="evaluation · 留空自动命名" /></el-form-item><el-form-item label="独立评估输出父目录"><el-input id="evaluation-output-parent" v-model="options.outputParent" placeholder="工作区内输出父目录；自动创建唯一子目录" /></el-form-item></div>
      <div class="sampler-footer"><div><p>XY 行：提示词 + 种子；XY 列：基座 + 选中检查点。共享负向、步数、CFG 与尺寸。</p><p class="muted">单 GPU 同时运行一个任务。{{ prompts.length }} 行 × {{ Number(options.includeBase) + options.adapters.length }} 列，共 {{ prompts.length * (Number(options.includeBase) + options.adapters.length) }} 张。</p><p v-if="disabledReason" class="warning-text">{{ disabledReason }}</p></div><el-button id="start-evaluation" type="primary" size="large" :icon="VideoPlay" :loading="launching === 'evaluation'" :disabled="!!launching || !!disabledReason" @click="emit('launch')">开始独立评估</el-button></div>
    </el-form>
  </el-card>
</template>
