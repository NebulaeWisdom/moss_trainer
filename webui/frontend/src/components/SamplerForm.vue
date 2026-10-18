<script setup>
import { computed } from 'vue'
import { Plus, VideoPlay } from '@element-plus/icons-vue'
const props = defineProps({ modelValue: Object, options: { type: Object, required: true }, configName: String, disabledReason: String, launching: String })
const emit = defineEmits(['update:modelValue', 'launch', 'configuration'])
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
    <template #header><div><h2>Sample studio · 独立采样</h2><p class="muted">使用当前表单：{{ configName }} · 所有提示词共享负向、步数、CFG 和种子</p></div><el-button text @click="emit('configuration')">基础配置</el-button></template>
    <el-empty v-if="!modelValue" description="先在 Configuration 加载模型配置" />
    <el-form v-else label-position="top">
      <div class="sampler-topline"><el-form-item label="采样模式"><el-radio-group v-model="options.sampleMode" id="sample-mode"><el-radio-button value="base">Base 基座</el-radio-button><el-radio-button value="adapter">Adapter 适配器</el-radio-button><el-radio-button value="compare">Compare 对照</el-radio-button></el-radio-group></el-form-item><div class="scheduler-note"><span class="eyebrow">SCHEDULER</span><span>MOSS 官方调度器 · 固定</span></div></div>
      <el-form-item v-if="options.sampleMode !== 'base'" label="适配器检查点路径"><el-input id="sample-adapter" v-model="options.adapter" placeholder="本地 .safetensors 检查点完整路径" /><p class="field-hint">沿用当前配置中的 adapter 算法与维度；对照模式以同一提示词、种子分别生成基座与适配器图片。</p></el-form-item>
      <div class="sampler-shared-grid"><el-form-item v-for="field in [{ key: 'width', label: 'Width 宽度', min: 16, step: 16 }, { key: 'height', label: 'Height 高度', min: 16, step: 16 }, { key: 'num_inference_steps', label: 'Steps 步数', min: 1, step: 1 }, { key: 'guidance_scale', label: 'CFG 引导强度', step: 0.1 }, { key: 'seed', label: 'Seed 随机种子', step: 1 }]" :key="field.key" :label="field.label"><el-input-number :id="`sample-${field.key}`" :model-value="sample[field.key]" @update:model-value="update(field.key, $event)" :step="field.step" :min="field.min" controls-position="right" /><code class="field-path">sample.{{ field.key }}</code></el-form-item></div>
      <el-form-item label="Shared negative prompt · 共享负向提示词"><el-input id="sample-negative" :model-value="sample.negative_prompt" @update:model-value="update('negative_prompt', $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 5 }" placeholder="所有提示词共享；可留空" /></el-form-item>
      <div class="prompt-heading"><h3>Prompt list · 提示词列表</h3><el-tag type="info">{{ prompts.length }} 条 · 不设数量上限</el-tag></div>
      <el-alert v-if="!validPrompts && sample.prompt !== undefined" title="当前 prompt 类型不兼容采样。请在 Configuration 显式调整；原值未改变。" type="warning" :closable="false" show-icon />
      <div v-if="validPrompts" class="prompt-list"><div v-for="(prompt, index) in prompts" :key="index" class="prompt-row"><label :for="`sample-prompt-${index}`" class="row-number">{{ String(index + 1).padStart(2, '0') }}</label><el-input :id="`sample-prompt-${index}`" :model-value="prompt" @update:model-value="setPrompt(index, $event)" type="textarea" :autosize="{ minRows: 2, maxRows: 10 }" :placeholder="`第 ${index + 1} 条提示词 · 描述主体、场景与风格`" /><el-button text type="danger" :disabled="prompts.length === 1" @click="removePrompt(index)" :aria-label="`移除第 ${index + 1} 条提示词`">移除</el-button></div></div>
      <el-button id="add-prompt" :icon="Plus" @click="addPrompt" :disabled="!validPrompts && sample.prompt !== undefined">添加提示词</el-button>
      <div class="sampler-output-grid"><el-form-item label="任务名称（可选）"><el-input id="sample-task-name" v-model="options.sampleName" placeholder="sample · 留空自动命名" /></el-form-item><el-form-item label="输出父目录"><el-input id="sample-output-parent" v-model="options.outputParent" placeholder="工作区内输出父目录；自动创建唯一子目录" /></el-form-item></div>
      <div class="sampler-footer"><div><p>新进程读取当前编辑配置的独立副本；不会改动训练进度或原文件。</p><p class="muted">单 GPU 同时运行一个任务。{{ options.sampleMode === 'compare' ? `预计生成 ${prompts.length * 2} 张图片（基座 + 适配器）。` : `预计生成 ${prompts.length} 张图片。` }}</p><p v-if="disabledReason" class="warning-text">{{ disabledReason }}</p></div><el-button id="start-sample" type="primary" size="large" :icon="VideoPlay" :loading="launching === 'sample'" :disabled="!!launching || !!disabledReason" @click="emit('launch')">开始采样</el-button></div>
    </el-form>
  </el-card>
</template>
