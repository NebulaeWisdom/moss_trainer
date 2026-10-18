<script setup>
import { computed, ref, watch } from 'vue'
import ConfigField from './ConfigField.vue'
const props = defineProps({ modelValue: { type: Object, required: true }, schema: { type: Object, default: () => ({}) } })
const emit = defineEmits(['update:modelValue'])
const selectedSection = ref('')
const sections = computed(() => {
  const declared = props.schema.sections || []
  const known = declared.filter(section => Object.hasOwn(props.modelValue, section.key))
  const other = Object.keys(props.modelValue).filter(key => !declared.some(section => section.key === key)).map(key => ({ key, label: key, description: '附加配置 · 保留原始值与类型' }))
  return [...known, ...other]
})
const openedSections = ref([])
watch(() => sections.value.map(section => section.key).join('\n'), (_, previous) => {
  const keys = sections.value.map(section => section.key)
  const oldKeys = previous?.split('\n') || []
  openedSections.value = previous === undefined
    ? keys.filter(key => ['model_dir', 'dataset'].includes(key))
    : [...openedSections.value.filter(key => keys.includes(key)), ...keys.filter(key => !oldKeys.includes(key))]
}, { immediate: true })
const missing = computed(() => (props.schema.sections || []).filter(section => !Object.hasOwn(props.modelValue, section.key)))
function update(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function addSection() {
  if (!selectedSection.value || Object.hasOwn(props.modelValue, selectedSection.value)) return
  update(selectedSection.value, selectedSection.value === 'model_dir' ? '' : {})
  selectedSection.value = ''
}
</script>

<template>
  <div class="schema-form">
    <div class="inline-actions">
      <el-button id="expand-all-config" text size="small" @click="openedSections = sections.map(section => section.key)">展开全部分组</el-button>
      <el-button id="collapse-all-config" text size="small" @click="openedSections = []">收起全部分组</el-button>
    </div>
    <el-collapse v-model="openedSections">
      <el-collapse-item v-for="section in sections" :key="section.key" :name="section.key">
        <template #title><div class="section-title"><span class="section-dot" /><span>{{ section.label }}</span><code>{{ section.key }}</code></div></template>
        <p class="section-description">{{ section.description }}</p>
        <ConfigField :model-value="modelValue[section.key]" :path="section.key" :schema="schema.fields || {}" @update:model-value="update(section.key, $event)" />
      </el-collapse-item>
    </el-collapse>
    <div class="add-section">
      <el-select v-model="selectedSection" filterable allow-create default-first-option placeholder="添加配置分组 / 自定义字段" aria-label="新增配置分组"><el-option v-for="section in missing" :key="section.key" :label="section.label" :value="section.key" /></el-select>
      <el-button @click="addSection" :disabled="!selectedSection">添加分组</el-button>
    </div>
  </div>
</template>
