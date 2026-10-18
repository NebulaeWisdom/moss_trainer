<script setup>
import { computed, ref } from 'vue'

const props = defineProps({ modelValue: { required: true }, path: { type: String, required: true }, schema: { type: Object, default: () => ({}) }, compact: Boolean })
const emit = defineEmits(['update:modelValue'])
const hint = computed(() => props.schema[props.path] || {})
const kind = computed(() => props.modelValue === null ? 'null' : Array.isArray(props.modelValue) ? 'array' : typeof props.modelValue)
const id = computed(() => `cfg-${props.path.replace(/[^a-zA-Z0-9_-]/g, '-')}`)
const label = computed(() => hint.value.label || props.path.split('.').at(-1))
const keyName = ref('')
const newType = ref('string')
const addError = ref('')
const types = [{ value: 'string', label: '文本' }, { value: 'number', label: '数字' }, { value: 'boolean', label: '开关' }, { value: 'null', label: '未设置 null' }, { value: 'array', label: '列表' }, { value: 'object', label: '对象' }]
const initial = type => ({ string: '', number: 0, boolean: false, null: null, array: [], object: {} })[type]
const suggestions = computed(() => Object.keys(props.schema).filter(path => path.startsWith(`${props.path}.`) && !path.slice(props.path.length + 1).includes('.')).map(path => path.slice(props.path.length + 1)).filter(key => !Object.hasOwn(props.modelValue || {}, key)))
const multiline = computed(() => /prompt|caption$/.test(props.path) || (typeof props.modelValue === 'string' && props.modelValue.includes('\n')))
function update(value) { emit('update:modelValue', value === undefined ? null : value) }
function setType(type) { update(initial(type)) }
function setChild(key, value) {
  const next = Array.isArray(props.modelValue) ? props.modelValue.slice() : { ...props.modelValue }
  Object.defineProperty(next, key, { value, enumerable: true, writable: true, configurable: true })
  update(next)
}
function removeChild(key) {
  const next = Array.isArray(props.modelValue) ? props.modelValue.filter((_, index) => index !== key) : { ...props.modelValue }
  if (!Array.isArray(next)) delete next[key]
  update(next)
}
function addKey() {
  const key = keyName.value.trim()
  if (!key) { addError.value = '请填写字段名称。'; return }
  if (Object.hasOwn(props.modelValue, key)) { addError.value = '字段已存在，不会覆盖。'; return }
  setChild(key, initial(newType.value))
  keyName.value = ''
  addError.value = ''
}
function addRow() {
  const first = props.modelValue[0]
  const type = first === null ? 'null' : Array.isArray(first) ? 'array' : typeof first === 'undefined' ? newType.value : typeof first
  update([...props.modelValue, initial(type)])
}
</script>

<template>
  <div class="config-field" :class="{ 'field-container': kind === 'object' || kind === 'array', compact }">
    <div class="field-heading">
      <label :for="id">{{ label }}</label>
      <el-select :model-value="kind" @update:model-value="setType" size="small" class="type-select" :aria-label="`${props.path} 值类型`" :id="`${id}-type`">
        <el-option v-for="type in types" :key="type.value" :value="type.value" :label="type.label" />
      </el-select>
    </div>
    <p v-if="hint.description" class="field-hint">{{ hint.description }}</p>
    <div v-if="kind === 'object'" class="nested-fields">
      <div v-for="(value, key) in modelValue" :key="key" class="object-entry">
        <ConfigField :model-value="value" :path="`${path}.${key}`" :schema="schema" compact @update:model-value="setChild(key, $event)" />
        <el-button text type="danger" size="small" @click="removeChild(key)" :aria-label="`移除字段 ${path}.${key}`">移除字段</el-button>
      </div>
      <div class="add-field-row">
        <el-select v-model="keyName" filterable allow-create default-first-option placeholder="新增字段名称" :aria-label="`${path} 新增字段名称`">
          <el-option v-for="key in suggestions" :key="key" :value="key" :label="`${schema[`${path}.${key}`]?.label || key} · ${key}`" />
        </el-select>
        <el-select v-model="newType" class="type-select" aria-label="新字段类型"><el-option v-for="type in types" :key="type.value" :value="type.value" :label="type.label" /></el-select>
        <el-button @click="addKey">添加字段</el-button>
      </div>
      <p v-if="addError" role="alert" class="error-text">{{ addError }}</p>
    </div>
    <div v-else-if="kind === 'array'" class="array-fields">
      <div v-for="(value, index) in modelValue" :key="index" class="array-entry">
        <span class="row-number">{{ index + 1 }}</span>
        <ConfigField :model-value="value" :path="`${path}.${index}`" :schema="schema" compact @update:model-value="setChild(index, $event)" />
        <el-button text type="danger" @click="removeChild(index)" :aria-label="`移除 ${path} 第 ${index + 1} 项`">移除</el-button>
      </div>
      <p v-if="!modelValue.length" class="muted">空列表 · 尚无条目</p>
      <div class="inline-actions"><el-select v-if="!modelValue.length" v-model="newType" class="type-select" aria-label="新列表项类型"><el-option v-for="type in types" :key="type.value" :value="type.value" :label="type.label" /></el-select><el-button @click="addRow" :id="`${id}-add`">＋ 添加一项</el-button></div>
    </div>
    <el-switch v-else-if="kind === 'boolean'" :id="id" :model-value="modelValue" @update:model-value="update" active-text="开启" inactive-text="关闭" />
    <el-input-number v-else-if="kind === 'number'" :id="id" :model-value="modelValue" @update:model-value="update" :min="hint.min" :max="hint.max" :step="hint.step || (Math.abs(modelValue) > 0 && Math.abs(modelValue) < 1 ? 0.0001 : 1)" controls-position="right" :aria-label="label" />
    <div v-else-if="kind === 'null'" class="null-value">未设置 <code>null</code><span>使用上方类型菜单填写可选值</span></div>
    <el-select v-else-if="hint.options?.length" :id="id" :model-value="modelValue" @update:model-value="update" filterable allow-create default-first-option :aria-label="label">
      <el-option v-for="option in hint.options" :key="typeof option === 'object' ? option.value : option" :value="typeof option === 'object' ? option.value : option" :label="typeof option === 'object' ? option.label : String(option)" />
    </el-select>
    <el-input v-else :id="id" :model-value="modelValue" @update:model-value="update" :type="multiline ? 'textarea' : 'text'" :autosize="multiline ? { minRows: 2, maxRows: 8 } : undefined" :placeholder="hint.description || label" :aria-label="label" />
    <code class="field-path">{{ path }}</code>
  </div>
</template>
