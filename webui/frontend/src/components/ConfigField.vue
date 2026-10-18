<script setup>
import { computed, ref } from 'vue'

const props = defineProps({ modelValue: { required: true }, path: { type: String, required: true }, schema: { type: Object, default: () => ({}) }, fieldSchema: { type: Object, default: null }, conversionCache: { type: Map, default: () => new Map() }, compact: Boolean })
const emit = defineEmits(['update:modelValue'])
const hint = computed(() => Object.hasOwn(props.schema, props.path) ? props.schema[props.path] : props.fieldSchema || {})
const valueType = value => value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value
const kind = computed(() => valueType(props.modelValue))
const legalTypes = computed(() => hint.value.types || [kind.value])
const typeLabels = { string: '文本', number: '数字', boolean: '开关', null: '未设置 null', array: '列表', object: '对象' }
const typeLabel = type => type === 'number' && hint.value.integer ? '整数' : typeLabels[type]
const id = computed(() => `cfg-${props.path.replace(/[^a-zA-Z0-9_-]/g, '-')}`)
const label = computed(() => hint.value.label || props.path.split('.').at(-1))
const keyName = ref('')
const addError = ref('')
const newHint = computed(() => {
  const path = `${props.path}.${keyName.value.trim()}`
  return Object.hasOwn(props.schema, path) ? props.schema[path] : null
})
const suggestions = computed(() => Object.keys(props.schema).filter(path => path.startsWith(`${props.path}.`) && !path.slice(props.path.length + 1).includes('.')).map(path => path.slice(props.path.length + 1)).filter(key => !Object.hasOwn(props.modelValue || {}, key)))
const multiline = computed(() => /prompt(?:\.\d+)?$|caption$/.test(props.path) || (typeof props.modelValue === 'string' && props.modelValue.includes('\n')))
const textList = computed(() => legalTypes.value.includes('string') && legalTypes.value.includes('array') && hint.value.items?.types?.length === 1 && hint.value.items.types[0] === 'string')
function initial(rule, type = rule.types[0]) {
  if (Object.hasOwn(rule, 'default') && valueType(rule.default) === type) return JSON.parse(JSON.stringify(rule.default))
  const option = rule.options?.find(value => valueType(value) === type)
  if (option !== undefined) return option
  return ({ string: '', number: rule.min ?? (rule.integer ? 1 : 0), boolean: false, null: null, array: [], object: {} })[type]
}
function update(value) { emit('update:modelValue', value) }
function updateNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || hint.value.integer && !Number.isInteger(value)) return
  update(value)
}
function setType(type) {
  if (type === kind.value || !legalTypes.value.includes(type)) return
  if (textList.value && type === 'string' && kind.value === 'array') {
    props.conversionCache.set(props.path, props.modelValue.slice())
    update(props.modelValue[0] ?? '')
  } else if (textList.value && type === 'array' && kind.value === 'string') {
    const remembered = props.conversionCache.get(props.path)
    update([props.modelValue, ...(remembered?.slice(1) || [])])
  } else update(initial(hint.value, type))
}
function forget(path) {
  for (const key of props.conversionCache.keys()) if (key === path || key.startsWith(`${path}.`)) props.conversionCache.delete(key)
}
function setChild(key, value) {
  const next = Array.isArray(props.modelValue) ? props.modelValue.slice() : { ...props.modelValue }
  Object.defineProperty(next, key, { value, enumerable: true, writable: true, configurable: true })
  update(next)
}
function removeChild(key) {
  const array = Array.isArray(props.modelValue)
  const next = array ? props.modelValue.filter((_, index) => index !== key) : { ...props.modelValue }
  if (!array) delete next[key]
  forget(array ? props.path : `${props.path}.${key}`)
  update(next)
}
function addKey() {
  const key = keyName.value.trim()
  if (!newHint.value) { addError.value = '请选择已声明类型的字段。'; return }
  if (Object.hasOwn(props.modelValue, key)) { addError.value = '字段已存在，不会覆盖。'; return }
  const rule = newHint.value
  const value = rule.defaultFrom && Object.hasOwn(props.modelValue, rule.defaultFrom) ? props.modelValue[rule.defaultFrom] : Object.hasOwn(rule, 'default') ? JSON.parse(JSON.stringify(rule.default)) : initial(rule)
  forget(`${props.path}.${key}`)
  setChild(key, value)
  keyName.value = ''
  addError.value = ''
}
function addRow() {
  if (!hint.value.items || hint.value.maxItems !== undefined && props.modelValue.length >= hint.value.maxItems) return
  const value = initial(hint.value.items)
  update([...props.modelValue, value])
  addError.value = ''
}
</script>

<template>
  <div class="config-field" :class="{ 'field-container': kind === 'object' || kind === 'array', compact }">
    <div class="field-heading">
      <label :for="id">{{ label }}</label>
      <el-select v-if="legalTypes.length > 1" :model-value="kind" @update:model-value="setType" size="small" class="type-select" :aria-label="`${props.path} 值类型`" :id="`${id}-type`">
        <el-option v-for="type in legalTypes" :key="type" :value="type" :label="typeLabel(type)" />
      </el-select>
      <span v-else class="field-type" :id="`${id}-type`" :aria-label="`${props.path} 固定类型`">{{ typeLabel(legalTypes[0]) }}</span>
    </div>
    <p v-if="hint.description" class="field-hint">{{ hint.description }}</p>
    <p v-if="!hint.types" class="field-hint">扩展字段，类型沿用已加载值。</p>
    <p v-else-if="!legalTypes.includes(kind)" role="alert" class="error-text">当前值类型不合法；应为 {{ legalTypes.map(typeLabel).join(' / ') }}。</p>
    <div v-if="kind === 'object'" class="nested-fields">
      <div v-for="(value, key) in modelValue" :key="key" class="object-entry">
        <ConfigField :model-value="value" :path="`${path}.${key}`" :schema="schema" :conversion-cache="conversionCache" compact @update:model-value="setChild(key, $event)" />
        <el-button text type="danger" size="small" @click="removeChild(key)" :aria-label="`移除字段 ${path}.${key}`">移除字段</el-button>
      </div>
      <div v-if="suggestions.length" class="add-field-row">
        <el-select v-model="keyName" filterable default-first-option placeholder="选择已知字段" :aria-label="`${path} 新增字段名称`">
          <el-option v-for="key in suggestions" :key="key" :value="key" :label="`${schema[`${path}.${key}`]?.label || key} · ${key}`" />
        </el-select>
        <span v-if="newHint" class="field-type">{{ newHint.types.map(type => type === 'number' && newHint.integer ? '整数' : typeLabels[type]).join(' / ') }} · 使用字段默认值</span>
        <el-button @click="addKey" :disabled="!newHint">添加字段</el-button>
      </div>
      <p v-if="addError" role="alert" class="error-text">{{ addError }}</p>
    </div>
    <div v-else-if="kind === 'array'" class="array-fields">
      <div v-for="(value, index) in modelValue" :key="index" class="array-entry">
        <span class="row-number">{{ index + 1 }}</span>
        <ConfigField :model-value="value" :path="`${path}.${index}`" :schema="schema" :field-schema="hint.items" :conversion-cache="conversionCache" compact @update:model-value="setChild(index, $event)" />
        <el-button text type="danger" @click="removeChild(index)" :disabled="modelValue.length <= (hint.minItems || 0)" :aria-label="`移除 ${path} 第 ${index + 1} 项`">移除</el-button>
      </div>
      <p v-if="!modelValue.length" class="muted">空列表 · 尚无条目</p>
      <div class="inline-actions"><el-button @click="addRow" :id="`${id}-add`" :disabled="!hint.items || hint.maxItems !== undefined && modelValue.length >= hint.maxItems">＋ 添加一项</el-button></div>
      <p v-if="!hint.items" class="field-hint">扩展列表未声明条目类型，仅编辑已加载条目。</p>
    </div>
    <el-switch v-else-if="kind === 'boolean'" :id="id" :model-value="modelValue" @update:model-value="update" active-text="开启" inactive-text="关闭" />
    <el-input-number v-else-if="kind === 'number'" :id="id" :model-value="modelValue" @update:model-value="updateNumber" :min="hint.min" :max="hint.max" :precision="hint.integer ? 0 : undefined" :step-strictly="hint.integer || false" :step="hint.integer ? 1 : hint.step || (Math.abs(modelValue) > 0 && Math.abs(modelValue) < 1 ? 0.0001 : 1)" controls-position="right" :aria-label="label" />
    <div v-else-if="kind === 'null'" class="null-value">未设置 <code>null</code><span v-if="legalTypes.length > 1">使用上方类型菜单填写可选值</span></div>
    <el-select v-else-if="hint.options?.length" :id="id" :model-value="modelValue" @update:model-value="update" filterable allow-create default-first-option :aria-label="label">
      <el-option v-for="option in hint.options" :key="typeof option === 'object' ? option.value : option" :value="typeof option === 'object' ? option.value : option" :label="typeof option === 'object' ? option.label : String(option)" />
    </el-select>
    <el-input v-else :id="id" :model-value="modelValue" @update:model-value="update" :type="multiline ? 'textarea' : 'text'" :autosize="multiline ? { minRows: 2, maxRows: 8 } : undefined" :placeholder="hint.description || label" :aria-label="label" />
    <code class="field-path">{{ path }}</code>
  </div>
</template>
