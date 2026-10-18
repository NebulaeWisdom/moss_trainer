<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { AriaComponent, DataZoomComponent, GridComponent, LegendComponent, ToolboxComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
echarts.use([LineChart, AriaComponent, DataZoomComponent, GridComponent, LegendComponent, ToolboxComponent, TooltipComponent, CanvasRenderer])
const props = defineProps({ rows: { type: Array, default: () => [] }, fields: { type: Array, required: true }, title: String, theme: String })
const element = ref(null)
let chart
let observer
const usable = computed(() => props.fields.filter(field => props.rows.some(row => typeof row[field.key] === 'number' && Number.isFinite(row[field.key]))))
async function render() {
  await nextTick()
  if (!element.value || !element.value.clientWidth) return
  if (!chart) chart = echarts.init(element.value)
  const style = getComputedStyle(document.documentElement)
  const text = style.getPropertyValue('--text-secondary').trim()
  const border = style.getPropertyValue('--border').trim()
  chart.setOption({
    animation: !matchMedia('(prefers-reduced-motion: reduce)').matches,
    animationDuration: 250,
    aria: { enabled: true, description: `${props.title}，${props.rows.length} 条真实训练记录。完整数据可在指标表中查看。` },
    textStyle: { color: text, fontFamily: 'Consolas, monospace', fontSize: 11 },
    grid: { left: 62, right: 26, top: 48, bottom: 66 },
    legend: { top: 5, left: 10, textStyle: { color: text } },
    tooltip: { trigger: 'axis', backgroundColor: style.getPropertyValue('--surface').trim(), borderColor: border, textStyle: { color: style.getPropertyValue('--text-primary').trim() }, confine: true },
    toolbox: { right: 8, feature: { saveAsImage: { title: '保存图表', name: props.title }, dataZoom: { title: { zoom: '区域缩放', back: '还原缩放' } } }, iconStyle: { borderColor: text } },
    xAxis: { type: 'value', name: 'Step', nameLocation: 'end', splitLine: { show: false }, axisLine: { lineStyle: { color: border } }, axisLabel: { color: text } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: text }, splitLine: { lineStyle: { color: border, type: 'dashed' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 12, borderColor: border, textStyle: { color: text } }],
    series: usable.value.map(field => ({ name: field.label, type: 'line', showSymbol: false, connectNulls: false, lineStyle: { width: 2, color: field.color }, itemStyle: { color: field.color }, data: props.rows.map(row => [row.step, typeof row[field.key] === 'number' ? row[field.key] : null]) })),
  }, { replaceMerge: ['series'] })
  chart.resize()
}
onMounted(() => { observer = new ResizeObserver(() => chart ? chart.resize() : render()); if (element.value) observer.observe(element.value); render() })
watch(() => [props.rows, props.fields, props.theme], render)
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<template>
  <el-card class="chart-card" shadow="never">
    <template #header><h3>{{ title }}</h3><span class="muted">{{ usable.length ? `${rows.length} 条真实记录` : '等待训练指标' }}</span></template>
    <div class="chart-wrap">
      <div ref="element" class="metric-chart" :class="{ 'chart-hidden': !usable.length }" role="img" :aria-label="`${title} 训练曲线；可在指标表读取全部数值`" />
      <el-empty v-if="!usable.length" class="chart-empty" :image-size="54" description="暂无记录 · 训练产生指标后显示" />
    </div>
  </el-card>
</template>
