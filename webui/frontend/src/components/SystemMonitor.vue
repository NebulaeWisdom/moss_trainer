<script setup>
import { computed } from 'vue'
import { bytes, numeric, dateTime } from '../api'
const props = defineProps({ system: Object })
const cards = computed(() => {
  const sample = props.system
  const gpu = sample?.gpus?.[0]
  return [
    { label: 'CPU', subtitle: '处理器', value: sample?.cpu?.percent, detail: sample?.cpu?.logical_cores != null ? `${sample.cpu.logical_cores} logical cores` : '等待设备数据', color: 'sapphire' },
    { label: 'RAM', subtitle: '系统内存', value: sample?.memory?.percent, detail: `${bytes(sample?.memory?.used_bytes)} / ${bytes(sample?.memory?.total_bytes)}`, color: 'sapphire' },
    { label: 'GPU', subtitle: gpu?.name || '图形处理器', value: gpu?.utilization_percent, detail: `${gpu?.temperature_c == null ? '温度 —' : `${numeric(gpu.temperature_c)} °C`} · ${gpu?.power_w == null ? '功耗 —' : `${numeric(gpu.power_w, 1)} W`}`, color: 'aquamarine' },
    { label: 'VRAM', subtitle: '显存', value: gpu?.memory_percent, detail: `${bytes(gpu?.memory_used_bytes)} / ${bytes(gpu?.memory_total_bytes)}`, color: gpu?.memory_percent > 85 ? 'danger' : 'taaffeite' },
  ]
})
const percentage = value => Math.max(0, Math.min(100, value || 0))
</script>

<template>
  <div class="telemetry-grid">
    <el-card v-for="card in cards" :key="card.label" class="telemetry-card" :class="card.color" shadow="never">
      <div class="telemetry-title"><span>{{ card.label }}</span><span class="telemetry-subtitle">{{ card.subtitle }}</span></div>
      <div class="telemetry-number">{{ numeric(card.value, 1) }}<small v-if="card.value != null">%</small><el-tag v-if="card.color === 'danger'" type="danger" size="small">High 高</el-tag></div>
      <el-progress v-if="card.value != null" :percentage="percentage(card.value)" :show-text="false" :stroke-width="5" :color="`var(--${card.color})`" />
      <div v-else class="unavailable-track" />
      <p class="telemetry-detail mono">{{ card.detail }}</p>
    </el-card>
  </div>
  <div class="telemetry-footer"><span>磁盘 {{ bytes(system?.disk?.used_bytes) }} / {{ bytes(system?.disk?.total_bytes) }} <b v-if="system?.disk?.percent != null">· {{ numeric(system.disk.percent, 1) }}%</b></span><span>采集于 {{ dateTime(system?.timestamp) }} · 每 2 秒更新</span></div>
  <el-alert v-if="system?.gpu_error" :title="`GPU 监测不可用：${system.gpu_error}`" type="warning" :closable="false" show-icon />
  <el-collapse v-if="system?.gpus?.length > 1" class="extra-gpus"><el-collapse-item title="全部 GPU 设备" name="gpus"><el-table :data="system.gpus"><el-table-column prop="index" label="#" width="50" /><el-table-column prop="name" label="GPU" /><el-table-column label="使用率"><template #default="{ row }">{{ numeric(row.utilization_percent) }}%</template></el-table-column><el-table-column label="显存"><template #default="{ row }">{{ bytes(row.memory_used_bytes) }} / {{ bytes(row.memory_total_bytes) }}</template></el-table-column><el-table-column label="温度 / 功耗"><template #default="{ row }">{{ numeric(row.temperature_c) }} °C / {{ numeric(row.power_w) }} W</template></el-table-column></el-table></el-collapse-item></el-collapse>
</template>
