import { mkdir, readdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { build } from 'vite'

const frontend = fileURLToPath(new URL('../', import.meta.url))
const releases = path.resolve(frontend, '../static')
await mkdir(releases, { recursive: true })
const prior = (await readdir(releases)).filter(name => /^build-\d+$/.test(name)).map(name => Number(name.slice(6)))
let timestamp = Math.max(Date.now(), ...prior.map(value => value + 1))
let output
for (;;) {
  output = path.join(releases, `build-${timestamp}`)
  try {
    await mkdir(output)
    break
  } catch (error) {
    if (error.code !== 'EEXIST') throw error
    timestamp += 1
  }
}
await build({ root: frontend, build: { outDir: output, emptyOutDir: false } })
console.log(`Preserved release: ${output}`)
