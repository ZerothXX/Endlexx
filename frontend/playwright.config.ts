import { defineConfig } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('..', import.meta.url))
const data = path.join(root, 'data', 'browser-tests')
const external = process.env.ENDLEXX_TEST_BASE_URL
const baseURL = external || 'http://127.0.0.1:8877'
const python = process.env.ENDLEXX_TEST_PYTHON || 'python'

export default defineConfig({
  testDir: './tests',
  use: { baseURL, headless: true },
  timeout: 30000,
  workers: 1,
  reporter: 'list',
  webServer: external ? undefined : {
    command: `"${python}" -m uvicorn endlexx.api:app --host 127.0.0.1 --port 8877 --no-access-log`,
    cwd: root,
    url: baseURL + '/api/health',
    reuseExistingServer: false,
    timeout: 30000,
    env: {
      ENDLEXX_DATABASE_URL: 'sqlite:///' + path.join(data, 'test.db').replaceAll('\\', '/'),
      ENDLEXX_DATA_DIR: data,
      ENDLEXX_TASK_BACKEND: 'local',
      ENDLEXX_REQUIRE_ACCESS_TOKEN: 'false',
      ENDLEXX_ENDLESS_ROOT: path.join(data, 'no-research-models'),
      ENDLEXX_MODEL_RUNTIME_URL: 'http://127.0.0.1:1',
      ENDLEXX_API_KEY: '',
    },
  },
})
