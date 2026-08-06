// server.js - OpenAI to NVIDIA NIM API Proxy (GLM-5.1 / GLM-5.2 compatible)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const DEFAULT_MAX_TOKENS = parseInt(process.env.DEFAULT_MAX_TOKENS || '32768', 10);

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;
const CLEAR_THINKING_HISTORY = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':          'moonshotai/kimi-k2.6',
  'gpt-4-turbo':    'deepseek-ai/deepseek-v4-pro',
  'gpt-4o':         'z-ai/glm-5.2',
  'claude-3-opus':  'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':     'qwen/qwen3-next-80b-a3b-thinking'
};

const GLM_THINKING_MODELS = new Set([
  'z-ai/glm-5.1', 'z-ai/glm-5.2', 'z-ai/glm-5', 'z-ai/glm-5-turbo',
  'z-ai/glm-4.7', 'z-ai/glm-4.6', 'z-ai/glm-4.5',
]);

function buildThinkingParams() {
  return {
    chat_template_kwargs: {
      enable_thinking: ENABLE_THINKING_MODE,
      clear_thinking: CLEAR_THINKING_HISTORY
    }
  };
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    clear_thinking_history: CLEAR_THINKING_HISTORY
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stream } = req.body;
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      try {
        const probe = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
          {
            headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            validateStatus: s => s < 500
          }
        );
        if (probe.status >= 200 && probe.status < 300) nimModel = model;
      } catch (_) {}

      if (!nimModel) {
        const ml = model.toLowerCase();
        if (ml.includes('gpt-4') || ml.includes('claude-opus') || ml.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (ml.includes('claude') || ml.includes('gemini') || ml.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    const isGlm = GLM_THINKING_MODELS.has(nimModel);
    const defaultTemp = isGlm ? 1.0 : 0.6;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? defaultTemp,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      top_p,
      frequency_penalty,
      presence_penalty,
      stream: stream || false,
      ...buildThinkingParams()
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); return; }

            const reasoning = delta.reasoning_content;
            const content = delta.content;

            if (SHOW_REASONING) {
              let combined = '';
              if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
              else if (reasoning) { combined = reasoning; }
              if (content && reasoningStarted) { combined += '\n</think>\n\n' + content; reasoningStarted = false; }
              else if (content) { combined += content; }
              if (combined) { delta.content = combined; delete delta.reasoning_content; }
            } else {
              delta.content = content || '';
              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => { console.error('Stream error:', err); res.end(); });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
