// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Startup guard — fail fast if no API key
if (!NIM_API_KEY) {
  console.error('❌ NIM_API_KEY environment variable is not set!');
  process.exit(1);
}

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for models that support it
// Each model in THINKING_MODELS gets its own correct parameter format
const ENABLE_THINKING_MODE = false; // Set to true to enable per-model thinking params

// Per-model thinking parameters — each model has its own correct format
const THINKING_MODELS = {
  'z-ai/glm-5.1':                                  { enable_thinking: true },
  'qwen/qwen3-next-80b-a3b-thinking':               { chat_template_kwargs: { thinking: true } },
  'qwen/qwen3-coder-480b-a35b-instruct':            { chat_template_kwargs: { thinking: true } },
  'nvidia/llama-3.1-nemotron-ultra-253b-v1':        { chat_template_kwargs: { thinking: true } },
};

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':          'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo':    'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o':         'z-ai/glm-5.1',
  'claude-3-opus':  'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':     'qwen/qwen3-next-80b-a3b-thinking'
};

// Fallback model selection for unmapped/unknown model strings
function resolveFallbackModel(model) {
  const modelLower = model.toLowerCase();
  if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Resolve NIM model — use mapping, then fall back (no live probe)
    const nimModel = MODEL_MAPPING[model] || resolveFallbackModel(model);

    // Build per-model thinking params (spread directly into body, not extra_body)
    const thinkingParams = (ENABLE_THINKING_MODE && THINKING_MODELS[nimModel])
      ? THINKING_MODELS[nimModel]
      : {};

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 16000,
      stream: stream || false,
      ...thinkingParams  // Spread thinking params directly — no extra_body wrapper
    };

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              return;
            }

            const reasoning = delta.reasoning_content;
            const content = delta.content;

            if (SHOW_REASONING) {
              let combinedContent = '';

              if (reasoning && !reasoningStarted) {
                combinedContent = '<think>\n' + reasoning;
                reasoningStarted = true;
              } else if (reasoning) {
                combinedContent = reasoning;
              }

              if (content && reasoningStarted) {
                combinedContent += '\n</think>\n\n' + content;
                reasoningStarted = false;
              } else if (content) {
                combinedContent += content;
              }

              if (combinedContent) {
                data.choices[0].delta.content = combinedContent;
                delete data.choices[0].delta.reasoning_content;
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              }
              // If no combinedContent, skip the delta (pure buffering, nothing to send)
            } else {
              // Reasoning hidden — only forward deltas that have real content
              if (content) {
                delete data.choices[0].delta.reasoning_content;
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              }
              // Skip reasoning-only deltas entirely (don't send empty content)
            }
          } catch (e) {
            // Unparseable line — forward as-is
            res.write(line + '\n\n');
          }
        });
      });

      response.data.on('end', () => {
        if (buffer.trim()) res.write(buffer);
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    const status = error.response?.status || 500;
    const detail = error.response?.data?.detail || error.message || 'Internal server error';

    res.status(status).json({
      error: {
        message: detail,
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`✅ OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`   Health check: http://localhost:${PORT}/health`);
  console.log(`   Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
