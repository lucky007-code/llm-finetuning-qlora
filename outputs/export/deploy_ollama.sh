#!/bin/bash
# Production Deployment Automation Script for Ollama
echo "🚀 Creating Ollama custom model: incident-triager..."
ollama create incident-triager -f ./outputs/export\Modelfile

echo "✅ Deployment successful! Run model interactively with:"
echo "ollama run incident-triager"
echo ""
echo "Or query via REST API:"
echo 'curl http://localhost:11434/api/generate -d \'
echo '  \'{"model": "incident-triager", "prompt": "[ERROR] Redis OOMKilled", "stream": false}\' '
