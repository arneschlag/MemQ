#!/bin/bash
# LlamaFactory setup inside the llamafactory container for the joint v14 run.
# Same pins as the working v9 environment (peft 0.18.1); additionally registers
# the joint WebQSP+CWQ+GrailQA dataset as `memq_v14`.
set -e
cd /root
[ -d LlamaFactory ] || git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
sed -i 's/peft>=0.18.0,<=0.18.1/peft>=0.14.0,<=0.18.1/' pyproject.toml
pip install peft==0.18.1 2>&1 | tail -1
pip install -e . 2>&1 | tail -3
pip install -r requirements/metrics.txt 2>&1 | tail -1
echo "=== Versions ==="
python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
python3 -c "import peft; print('peft', peft.__version__)"

python3 -c "
import json
p = 'data/dataset_info.json'
with open(p) as f:
    info = json.load(f)
# v9 corpus stays registered so the old run remains reproducible.
info['memq'] = {'file_name': '/root/data/memq_finetune_data.json', 'split': 'train'}
info['memq_v14'] = {'file_name': '/root/data/memq_finetune_data_v14.json', 'split': 'train'}
with open(p, 'w') as f:
    json.dump(info, f, indent=2)
print('datasets registered:', [k for k in info if k.startswith('memq')])
"
echo "=== SETUP DONE ==="
