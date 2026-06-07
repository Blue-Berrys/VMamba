import torch
# 设置允许的全局类
torch.serialization.add_safe_globals(['mmengine.logging.history_buffer.HistoryBuffer'])

# 然后运行原始的test脚本
import sys
sys.path.insert(0, 'tools')
from test import main
if __name__ == '__main__':
    main()
