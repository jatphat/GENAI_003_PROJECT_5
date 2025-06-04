import sys
import types
import torch

# Prevent Streamlit from introspecting torch.classes
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes