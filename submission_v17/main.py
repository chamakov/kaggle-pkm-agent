import os
import sys

# Get the path to the current file (handling Kaggle's exec environment where __file__ is undefined)
if "__file__" in globals():
    agent_dir = os.path.dirname(os.path.abspath(__file__))
else:
    agent_dir = "/kaggle_simulations/agent/"
sys.path.append(agent_dir)

from vectorizer import vectorize_state, map_semantic_to_env_action

try:
    import torch
    import torch.nn as nn
    from pytorch_inference import load_agent
    pytorch_available = True
except ImportError as e:
    print(f"Warning: PyTorch o módulos de inferencia no disponibles: {e}")
    pytorch_available = False

# Variable global para cargar el modelo una sola vez por sesión
global_model = None

slowking_deck = [
    162, 162, 162, 162, 163, 163, 163, 144, 144, 756, 756, 115, 184, 140, 224, 
    1071, 183, 880, 272, 1227, 1227, 1227, 1227, 1188, 1188, 1188, 1210, 1231, 
    1184, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1097, 1097, 1097, 1146, 
    1146, 1146, 1092, 1123, 1156, 1248, 1248, 1248, 1248, 19, 19, 19, 19, 5, 5, 5, 9, 9, 9
]

def agent(obs_dict, config=None):
    global global_model
    
    # 1. Fase de Selección de Mazo (Step 0)
    if obs_dict.get('select') is None:
        return slowking_deck
        
    # 2. Si no hay opciones, debemos pasar []
    opts = obs_dict.get('select', {}).get('option', [])
    if len(opts) == 0:
        return []
        
    try:
        # Cargar el modelo si aún no está cargado
        if global_model is None and pytorch_available:
            model_path = os.path.join(agent_dir, "policy.pth")
            global_model = load_agent(model_path)
            
        my_index = obs_dict.get("current", {}).get("yourIndex", 0)
        
        # 3. Vectorizar el estado (esto también calcula la máscara de acciones básicas)
        vec = vectorize_state(obs_dict, my_index=my_index)
        action_mask = vec.pop("action_mask")
        
        # 4. Predicción del modelo con soporte iterativo para Multi-Select
        min_count = obs_dict.get('select', {}).get('minCount', 1)
        ans = []
        
        if global_model is not None:
            current_mask = action_mask.copy()
            for _ in range(max(1, min_count)):
                if not any(current_mask):
                    break # No more valid options
                
                # Convertir a tensores PyTorch
                card_ids_t = torch.tensor(vec['card_ids']).unsqueeze(0).long()
                scalars_t = torch.tensor(vec['scalars']).unsqueeze(0).float()
                
                # Inferencia sin gradientes
                with torch.no_grad():
                    logits = global_model(card_ids_t, scalars_t).squeeze(0) # [700]
                
                # Aplicar máscara (poner a -inf los inválidos)
                mask_t = torch.tensor(current_mask, dtype=torch.bool)
                logits[~mask_t] = -float('inf')
                
                # Obtener la mejor acción (argmax)
                sem_int = int(torch.argmax(logits).item())
                
                env_action = map_semantic_to_env_action(sem_int, obs_dict)
                ans.append(env_action)
                current_mask[sem_int] = 0
        else:
            import random
            valid_indices = [i for i, m in enumerate(action_mask) if m == 1]
            for _ in range(max(1, min_count)):
                if not valid_indices:
                    break
                idx = random.randint(0, len(valid_indices) - 1)
                sem_int = valid_indices.pop(idx)
                env_action = map_semantic_to_env_action(sem_int, obs_dict)
                ans.append(env_action)
                
        if not ans:
            ans = [0]
            
        return ans
    except Exception as e:
        import traceback
        traceback.print_exc()
        import random
        return [random.randint(0, max(0, len(opts)-1))]
