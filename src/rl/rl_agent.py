import os
import sys

# Cachear el modelo para no recargarlo en cada turno durante el torneo
_model = None

def rl_agent(obs_dict, config=None):
    global _model
    
    # 1. Carga segura de dependencias (por si falta sb3_contrib en el entorno de evaluación)
    try:
        from src.rl.vectorizer import vectorize_state
        from sb3_contrib import MaskablePPO
    except ImportError:
        # Fallback a random si la librería no está instalada
        import random
        opts = obs_dict.get('select', {}).get('option', [])
        return [random.randint(0, max(0, len(opts)-1))] if opts else []

    # 2. Cargar el modelo SOLO en el primer turno de la partida
    if _model is None:
        # Busca el mejor checkpoint que el usuario haya puesto junto a este archivo
        model_name = "ppo_cabt_model_final"
        model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.zip")
        
        if not os.path.exists(model_path):
            model_path = f"{model_name}.zip" # Intenta en el directorio de ejecución local
            
        try:
            _model = MaskablePPO.load(model_path)
        except Exception as e:
            # Fallback en caso de que el usuario no haya subido el zip del modelo
            print(f"Agent Error: No se pudo cargar {model_path}. Usando Random. ({e})")
            import random
            opts = obs_dict.get('select', {}).get('option', [])
            return [random.randint(0, max(0, len(opts)-1))] if opts else []
            
    # 3. Vectorizar el estado
    # (El agente siempre recibe la perspectiva index 0 en Kaggle)
    vec = vectorize_state(obs_dict, my_index=0)
    
    # Extraer y remover el action_mask del vector (porque el entorno espera que se pasen por separado a MaskablePPO)
    action_mask = vec.pop("action_mask", None)
    
    if action_mask is None:
        # Fallback de emergencia
        import random
        opts = obs_dict.get('select', {}).get('option', [])
        return [random.randint(0, max(0, len(opts)-1))] if opts else []
    
    # 4. Predecir con el Cerebro
    try:
        min_count = obs_dict.get('select', {}).get('minCount', 1)
        ans = []
        current_mask = action_mask.copy()
        for _ in range(max(1, min_count)):
            if not any(current_mask):
                break
            action, _states = _model.predict(vec, action_masks=current_mask, deterministic=True)
            ans.append(int(action))
            current_mask[int(action)] = 0
            
        if not ans:
            ans = [0]
        return ans
    except Exception as e:
        print(f"Agent Error predict: {e}")
        import random
        opts = obs_dict.get('select', {}).get('option', [])
        return [random.randint(0, max(0, len(opts)-1))] if opts else []

# Kaggle Environments buscará una función llamada agent_fn o rl_agent
agent_fn = rl_agent
