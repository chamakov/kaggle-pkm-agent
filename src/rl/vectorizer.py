import numpy as np

MAX_HAND_SIZE = 30
MAX_BENCH_SIZE = 5
MAX_OPTIONS = 150
MAX_CARD_ID = 1300  # Added padding above 1267 just in case

def get_pokemon_stats(pokemon_obj):
    """Extrae el ID y estadísticas de un Pokémon de cg.api.Pokemon o diccionario"""
    if pokemon_obj is None:
        return 0, 0.0, 0.0
    
    # Manejar caso de que sea un objeto de la API (en el entorno)
    if hasattr(pokemon_obj, 'id'):
        cid = pokemon_obj.id + 1  # 1-indexed, 0 is empty
        hp = float(pokemon_obj.hp)
        energies = float(len(getattr(pokemon_obj, 'energyCards', [])))
        return cid, hp, energies
    
    # Manejar caso de diccionario en crudo
    if isinstance(pokemon_obj, dict):
        cid = pokemon_obj.get('id', -1) + 1
        hp = float(pokemon_obj.get('hp', 0))
        energies = float(len(pokemon_obj.get('energyCards', [])))
        return cid, hp, energies
        
    return 0, 0.0, 0.0

def vectorize_state(obs_dict, my_index):
    """
    Convierte el obs_dict de Kaggle en un diccionario compatible con MultiInputPolicy:
    - card_ids: np.array de enteros
    - scalars: np.array de floats
    - action_mask: np.array booleano
    """
    import sys
    import os
    try:
        from cg.api import to_observation_class
    except ImportError:
        sys.path.append(os.path.join(os.getcwd(), "remotethings", "cg_custom"))
        from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    
    my_state = obs.current.players[my_index]
    opp_state = obs.current.players[1 - my_index]
    
    # --- 1. Extraer IDs de Cartas ---
    card_ids = []
    scalars = []
    
    # My Active
    active_id, active_hp, active_energy = get_pokemon_stats(my_state.active[0] if my_state.active else None)
    card_ids.append(active_id)
    scalars.extend([active_hp, active_energy])
    
    # My Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en = get_pokemon_stats(my_state.bench[i] if i < len(my_state.bench) else None)
        card_ids.append(b_id)
        scalars.extend([b_hp, b_en])
        
    # My Hand
    for i in range(MAX_HAND_SIZE):
        if i < len(my_state.hand):
            card = my_state.hand[i]
            cid = (card.id + 1) if hasattr(card, 'id') else (card.get('id', -1) + 1 if isinstance(card, dict) else 0)
            card_ids.append(cid)
        else:
            card_ids.append(0)
            
    # Opp Active
    opp_act_id, opp_act_hp, opp_act_en = get_pokemon_stats(opp_state.active[0] if opp_state.active else None)
    card_ids.append(opp_act_id)
    scalars.extend([opp_act_hp, opp_act_en])
    
    # Opp Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en = get_pokemon_stats(opp_state.bench[i] if i < len(opp_state.bench) else None)
        card_ids.append(b_id)
        scalars.extend([b_hp, b_en])
        
    # Variables globales
    scalars.append(float(len(my_state.prize)))
    scalars.append(float(len(opp_state.prize)))
    scalars.append(float(my_state.deckCount))
    scalars.append(float(opp_state.handCount))
    
    # --- 2. Action Mask ---
    action_mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
    options = obs_dict.get('select', {}).get('option', [])
    num_options = len(options)
    
    if num_options > 0:
        # Habilitar los índices válidos
        for i in range(min(num_options, MAX_OPTIONS)):
            action_mask[i] = 1
    else:
        # Si no hay opciones, permitimos la acción 0 para pasar
        action_mask[0] = 1

    return {
        "card_ids": np.array(card_ids, dtype=np.int32),
        "scalars": np.array(scalars, dtype=np.float32),
        "action_mask": action_mask
    }
