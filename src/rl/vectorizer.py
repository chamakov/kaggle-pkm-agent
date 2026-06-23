import numpy as np

MAX_HAND_SIZE = 30
MAX_BENCH_SIZE = 5
MAX_OPTIONS = 150
MAX_CARD_ID = 1300  # Added padding above 1267 just in case

def get_pokemon_stats(pokemon_obj):
    """Extrae el ID y estadísticas de un Pokémon de cg.api.Pokemon o diccionario"""
    # Retorna: cid, hp, energies_count, retreat_cost, e1, e2, e3, e4
    if pokemon_obj is None:
        return 0, 0.0, 0.0, 0.0, 0, 0, 0, 0
    
    if hasattr(pokemon_obj, 'id'):
        cid = pokemon_obj.id + 1
        hp = float(pokemon_obj.hp)
        en_cards = getattr(pokemon_obj, 'energyCards', [])
        energies_count = float(len(en_cards))
        retreat_cost = float(getattr(pokemon_obj, 'retreatCost', 0))
        
        e_ids = [0, 0, 0, 0]
        for idx, e in enumerate(en_cards[:4]):
            e_ids[idx] = e.id + 1 if hasattr(e, 'id') else 0
            
        return cid, hp, energies_count, retreat_cost, e_ids[0], e_ids[1], e_ids[2], e_ids[3]
    
    if isinstance(pokemon_obj, dict):
        cid = pokemon_obj.get('id', -1) + 1
        hp = float(pokemon_obj.get('hp', 0))
        en_cards = pokemon_obj.get('energyCards', [])
        energies_count = float(len(en_cards))
        retreat_cost = float(pokemon_obj.get('retreatCost', 0))
        
        e_ids = [0, 0, 0, 0]
        for idx, e in enumerate(en_cards[:4]):
            if isinstance(e, dict):
                e_ids[idx] = e.get('id', -1) + 1
            else:
                e_ids[idx] = e.id + 1 if hasattr(e, 'id') else 0
                
        return cid, hp, energies_count, retreat_cost, e_ids[0], e_ids[1], e_ids[2], e_ids[3]
        
    return 0, 0.0, 0.0, 0.0, 0, 0, 0, 0

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
    active_id, active_hp, active_energy, active_rc, ae1, ae2, ae3, ae4 = get_pokemon_stats(my_state.active[0] if my_state.active else None)
    card_ids.extend([active_id, ae1, ae2, ae3, ae4])
    scalars.extend([active_hp, active_energy, active_rc])
    
    # My Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en, b_rc, be1, be2, be3, be4 = get_pokemon_stats(my_state.bench[i] if i < len(my_state.bench) else None)
        card_ids.extend([b_id, be1, be2, be3, be4])
        scalars.extend([b_hp, b_en, b_rc])
        
    # My Hand
    for i in range(MAX_HAND_SIZE):
        if i < len(my_state.hand):
            card = my_state.hand[i]
            cid = (card.id + 1) if hasattr(card, 'id') else (card.get('id', -1) + 1 if isinstance(card, dict) else 0)
            card_ids.append(cid)
        else:
            card_ids.append(0)
            
    # Opp Active
    opp_act_id, opp_act_hp, opp_act_en, opp_act_rc, oae1, oae2, oae3, oae4 = get_pokemon_stats(opp_state.active[0] if opp_state.active else None)
    card_ids.extend([opp_act_id, oae1, oae2, oae3, oae4])
    scalars.extend([opp_act_hp, opp_act_en, opp_act_rc])
    
    # Opp Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en, b_rc, be1, be2, be3, be4 = get_pokemon_stats(opp_state.bench[i] if i < len(opp_state.bench) else None)
        card_ids.extend([b_id, be1, be2, be3, be4])
        scalars.extend([b_hp, b_en, b_rc])
        
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
        # Load ComboSequencer for heuristic masking
        try:
            from src.agent.synergy_analyzer import ComboSequencer
            sequencer = ComboSequencer()
            
            # Create a mock state to pass to sequencer
            class MockState:
                def __init__(self, obs):
                    self.observation = obs
            mock_state = MockState(obs_dict)
            
            # Score each action and mask bad ones
            scores = []
            for i in range(num_options):
                score = sequencer.evaluate_synergy(i, mock_state)
                scores.append(score)
                
            max_score = max(scores) if scores else 0
            
            # Habilitar índices válidos que pasen el filtro heurístico
            for i in range(min(num_options, MAX_OPTIONS)):
                # Permitir siempre la mejor acción (para no quedar bloqueados) o aquellas que no sean terriblemente malas (-5.0)
                if scores[i] >= -5.0 or scores[i] == max_score:
                    action_mask[i] = 1
        except ImportError:
            # Fallback si no está disponible la heurística
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
