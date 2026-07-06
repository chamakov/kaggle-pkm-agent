import numpy as np

MAX_HAND_SIZE = 30
MAX_BENCH_SIZE = 5
MAX_OPTIONS = 700
MAX_CARD_ID = 1300  # Added padding above 1267 just in case

def map_env_to_semantic_action(opt_idx: int, obs_dict: dict) -> int:
    my_index = obs_dict.get('current', {}).get('yourIndex', 0)
    options = obs_dict.get('select', {}).get('option', [])
    if opt_idx < 0 or opt_idx >= len(options):
        return 650
        
    opt = options[opt_idx]
    opt_type = opt.get('type', -1)
    
    if opt_type == 14: # END
        return 0
    elif opt_type == 12: # RETREAT
        return 1
    elif opt_type == 13: # ATTACK
        attack_options = [o for o in options if o.get('type') == 13]
        attack_options.sort(key=lambda x: x.get('attackId', 0) if isinstance(x, dict) else getattr(x, 'attackId', 0))
        for i, ao in enumerate(attack_options):
            if ao == opt:
                return 2 + min(i, 3)
        return 2
    elif opt_type == 7: # PLAY
        idx = opt.get('index', 0)
        return 6 + min(idx, 29)
    elif opt_type == 10: # ABILITY
        area = opt.get('area', -1)
        idx = opt.get('index', 0)
        if area == 2: # HAND
            return 36 + min(idx, 29)
        elif area == 4: # ACTIVE
            return 66
        elif area == 5: # BENCH
            return 67 + min(idx, 4)
    elif opt_type in (8, 9): # ATTACH (8) or EVOLVE (9)
        src_idx = opt.get('index', 0)
        in_play_area = opt.get('inPlayArea', -1)
        in_play_idx = opt.get('inPlayIndex', 0)
        target = 0
        if in_play_area == 5: # BENCH
            target = 1 + min(in_play_idx, 4)
        offset = 72 if opt_type == 8 else 252
        return offset + min(src_idx, 29) * 6 + target
    elif opt_type == 3: # CARD
        area = opt.get('area', -1)
        idx = opt.get('index', 0)
        player = opt.get('playerIndex', my_index)
        if area == 1: # DECK
            return 432 + min(idx, 59)
        elif area == 2: # HAND
            return 492 + min(idx, 29)
        elif area == 3: # DISCARD
            return 522 + min(idx, 59)
        elif area == 6: # PRIZE
            return 582 + min(idx, 5)
        elif area == 4: # ACTIVE
            return 588 if player == my_index else 589
        elif area == 5: # BENCH
            return 590 + min(idx, 4) if player == my_index else 595 + min(idx, 4)
        elif area == 12: # LOOKING
            return 600 + min(idx, 9)
    elif opt_type == 0: # NUMBER
        num = opt.get('number', 0)
        return 610 + min(num, 19)
    elif opt_type == 1: # YES
        return 630
    elif opt_type == 2: # NO
        return 631
    elif opt_type == 11: # DISCARD
        area = opt.get('area', -1)
        idx = opt.get('index', 0)
        if area == 4:
            return 632
        elif area == 5:
            return 633 + min(idx, 4)
            
    return 650 + min(opt_idx, 49)

def map_semantic_to_env_action(semantic_action: int, obs_dict: dict) -> int:
    options = obs_dict.get('select', {}).get('option', [])
    if not options:
        return 0
    for i in range(len(options)):
        if map_env_to_semantic_action(i, obs_dict) == semantic_action:
            return i
    return 0 # Fallback if action is invalid


def get_pokemon_stats(pokemon_obj):
    """Extrae el ID y estadísticas de un Pokémon de cg.api.Pokemon o diccionario"""
    # Retorna: cid, hp, energies_count, retreat_cost, dmg, e1, e2, e3, e4, maxHp, tools_count, appearThisTurn
    if pokemon_obj is None:
        return 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0
    
    if hasattr(pokemon_obj, 'id'):
        cid = pokemon_obj.id + 1
        hp = float(pokemon_obj.hp)
        maxHp = float(getattr(pokemon_obj, 'maxHp', 0))
        en_cards = getattr(pokemon_obj, 'energyCards', [])
        energies_count = float(len(en_cards))
        retreat_cost = float(getattr(pokemon_obj, 'retreatCost', 0))
        tools_count = float(len(getattr(pokemon_obj, 'tools', [])))
        appearThisTurn = 1.0 if getattr(pokemon_obj, 'appearThisTurn', False) else 0.0
        
        # Extract damage of first attack if available
        dmg = 0.0
        attacks = getattr(pokemon_obj, 'attacks', [])
        if attacks and len(attacks) > 0:
            first_attack = attacks[0]
            dmg = float(getattr(first_attack, 'damage', 0))
            
        e_ids = [0, 0, 0, 0]
        for idx, e in enumerate(en_cards[:4]):
            e_ids[idx] = e.id + 1 if hasattr(e, 'id') else 0
            
        return cid, hp, energies_count, retreat_cost, dmg, e_ids[0], e_ids[1], e_ids[2], e_ids[3], maxHp, tools_count, appearThisTurn
    
    if isinstance(pokemon_obj, dict):
        cid = pokemon_obj.get('id', -1) + 1
        hp = float(pokemon_obj.get('hp', 0))
        maxHp = float(pokemon_obj.get('maxHp', 0))
        en_cards = pokemon_obj.get('energyCards', [])
        energies_count = float(len(en_cards))
        retreat_cost = float(pokemon_obj.get('retreatCost', 0))
        tools_count = float(len(pokemon_obj.get('tools', [])))
        appearThisTurn = 1.0 if pokemon_obj.get('appearThisTurn', False) else 0.0
        
        # Extract damage of first attack if available
        dmg = 0.0
        attacks = pokemon_obj.get('attacks', [])
        if attacks and len(attacks) > 0:
            first_attack = attacks[0]
            if isinstance(first_attack, dict):
                dmg = float(first_attack.get('damage', 0))
            else:
                dmg = float(getattr(first_attack, 'damage', 0))
                
        e_ids = [0, 0, 0, 0]
        for idx, e in enumerate(en_cards[:4]):
            if isinstance(e, dict):
                e_ids[idx] = e.get('id', -1) + 1
            else:
                e_ids[idx] = e.id + 1 if hasattr(e, 'id') else 0
                
        return cid, hp, energies_count, retreat_cost, dmg, e_ids[0], e_ids[1], e_ids[2], e_ids[3], maxHp, tools_count, appearThisTurn
        
    return 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0

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
    active_id, active_hp, active_energy, active_rc, active_dmg, ae1, ae2, ae3, ae4, a_maxHp, a_tools, a_appear = get_pokemon_stats(my_state.active[0] if my_state.active else None)
    card_ids.extend([active_id, ae1, ae2, ae3, ae4])
    scalars.extend([active_hp, active_energy, active_rc, active_dmg, a_maxHp, a_tools, a_appear])
    
    # My Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en, b_rc, b_dmg, be1, be2, be3, be4, b_maxHp, b_tools, b_appear = get_pokemon_stats(my_state.bench[i] if i < len(my_state.bench) else None)
        card_ids.extend([b_id, be1, be2, be3, be4])
        scalars.extend([b_hp, b_en, b_rc, b_dmg, b_maxHp, b_tools, b_appear])
        
    # My Hand
    for i in range(MAX_HAND_SIZE):
        if i < len(my_state.hand):
            card = my_state.hand[i]
            cid = (card.id + 1) if hasattr(card, 'id') else (card.get('id', -1) + 1 if isinstance(card, dict) else 0)
            card_ids.append(cid)
        else:
            card_ids.append(0)
            
    # Opp Active
    opp_act_id, opp_act_hp, opp_act_en, opp_act_rc, opp_act_dmg, oae1, oae2, oae3, oae4, oa_maxHp, oa_tools, oa_appear = get_pokemon_stats(opp_state.active[0] if opp_state.active else None)
    card_ids.extend([opp_act_id, oae1, oae2, oae3, oae4])
    scalars.extend([opp_act_hp, opp_act_en, opp_act_rc, opp_act_dmg, oa_maxHp, oa_tools, oa_appear])
    
    # Opp Bench
    for i in range(MAX_BENCH_SIZE):
        b_id, b_hp, b_en, b_rc, b_dmg, be1, be2, be3, be4, b_maxHp, b_tools, b_appear = get_pokemon_stats(opp_state.bench[i] if i < len(opp_state.bench) else None)
        card_ids.extend([b_id, be1, be2, be3, be4])
        scalars.extend([b_hp, b_en, b_rc, b_dmg, b_maxHp, b_tools, b_appear])
        
    # Variables globales
    scalars.append(float(len(my_state.prize)))
    scalars.append(float(len(opp_state.prize)))
    scalars.append(float(my_state.deckCount))
    scalars.append(float(opp_state.handCount))
    scalars.append(float(opp_state.deckCount))
    
    # Game level flags
    current = obs.current
    scalars.append(float(current.turn))
    scalars.append(1.0 if current.firstPlayer == my_index else 0.0)
    scalars.append(1.0 if current.energyAttached else 0.0)
    scalars.append(1.0 if current.supporterPlayed else 0.0)
    scalars.append(1.0 if current.retreated else 0.0)
    scalars.append(1.0 if current.stadiumPlayed else 0.0)
    scalars.append(float(current.turnActionCount))
    
    # Stadium
    stadium = current.stadium
    stadium_id = float(stadium[0].id + 1) if stadium and len(stadium) > 0 else 0.0
    stadium_owner = 1.0 if stadium and len(stadium) > 0 and stadium[0].playerIndex == my_index else 0.0
    scalars.extend([stadium_id, stadium_owner])
    
    # Special conditions my
    scalars.append(1.0 if my_state.poisoned else 0.0)
    scalars.append(1.0 if my_state.burned else 0.0)
    scalars.append(1.0 if my_state.asleep else 0.0)
    scalars.append(1.0 if my_state.paralyzed else 0.0)
    scalars.append(1.0 if my_state.confused else 0.0)
    
    # Special conditions opp
    scalars.append(1.0 if opp_state.poisoned else 0.0)
    scalars.append(1.0 if opp_state.burned else 0.0)
    scalars.append(1.0 if opp_state.asleep else 0.0)
    scalars.append(1.0 if opp_state.paralyzed else 0.0)
    scalars.append(1.0 if opp_state.confused else 0.0)
    
    # Discard counts
    my_discard = my_state.discard if hasattr(my_state, 'discard') else []
    scalars.append(float(len(my_discard)))
    
    # Select info
    select_dict = obs_dict.get('select', {})
    scalars.append(float(select_dict.get('type', -1)))
    scalars.append(float(select_dict.get('context', -1)))
    
    # --- 2. Action Mask ---
    action_mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
    options = obs_dict.get('select', {}).get('option', [])
    num_options = len(options)
    
    if num_options > 0:
        for i in range(num_options):
            opt = options[i]
            opt_type = opt.get('type', -1)
            
            # --- SYNERGY MASKING: Do not attach energy to Pokemon that are already full ---
            is_valid_synergy = True
            if opt_type == 8: # OptionType.ATTACH
                # Check how many energies the target has
                in_play_area = opt.get('inPlayArea', -1)
                in_play_idx = opt.get('inPlayIndex', 0)
                target = None
                if in_play_area == 4 and my_state.active: # ACTIVE
                    target = my_state.active[0]
                elif in_play_area == 5 and in_play_idx < len(my_state.bench): # BENCH
                    target = my_state.bench[in_play_idx]
                
                if target:
                    # if target is a dict or object
                    en_cards = target.get('energyCards', []) if isinstance(target, dict) else getattr(target, 'energyCards', [])
                    target_id = target.get('id', -1) if isinstance(target, dict) else getattr(target, 'id', -1)
                    target_id = (target_id + 1) if target_id != -1 else 0
                    # Synergy Masking - Hard constraints on who can receive energy
                    # 162: Slowpoke, 163: Slowking (Primary)
                    # 756: Kangaskhan, 272: Clefairy (Secondary)
                    # 184: Latias ex (Pivot, needs 1 energy for combo)
                    if target_id in {162, 163, 756, 272}:
                        max_energy = 3
                    elif target_id == 184:
                        max_energy = 1
                    else:
                        # Utility Pokemon
                        if in_play_area == 4: # ACTIVE spot
                            # If they have retreat cost 1, allow 1 energy to retreat
                            if target_id in {140, 115}: # Fezandipiti, Bidoof
                                max_energy = 1
                            else:
                                # High retreat cost (Bibarel, Kyurem) -> let them get knocked out to save energy
                                max_energy = 0
                        else:
                            # If BENCHED, block energy on utility pokemon
                            max_energy = 0
                    
                    if len(en_cards) >= max_energy:
                        is_valid_synergy = False
            
            if is_valid_synergy:
                sem_act = map_env_to_semantic_action(i, obs_dict)
                if 0 <= sem_act < MAX_OPTIONS:
                    action_mask[sem_act] = 1
    else:
        # Si no hay opciones, permitimos la acción 0 para pasar
        action_mask[0] = 1
    
    # Rellenar con ceros hasta 111 por si acaso, y cortar a 111
    while len(scalars) < 111:
        scalars.append(0.0)
        
    return {
        "card_ids": np.array(card_ids, dtype=np.int32),
        "scalars": np.array(scalars[:111], dtype=np.float32),
        "action_mask": action_mask
    }
