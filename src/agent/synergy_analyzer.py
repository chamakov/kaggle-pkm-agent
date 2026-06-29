class ComboSequencer:
    """
    Analyzes Kaggle CABT options and board state to assign a Progressive Bias (Synergy Score)
    to each legal move. This guides MCTS exploration towards logical sequences and safe plays.
    """
    
    def __init__(self):
        # Action type constants based on Kaggle cabt engine
        self.ACTION_PLAY_SUPPORTER_ITEM = 3
        self.ACTION_BENCH_POKEMON = 7
        self.ACTION_ATTACH_ENERGY = 8
        self.ACTION_EVOLVE = 9
        self.ACTION_ABILITY = 10
        self.ACTION_RETREAT = 12
        self.ACTION_ATTACK = 13
        self.ACTION_END_TURN = 14
        
        # Load MetaAnalyzer for persistent learning
        from src.agent.meta_analyzer import MetaAnalyzer
        self.meta_analyzer = MetaAnalyzer()
        self.threat_weights = {"hp": 1.5, "damage": 2.0}
        
    def evaluate_synergy(self, move_index: int, state) -> float:
        """
        Calculates a heuristic synergy score for a given move index.
        Returns a float that will be used in UCB1 Progressive Bias.
        """
        # If the state does not have cabt options or current board state, fallback to 0
        if not hasattr(state, "observation"):
            return 0.0
            
        obs = state.observation
        select_dict = obs.get("select", {})
        options = select_dict.get("option", [])
        
        if move_index >= len(options):
            return 0.0
            
        action = options[move_index]
        action_type = action.get("type")
        
        # Threat Evaluation & Archetype Detection Phase
        current = obs.get("current") or {}
        players = current.get("players", []) if current else []
        player_idx = obs.get("player", 0)
        opp_idx = 1 - player_idx
        
        threat_bonus = 0.0
        is_stall_mode = False
        is_utility_attack = False
        my_active = None
        opp_active = None
        my_state = {}
        opp_state = {}
        bench = []
        
        try:
            return self._evaluate_synergy_inner(
                action, action_type, obs, players, player_idx, opp_idx,
                threat_bonus, is_stall_mode, is_utility_attack,
                my_active, opp_active, my_state, opp_state, bench
            )
        except Exception:
            return 0.0
    
    def _evaluate_synergy_inner(self, action, action_type, obs, players, player_idx, opp_idx,
                                 threat_bonus, is_stall_mode, is_utility_attack,
                                 my_active, opp_active, my_state, opp_state, bench):
        if len(players) > 1:
            my_state = players[player_idx]
            opp_state = players[opp_idx]
            
            # Archetype Detection (Stall Mode)
            # If the opponent is much closer to decking out than us, we are in a Stall scenario
            my_deck_count = my_state.get("deckCount", 0)
            opp_deck_count = opp_state.get("deckCount", 0)
            if opp_deck_count <= my_deck_count - 10:
                is_stall_mode = True
                
            # Meta-Learning: Predict Opponent Archetype
            import src.agent.agent as agent_module
            if agent_module.global_belief_tracker and hasattr(agent_module.global_belief_tracker, "seen_opponent_cards"):
                seen = [str(c) for c in agent_module.global_belief_tracker.seen_opponent_cards]
                archetype_data = self.meta_analyzer.guess_opponent_archetype(seen)
                if archetype_data:
                    # We know what deck they are likely playing!
                    # For now, we print it out for debugging or use it to adjust threat weights.
                    pass # In future, we can target high-frequency cards in their archetype
                    
            
            my_active_list = my_state.get("active", [])
            opp_active_list = opp_state.get("active", [])
            my_active = my_active_list[0] if my_active_list and my_active_list[0] else None
            opp_active = opp_active_list[0] if opp_active_list and opp_active_list[0] else None
            bench = [b for b in my_state.get("bench", []) if b is not None]
            
            if my_active:
                # Detect Utility Attacks (0 damage or strategic effects)
                # Known examples: Elgyem BLK, Dedenne SSP
                # We flag attacking as 'utility' if our active pokemon is a known utility/stall piece
                # (In a full implementation, we would check a card DB for damage == 0)
                my_active_id = my_active.get("id", -1)
                # We assume if it's a control/stall deck, its attacks are mostly utility
                if is_stall_mode or my_active_id in [40, 87, 297]: 
                    is_utility_attack = True
            
            if my_active and opp_active:
                my_hp = my_active.get("hp", 0)
                my_damage = my_active.get("damage", 0)
                my_remaining_hp = my_hp - my_damage
                
                # Threat Heuristic: Is our active pokemon in danger?
                if my_remaining_hp > 0 and my_remaining_hp <= 60:
                    # If we are in danger, attacking to KO them or switching is good
                    if action_type == self.ACTION_ATTACK and not is_stall_mode:
                        threat_bonus += 20.0
                    elif action_type == self.ACTION_PLAY_SUPPORTER_ITEM:
                        threat_bonus += 5.0
                        
        # Dynamic Energy Caps Dictionary
        energy_caps = {
            162: {"cost": 2, "scales": False}, # Slowpoke
            163: {"cost": 3, "scales": False}, # Slowking
            183: {"cost": 0, "scales": False}, # Smoochum
            756: {"cost": 3, "scales": False}, # M Kangaskhan ex
            140: {"cost": 3, "scales": False}, # Fezandipiti ex
            184: {"cost": 3, "scales": False}, # Latias ex
            272: {"cost": 3, "scales": False}, # Lillie's Clefairy ex
            1071: {"cost": 1, "scales": False}, # Meowth ex (solo 1 para retirada)
            # Lucario deck
            335: {"cost": 2, "scales": False}, # Makuhita
            336: {"cost": 3, "scales": False}, # Hariyama
            423: {"cost": 1, "scales": False}, # Lunatone
            424: {"cost": 1, "scales": False}, # Solrock
            504: {"cost": 1, "scales": False}, # Riolu
            505: {"cost": 3, "scales": False}, # Mega Lucario ex
        }
        
        # Check Board Control (Gameplan)
        is_board_controlled = False
        if my_active and opp_active:
            active_id = my_active.get("id", -1)
            active_cap = energy_caps.get(active_id, {"cost": 3, "scales": False})
            my_energies = len(my_active.get("energyCards", []))
            my_hp = my_active.get("hp", 0) - my_active.get("damage", 0)
            opp_energies = len(opp_active.get("energyCards", []))
            
            # We control the board if our attacker is loaded & healthy, and opponent is not an immediate threat
            if my_energies >= active_cap["cost"] and my_hp > 100 and opp_energies <= 1:
                is_board_controlled = True
                
        # --- Pre-calculate Bench Metrics (Fixes NameErrors) ---
        bench_count = len(bench)
        
        # Known damage per Pokemon ID (best attack damage)
        pokemon_best_damage = {
            163: 200,  # Slowking
            162: 30,   # Slowpoke
            183: 0,    # Smoochum
            756: 100,  # M Kangaskhan
            140: 100,  # Fezandipiti
            184: 100,  # Latias
            272: 100,  # Clefairy
            # Lucario deck
            335: 20,   # Makuhita
            336: 80,   # Hariyama
            423: 20,   # Lunatone
            424: 20,   # Solrock
            504: 30,   # Riolu
            505: 150,  # Mega Lucario ex
        }
        
        my_active_id = my_active.get("id", -1) if my_active else -1
        my_active_best_dmg = pokemon_best_damage.get(my_active_id, 50)
        
        best_bench_dmg = 0
        best_bench_ready_dmg = 0
        best_bench_ready = False
        
        for b in bench:
            if b is None: continue
            bid = b.get("id", -1)
            b_dmg = pokemon_best_damage.get(bid, 50)
            b_energies = len(b.get("energyCards", []))
            b_cap = energy_caps.get(bid, {"cost": 3, "scales": False})
            
            if b_dmg > best_bench_dmg:
                best_bench_dmg = b_dmg
                
            if b_energies >= b_cap["cost"] and b_dmg >= 30:
                best_bench_ready = True
                if b_dmg > best_bench_ready_dmg:
                    best_bench_ready_dmg = b_dmg
                    
        if best_bench_ready:
            best_bench_dmg = best_bench_ready_dmg
        # ----------------------------------------------------
        
        # Context-Aware Sequencing Combo Flow
        base_score = 0.0
        
        if action_type == self.ACTION_PLAY_SUPPORTER_ITEM:
            # Gameplan-driven Supporter Logic
            index = action.get("index", -1)
            hand = my_state.get("hand", [])
            card_id = hand[index].get("id", -1) if 0 <= index < len(hand) else -1
            
            # Sub-menu selection (Discarding or Searching)
            # If the card selected is not a trainer, it's likely a sub-menu choice from hand (like Ultra Ball discard)
            is_submenu_card = card_id not in [1227, 1188, 1210, 1231, 1184, 1152, 1121, 1097, 1146, 1092, 1123, 1156, 1248]
            
            if is_submenu_card:
                if card_id in [162, 163, 1152, 1097, 1188, 1248]: # Do not discard core pieces or recovery
                    base_score = -20.0
                elif card_id in [144, 115, 224, 880]: # Bait/Combo pieces - PERFECT FOR DISCARD
                    base_score = 40.0
                else:
                    base_score = 5.0 # OK to discard/select
            else:
                # We are PLAYING an item or supporter!
                my_deck_count = my_state.get("deckCount", len(my_state.get("deck", [])))
                
                if card_id in [1188, 1248]: # Ciphermaniac, Academy at Night
                    # Solo usar si Slowking está en mesa, o si necesitamos piezas del combo con urgencia
                    has_slowking = any(p and p.get("id") == 163 for p in ([my_active] + bench) if p is not None)
                    needs_combo_pieces = bench_count == 0 or not has_slowking
                    
                    if has_slowking:
                        base_score = 90.0 # ¡Prioridad absoluta para preparar el ataque!
                    elif needs_combo_pieces:
                        base_score = 75.0 # Usar para buscar a Slowpoke/Slowking
                    else:
                        base_score = -10.0 # ¡Guardar en mano! No desperdiciarlo sin motivo.
                elif card_id in [1121, 1092]: # Ultra Ball, Secret Box
                    base_score = 60.0 # Good for searching
                elif card_id in [1227, 1210, 1231, 1184, 1152, 1097]: # Recursion and Draw
                    base_score = 50.0
                elif is_board_controlled:
                    base_score = 25.0 # Lower priority, but still better than ending turn if free
                elif is_stall_mode:
                    base_score = 55.0 # High priority for disruption
                else:
                    base_score = 50.0 # Standard item usage
                
        elif action_type == self.ACTION_BENCH_POKEMON:
            index = action.get("index", -1)
            # Check if this is a promotion (Active Pokemon was KO'd)
            
            # Prize value dictionary: EX Pokemon give 2 prizes when KO'd
            prize_value = {
                140: 2, 184: 2, 272: 2, 756: 2, 1071: 2  # EX Pokemon
                # All other basics are worth 1 prize
            }
            
            # If we have NO Active Pokemon, this action is a "Promote to Active" action.
            is_promoting = (my_active is None or not my_active.get("id"))
            
            if is_promoting:
                # We MUST promote something. Pick wisely.
                if index < len(bench):
                    target = bench[index]
                    target_id = target.get("id", -1)
                    energies = len(target.get("energyCards", []))
                    hp = target.get("hp", 0) - target.get("damage", 0)
                    pv = prize_value.get(target_id, 1)
                    
                    caps = energy_caps.get(target_id, {"cost": 3, "scales": False})
                    optimal_cost = caps["cost"]
                    
                    if energies >= optimal_cost and optimal_cost > 1:
                        # FULLY LOADED! Unleash the beast, regardless of prize value!
                        base_score = 40.0
                    elif energies >= 2 and pv == 1:
                        base_score = 30.0 # Promote partially loaded NON-EX attacker
                    elif energies >= 2 and pv == 2:
                        # Partially loaded EX - promote if needed
                        base_score = 25.0 
                    elif hp > 100 and pv == 1:
                        base_score = 15.0 # Non-EX tank
                    elif pv == 1:
                        base_score = 10.0 # 1-prize sacrifice. Better to lose this than an EX.
                    else:
                        base_score = 2.0  # UNLOADED EX. DO NOT PROMOTE UNLESS DESPERATE!
                else:
                    base_score = 5.0 # Fallback
            else:
                # We are playing a Basic Pokemon from Hand to Bench
                hand = my_state.get("hand", [])
                card_id = hand[index].get("id", -1) if 0 <= index < len(hand) else -1
                pv = prize_value.get(card_id, 1)
                
                # EMERGENCY BENCH: If bench has <2 Pokemon, benching is CRITICAL to avoid auto-loss
                if bench_count == 0:
                    emergency_bonus = 100.0  # MUST BENCH! Override all attacks!
                elif bench_count == 1:
                    emergency_bonus = 30.0   # Highly prioritize backup
                else:
                    emergency_bonus = 0.0
                
                bait_ids = [144, 115, 224, 880] # Kyurem, Conkeldurr, Annihilape, Spectrier
                
                # Score based on card value and bench urgency
                if card_id in bait_ids:
                    if bench_count == 0:
                        base_score = 50.0 + emergency_bonus # Bench only if desperate
                    else:
                        base_score = -50.0 # DO NOT BENCH BAIT! Keep in hand for Academy at Night!
                elif card_id == 162: # Slowpoke
                    base_score = 70.0 + emergency_bonus
                elif card_id == 183: # Smoochum
                    if bench_count == 0 or (my_active and my_active.get("id") == 183):
                        base_score = 65.0 + emergency_bonus
                    else:
                        base_score = 50.0 + emergency_bonus
                elif card_id == 756: # M Kangaskhan ex
                    base_score = 60.0 + emergency_bonus
                else:
                    base_score = 45.0 + emergency_bonus
                
        elif action_type == self.ACTION_EVOLVE:
            base_score = 65.0 # Always evolve if possible
            
        elif action_type == self.ACTION_ATTACH_ENERGY:
            in_play_area = action.get("inPlayArea", -1)
            in_play_index = action.get("inPlayIndex", 0)
            
            target_pokemon = None
            if in_play_area == 4: # Active
                target_pokemon = my_active
            elif in_play_area == 5: # Bench
                bench_list = my_state.get("bench", [])
                if in_play_index < len(bench_list):
                    target_pokemon = bench_list[in_play_index]
                    
            if target_pokemon:
                target_id = target_pokemon.get("id", -1)
                target_energies = len(target_pokemon.get("energyCards", []))
                
                caps = energy_caps.get(target_id, {"cost": 3, "scales": False})
                optimal_cost = caps["cost"]
                scales = caps["scales"]
                
                # Check for Hard Cap
                if target_energies >= optimal_cost and not scales:
                    base_score = -100.0 # DO NOT OVER-ATTACH! Enmascarar completamente.
                elif target_id == 183:
                    base_score = -100.0 # NUNCA dar energía a Smoochum (costo de ataque 0)
                else:
                    # Heuristics for Energy Attachment
                    active_energies = len(my_active.get("energyCards", [])) if my_active else 0
                    active_id = my_active.get("id", -1) if my_active else -1
                    active_cap = energy_caps.get(active_id, {"cost": 3, "scales": False})
                    active_dmg = pokemon_best_damage.get(active_id, 50)
                    
                    # Verificar si tenemos un atacante principal hambriento (Slowpoke/Slowking)
                    has_hungry_slowpoke = False
                    for p in ([my_active] + bench):
                        if p and p.get("id") in [162, 163]:
                            p_en = len(p.get("energyCards", []))
                            p_cap = energy_caps.get(p.get("id"))["cost"]
                            if p_en < p_cap:
                                has_hungry_slowpoke = True
                                break
                    
                    if target_id == 1071 and has_hungry_slowpoke:
                        base_score = -50.0 # NUNCA dar energía a Meowth si Slowpoke/Slowking la necesita
                    elif in_play_area == 4:
                        # ACTIVE ATTACHMENT
                        retreat_cost = my_active.get("retreatCost", 1) if my_active else 1
                        
                        if best_bench_ready and active_dmg <= 20 and active_energies < retreat_cost:
                            # ESCAPE PLAN: We have a loaded sweeper on the bench, and our active is a weak shield that is stuck!
                            # Force attachment to active so it can retreat!
                            base_score = 85.0
                        elif active_dmg >= 100 and active_energies < active_cap["cost"]:
                            base_score = 80.0 # Active is a strong attacker and NEEDS energy! Feed it!
                        elif active_energies < active_cap["cost"] and not best_bench_ready:
                            # LOAD AND HIT: If bench isn't ready, at least try to load the active so it can fight back
                            base_score = 65.0
                        else:
                            base_score = 60.0
                            
                        if threat_bonus > 0:
                            base_score += 5.0
                    elif in_play_area == 5:
                        # BENCH ATTACHMENT
                        target_dmg = pokemon_best_damage.get(target_id, 50)
                        if target_dmg >= 100 and target_energies < optimal_cost:
                            # HIGH PRIORITY: Energize a strong bench attacker (XerneasEX, Houndour)
                            if (active_energies >= active_cap["cost"] and active_dmg > 20) or (best_bench_ready and active_energies < my_active.get("retreatCost", 1) and active_dmg <= 20):
                                base_score = 50.0 # DO NOT rush energy here if active needs to escape, or is already good
                            elif active_energies >= active_cap["cost"] or active_dmg <= 20:
                                base_score = 75.0  # Active is full or weak, RUSH energy to strong bench!
                            else:
                                base_score = 65.0   # Active still needs energy, but bench is good backup
                            
                            # TIEBREAKER: Focus energy on one pokemon at a time, and prefer cheaper attackers
                            base_score += target_energies * 2.0
                            
                            opp_has_ex = False
                            if opp_active and {331: 2}.get(opp_active.get("id", -1), 1) == 2: opp_has_ex = True
                            for ob in opp_state.get("bench", []):
                                if ob and {331: 2}.get(ob.get("id", -1), 1) == 2: opp_has_ex = True
                                
                            if opp_has_ex and target_id == 331:
                                # We NEED our heavy hitter (XerneasEX) to fight their EX!
                                base_score += 10.0
                            else:
                                base_score -= optimal_cost * 1.0
                                
                        elif target_id in [40, 87, 530, 474, 297] and target_energies == 0:
                            base_score = 55.0  # Utility setup
                        elif active_energies >= active_cap["cost"]:
                            base_score = 70.0 # Active is ready, build backup!
                        else:
                            base_score = 50.0
            else:
                base_score = 1.0 # Fallback
        # Known attack damage dictionary (learned from replays)
        known_damage = {
            89: 20,    # Litten - weak
            205: 20,   # Crabominable - weak
            206: 120,  # Chewtle - strong
            458: 50,   # XerneasEX Aurora Beam
            459: 140,  # XerneasEX Rising Horns
            571: 0,    # Houndour Utility - 0 damage utility
            572: 30,   # Houndour attack
        }
        
        # Prize value for active pokemon
        active_prize_value = {331: 2, 756: 2, 505: 3, 140: 2, 184: 2, 272: 2}.get(my_active_id, 1) if my_active else 1
        
        if action_type == self.ACTION_RETREAT:
            # Retreat Logic - defensive, offensive, AND prize-protection
            my_remaining_hp = (my_active.get("hp", 0) - my_active.get("damage", 0)) if my_active else 0
            
            # MEAT-SHIELD LOGIC: If we are a 1-prize utility
            # and Slowking (163) is on the bench but NOT ready, we MUST NOT retreat!
            is_meat_shield = active_prize_value == 1 and my_active_id in [162, 183]
            has_slowking_on_bench = any(b and b.get("id") == 163 for b in bench)
            
            if is_meat_shield and has_slowking_on_bench and not best_bench_ready:
                base_score = -50.0 # DO NOT RETREAT! Hold the line!
            elif my_active and threat_bonus > 0 and my_remaining_hp <= 60:
                # DEFENSIVE: We are dying
                if opp_active and (opp_active.get("hp", 0) - opp_active.get("damage", 0)) <= 80 and len(my_active.get("energyCards", [])) > 0:
                    base_score = -20.0 # Stay and KO them!
                else:
                    # Flee! Extra urgency if we're an EX (2 prizes at stake)
                    base_score = 25.0 + (10.0 if active_prize_value == 2 else 0.0)
            elif active_prize_value == 2 and my_remaining_hp <= 120 and bench_count >= 1:
                # PRIZE PROTECTION: Our EX is taking damage, save it before they finish it!
                # Only retreat if there's a 1-prize pokemon to take the hit
                has_cheap_blocker = any(
                    b is not None and {331: 2}.get(b.get("id", -1), 1) == 1
                    for b in bench
                )
                if has_cheap_blocker:
                    base_score = 15.0 # Protect the EX!
                else:
                    base_score = -5.0 # No cheap blocker available
            elif best_bench_ready and best_bench_dmg > my_active_best_dmg * 2:
                # OFFENSIVE RETREAT: Swap to the heavy hitter
                base_score = 35.0
            else:
                base_score = -5.0 # Evitar retiros al azar
                
        elif action_type == self.ACTION_ATTACK:
            attack_id = action.get("attackId", -1)
            atk_dmg = known_damage.get(attack_id, -1) 
            
            if is_utility_attack:
                base_score = 12.0
            elif is_stall_mode:
                base_score = -15.0
            else:
                if atk_dmg == -1:
                    # Fallback if attack damage is unknown
                    active_id = my_active.get("id", -1) if my_active else -1
                    best_dmg = pokemon_best_damage.get(active_id, 50)
                    active_energies = len(my_active.get("energyCards", [])) if my_active else 0
                    active_cap = energy_caps.get(active_id, {"cost": 3})
                    
                    if active_energies >= active_cap["cost"]:
                        atk_dmg = best_dmg
                    else:
                        atk_dmg = 50 # Assume weak attack if not fully energized
                        
                # Base score scaled by damage output
                if atk_dmg >= 100:
                    base_score = 20.0  # Strong attack, high priority!
                elif atk_dmg >= 40:
                    base_score = 10.0  # Medium attack
                else:
                    # Weak attack (20 dmg). Only use if no better options exist.
                    # Check if there's a stronger attacker on bench ready to go
                    if best_bench_ready and best_bench_dmg > atk_dmg * 2:
                        base_score = -5.0  # DON'T WASTE A TURN on 20 dmg. Retreat instead!
                    else:
                        base_score = 4.0   # No better option, chip away
                        
                # Tiebreaker: Prefer higher attack IDs (usually the stronger attack)
                base_score += (attack_id * 0.001)
                
                # GAP 1 FIX: ATTACK WHEN READY boost (only for strong attacks)
                if my_active and atk_dmg >= 40:
                    active_energies = len(my_active.get("energyCards", []))
                    active_id = my_active.get("id", -1)
                    active_cap = energy_caps.get(active_id, {"cost": 3, "scales": False})
                    if active_energies >= active_cap["cost"]:
                        base_score += 10.0  # We're loaded with a real attack!
                        
                # THE LUCK FACTOR: Always prioritize attacking with Slowking if possible
                if my_active and my_active.get("id") == 163:
                    base_score += 35.0 # Attack even if top deck isn't guaranteed
                
                if threat_bonus > 0:
                    base_score += 15.0
                    
                # SECURE THE KO: Boost if opponent is low HP
                if opp_active and (opp_active.get("hp", 0) - opp_active.get("damage", 0)) <= atk_dmg:
                    base_score += 30.0  # This attack will KO them!
                elif opp_active and (opp_active.get("hp", 0) - opp_active.get("damage", 0)) <= 80:
                    base_score += 15.0
                    
                # Prefer the STRONGER attack when multiple options exist
                select_dict = obs.get("select", {})
                all_options = select_dict.get("option", [])
                attack_options = [(i, known_damage.get(o.get("attackId", -1), 50)) for i, o in enumerate(all_options) if o.get("type") == self.ACTION_ATTACK]
                if len(attack_options) > 1:
                    max_dmg = max(d for _, d in attack_options)
                    if atk_dmg == max_dmg:
                        base_score += 8.0  # This IS the strongest attack
                    elif atk_dmg < max_dmg:
                        base_score -= 10.0 # Penalize the weaker attack heavily
                
        elif action_type == self.ACTION_END_TURN:
            # GAP 2 FIX: End Turn should ALWAYS be the absolute last resort
            if is_stall_mode:
                # In stall mode, passing is acceptable but still not preferred over setup
                base_score = -2.0
            else:
                # In aggro, passing without attacking is terrible
                base_score = -15.0
        elif action_type == self.ACTION_ABILITY:
            # Use abilities like Academy at Night, Concealed Cards, etc.
            # Usually playing abilities is good, but we don't want infinite loops
            base_score = 45.0
        else:
            base_score = 1.0
            
        return base_score + threat_bonus
