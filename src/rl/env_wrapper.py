import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make

import sys
import os
try:
    from src.rl.vectorizer import vectorize_state, MAX_OPTIONS, MAX_CARD_ID
except ImportError:
    sys.path.append(os.getcwd())
    from src.rl.vectorizer import vectorize_state, MAX_OPTIONS, MAX_CARD_ID

class CabtGymEnv(gym.Env):
    def __init__(self, opponent_agent=None, my_index=0):
        super().__init__()
        self.env = make("cabt")
        self.my_index = my_index
        self.opp_index = 1 - my_index
        
        # Load Opponent Agents
        self.opponent_agents = {}
        
        # Heuristic Agent (Phase 1)
        try:
            from src.agent.agent import agent_fn as heuristic_fn
            self.opponent_agents["heuristic"] = heuristic_fn
        except ImportError:
            self.opponent_agents["heuristic"] = None
            
        # Lucario Agent
        try:
            import importlib.util
            import os
            import sys
            
            filepath = os.path.join(os.getcwd(), "scratch", "lucario_agent.py")
            if os.path.exists(filepath):
                spec = importlib.util.spec_from_file_location("lucario_agent", filepath)
                lucario_module = importlib.util.module_from_spec(spec)
                sys.modules["lucario_agent"] = lucario_module
                spec.loader.exec_module(lucario_module)
                self.opponent_agents["lucario"] = lucario_module.agent
            else:
                self.opponent_agents["lucario"] = None
        except Exception:
            self.opponent_agents["lucario"] = None
            
        # Slowking Agent (Mirror match)
        try:
            filepath = os.path.join(os.getcwd(), "scratch", "slowking_agent.py")
            if os.path.exists(filepath):
                spec = importlib.util.spec_from_file_location("slowking_agent", filepath)
                slowking_module = importlib.util.module_from_spec(spec)
                sys.modules["slowking_agent"] = slowking_module
                spec.loader.exec_module(slowking_module)
                self.opponent_agents["slowking"] = slowking_module.agent
            else:
                self.opponent_agents["slowking"] = None
        except Exception:
            self.opponent_agents["slowking"] = None
            
        # Random Agent (Fallback)
        self.opponent_agents["random"] = "random"
        
        self.current_opponent_type = "random"
            
        self.observation_space = spaces.Dict({
            "card_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(90,), dtype=np.int32),
            "scalars": spaces.Box(low=-1000.0, high=1000.0, shape=(111,), dtype=np.float32),
        })
        
        self.current_action_mask = np.ones(MAX_OPTIONS, dtype=np.int8)
        self.action_space = spaces.Discrete(MAX_OPTIONS)
        self.runner = None
        self._last_state = None
        
        # Cache attack damage for O(1) lookup by attackId (attack option has attackId, not damage directly)
        self._attack_damage_cache = {}
        try:
            _cwd = os.getcwd()
            _candidate_paths = [
                _cwd,
                os.path.join(_cwd, 'submission_v13'),
                os.path.join(_cwd, 'remotethings', 'cg_custom'),
                os.path.join(_cwd, '..', 'submission_v13'),
            ]
            for _p in _candidate_paths:
                if os.path.exists(os.path.join(_p, 'cg', 'api.py')):
                    sys.path.insert(0, _p)
                    break
            from cg.api import all_attack
            self._attack_damage_cache = {atk.attackId: atk.damage for atk in all_attack()}
        except Exception:
            pass  # Will fall back to 0 damage if cache fails
        self._slowking_deck = [
            162, 162, 162, 162,  # Slowpoke
            163, 163, 163,       # Slowking
            144, 144,            # Kyurem
            756, 756,            # Mega Kangaskhan ex
            115,                 # Conkeldurr
            184,                 # Latias ex
            140,                 # Fezandipiti ex
            224,                 # Annihilape
            1071,                # Meowth ex
            183,                 # Smoochum
            880,                 # Spectrier
            272,                 # Lillie's Clefairy ex
            1227, 1227, 1227, 1227,  # Lillie's Determination
            1188, 1188, 1188,        # Ciphermaniac's Codebreaking
            1210,                # Brock's Scouting
            1231,                # Dawn
            1184,                # Lana's Aid
            1152, 1152, 1152, 1152,  # Poké Pad
            1121, 1121, 1121, 1121,  # Ultra Ball
            1097, 1097, 1097,        # Night Stretcher
            1146, 1146, 1146,        # Wondrous Patch
            1092,                # Secret Box
            1123,                # Switch
            1156,                # Lucky Helmet
            1248, 1248, 1248, 1248,  # Academy at Night
            19, 19, 19, 19,          # Telepathic Psychic Energy
            5, 5, 5,                 # Psychic Energy
            9, 9, 9                  # Boomerang Energy
        ]
        
        # Load alternative deck for opponents
        self._lucario_deck = []
        try:
            with open("deck.csv", "r") as f:
                self._lucario_deck = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
        except Exception:
            self._lucario_deck = self._slowking_deck
            
    def action_masks(self) -> np.ndarray:
        """Required by sb3-contrib MaskablePPO to fetch valid actions."""
        return self.current_action_mask
        
    def _get_obs_dict(self, state):
        return state[self.my_index].observation
        
    def _is_my_turn(self, state):
        obs_dict = self._get_obs_dict(state)
        # It's our turn if we have options to select
        opts = obs_dict.get('select', {})
        if opts is None:
            return False
        return len(opts.get('option', [])) > 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select Opponent for this episode randomly
        import random
        available_opponents = [k for k, v in self.opponent_agents.items() if v is not None]
        if available_opponents:
            self.current_opponent_type = random.choice(available_opponents)
        else:
            self.current_opponent_type = "random"
            
        # We run env.reset() which initializes the state
        state = self.env.reset(num_agents=2)
        
        # Send deck for both players (Step 0)
        opp_deck = self._slowking_deck if self.current_opponent_type == "slowking" else self._lucario_deck
        
        actions = [None, None]
        actions[self.my_index] = self._slowking_deck
        actions[self.opp_index] = opp_deck
        state = self.env.step(actions)
        
        # Fast forward until it's our turn
        state = self._fast_forward(state)
        self._last_state = state
        
        vec = vectorize_state(self._get_obs_dict(state), self.my_index)
        self.current_action_mask = vec.pop("action_mask")
        return vec, {}
        
    def _fast_forward(self, state):
        """Step the game forward through opponent turns until it's our turn or game over."""
        max_ff_steps = 500  # Safety guard: never loop infinitely
        ff_steps = 0
        while not self._is_my_turn(state) and not self.env.done:
            ff_steps += 1
            if ff_steps > max_ff_steps:
                break  # Safety exit

            my_action = []
            opp_obs  = state[self.opp_index].observation
            opp_opts = opp_obs.get('select', {})

            if opp_opts and len(opp_opts.get('option', [])) > 0:
                agent_fn  = self.opponent_agents.get(self.current_opponent_type)
                min_count = opp_opts.get('minCount', 1)
                num_opts  = len(opp_opts['option'])

                if callable(agent_fn):
                    try:
                        actions_returned = agent_fn(opp_obs, None)
                        opp_action = actions_returned if isinstance(actions_returned, list) else [actions_returned]
                        # Fill remaining if minCount > 1
                        if len(opp_action) < min_count:
                            import random
                            valid_indices = list(range(num_opts))
                            for act in opp_action:
                                if act in valid_indices:
                                    valid_indices.remove(act)
                            random.shuffle(valid_indices)
                            opp_action.extend(valid_indices[:max(0, min_count - len(opp_action))])
                    except Exception:
                        import random
                        opp_action = [random.randint(0, num_opts - 1)]
                else:
                    # Random opponent — must respect minCount
                    import random
                    indices = list(range(num_opts))
                    random.shuffle(indices)
                    opp_action = indices[:max(1, min_count)]
            else:
                opp_action = []

            actions = [None, None]
            actions[self.my_index]  = my_action
            actions[self.opp_index] = opp_action
            state = self.env.step(actions)

        return state
        
    def step(self, action):
        my_action = [int(action)]
        
        # Check if the action is valid using the action mask
        # Since we use MaskablePPO, invalid actions are never sampled by the policy!
        # But we double check just in case (e.g. if a random fallback occurs).
        is_valid = False
        if self._last_state is not None:
            if self.current_action_mask[int(action)] == 1:
                is_valid = True
                
        if not is_valid:
            # Fallback to passing turn or first valid option if MaskablePPO somehow fails
            valid_indices = [i for i, m in enumerate(self.current_action_mask) if m == 1]
            if valid_indices:
                my_action = [valid_indices[0]]
            else:
                my_action = [0]
        
        # Handle multi-select actions
        obs_dict = self._get_obs_dict(self._last_state)
        select_dict = obs_dict.get('select', {})
        min_count = select_dict.get('minCount', 1)
        
        if min_count > 1:
            import random
            remaining_mask = self.current_action_mask.copy()
            if my_action[0] < len(remaining_mask):
                remaining_mask[my_action[0]] = 0
            for _ in range(min_count - 1):
                valid_idx = [i for i, m in enumerate(remaining_mask) if m == 1]
                if not valid_idx:
                    break
                extra = random.choice(valid_idx)
                my_action.append(extra)
                remaining_mask[extra] = 0
        
        opp_obs = self._last_state[self.opp_index].observation
        opp_opts = opp_obs.get('select', {})
        
        if opp_opts and len(opp_opts.get('option', [])) > 0:
            agent_fn = self.opponent_agents.get(self.current_opponent_type)
            
            if callable(agent_fn):
                try:
                    actions_returned = agent_fn(opp_obs, None)
                    opp_action = actions_returned if isinstance(actions_returned, list) else [actions_returned]
                except Exception as e:
                    import random
                    opp_action = [random.randint(0, len(opp_opts['option'])-1)]
            else:
                import random
                opp_action = [random.randint(0, len(opp_opts['option'])-1)]
        else:
            opp_action = []
            
        actions = [None, None]
        actions[self.my_index] = my_action
        actions[self.opp_index] = opp_action
        
        state = self.env.step(actions)
        
        # Check if the game is done
        done = self.env.done
        
        # Obtain the selected action object to shape rewards
        last_obs = self._get_obs_dict(self._last_state)
        options = last_obs.get('select', {}).get('option', [])
        action_obj = options[int(action)] if options and int(action) < len(options) else {}
        action_type = action_obj.get('type')
        
        intermediate_reward = 0.0
        played_card_id = -1
        
        try:
            if action_type in [7, 9]: # Bench or Evolve
                current = last_obs.get('current', {})
                players = current.get('players', [])
                if len(players) > self.my_index:
                    my_state = players[self.my_index]
                    hand = my_state.get('hand', [])
                    idx = action_obj.get('index', -1)
                    if 0 <= idx < len(hand):
                        c = hand[idx]
                        played_card_id = c.get('id', -1) if isinstance(c, dict) else getattr(c, 'id', -1)
        except Exception:
            pass
            
        # ------------------------------------------------------------------
        # Known card ID sets for the Slowking deck
        # ------------------------------------------------------------------
        POKEMON_IDS  = {162, 163, 144, 756, 115, 184, 140, 224, 1071, 183, 880, 272}
        TRAINER_IDS  = {1227, 1188, 1210, 1231, 1184, 1152, 1121, 1097, 1146, 1092, 1123, 1156, 1248}
        ENERGY_IDS   = {5, 9, 19}

        select_context = last_obs.get('select', {}).get('context', -1)
        if action_type == 7:  # OptionType.PLAY — play any card from hand
            if played_card_id in POKEMON_IDS:
                # ---- Board state assessment for situational decisions ----
                try:
                    current = last_obs.get('current', {})
                    players = current.get('players', [])
                    my_state   = players[self.my_index] if len(players) > self.my_index else {}
                    opp_state  = players[self.opp_index] if len(players) > self.opp_index else {}

                    bench      = my_state.get('bench', [])
                    active     = my_state.get('active', [])
                    bench_ids  = {b.get('id') for b in bench if isinstance(b, dict)}
                    active_id  = active[0].get('id', -1) if active else -1

                    # Is a Slowking already in play? (active or bench)
                    has_slowking = (163 in bench_ids) or (active_id == 163)
                    # How many Slowpoke/Slowking are in play total?
                    slowking_line_count = sum(
                        1 for bid in list(bench_ids) + [active_id] if bid in {162, 163}
                    )
                    # How many bench slots are free?
                    bench_slots_free = 5 - len(bench)
                    # Opponent prizes remaining (fewer = late game = need secondary attackers)
                    opp_prizes = len(opp_state.get('prize', []))
                    late_game = opp_prizes <= 2  # opponent close to winning, need damage NOW
                except Exception:
                    has_slowking = False
                    slowking_line_count = 0
                    bench_slots_free = 3
                    opp_prizes = 3
                    late_game = False

                # ============================================================
                # TIER 1 — Always bench Slowpoke. It is the heart of the deck.
                # ============================================================
                if played_card_id == 162:  # Slowpoke
                    if slowking_line_count < 3:
                        intermediate_reward += 3.0  # Strong: we want 2-3 Slowpoke/Slowking lines
                    else:
                        intermediate_reward += 0.5  # Already have enough — marginal value

                # ============================================================
                # TIER 1 — Latias ex: free-retreat engine, bench ASAP
                # ============================================================
                elif played_card_id == 184:  # Latias ex
                    if 184 not in bench_ids:
                        intermediate_reward += 2.5  # First Latias: essential
                    else:
                        intermediate_reward += 0.2  # Second: usually redundant

                # ============================================================
                # TIER 2 — Secondary attackers. Value depends on board state.
                # Condition: we have a Slowking line, OR we're in late game.
                # ============================================================
                elif played_card_id == 756:  # Kangaskhan ex — heavy hitter
                    if has_slowking or late_game:
                        if 756 not in bench_ids:
                            intermediate_reward += 2.0  # First Kangaskhan: backup attacker
                        else:
                            intermediate_reward += 0.2  # Already have one
                    else:
                        intermediate_reward += 0.5  # Setup phase — not a priority yet

                elif played_card_id == 272:  # Lillie's Clefairy ex — tech attacker
                    if has_slowking or late_game:
                        if 272 not in bench_ids:
                            intermediate_reward += 1.5
                        else:
                            intermediate_reward += 0.2
                    else:
                        intermediate_reward += 0.3

                elif played_card_id == 140:  # Fezandipiti ex — situational attacker
                    # Fezandipiti has a useful attack (Dark Feather can KO basics)
                    # Only valuable in late game or as last resort, never as primary
                    if late_game and bench_slots_free > 1:
                        intermediate_reward += 1.0  # Late-game emergency attacker
                    elif has_slowking and bench_slots_free > 2:
                        intermediate_reward += 0.0  # Neutral: bench is crowded
                    else:
                        intermediate_reward -= 1.5  # Early game or no room: discourage

                # ============================================================
                # TIER 3 — Support / pivot Pokemon (minor value)
                # ============================================================
                elif played_card_id == 144:  # Kyurem — bench blocker/pivot
                    if bench_slots_free > 2:
                        intermediate_reward += 0.8
                    else:
                        intermediate_reward += 0.1  # Don't waste bench slots

                elif played_card_id in {115, 224, 880, 183, 1071}:  # Pure bench fodder
                    intermediate_reward -= 0.5  # Minor penalty: fills bench without purpose

                else:
                    intermediate_reward += 0.1

            elif played_card_id in TRAINER_IDS:
                # Trainer card played from hand (supporter/item/stadium/tool)
                if played_card_id == 1248:  # Academy at Night (stadium)
                    intermediate_reward += 1.0
                elif played_card_id in {1227, 1188, 1210, 1231, 1184}:  # Supporters
                    intermediate_reward += 0.5
                elif played_card_id in {1152, 1121, 1097, 1146, 1092, 1123, 1156}:  # Items/tools
                    intermediate_reward += 0.3
            elif played_card_id in ENERGY_IDS:
                # Energy played from hand (manual attach — should rarely happen here)
                intermediate_reward += 0.2

        elif action_type == 3:  # OptionType.CARD — sub-selection (pick a card from revealed set)
            # This fires during Ciphermaniac, Ultra Ball, etc.
            # area==12 means LOOKING zone (cards revealed from deck)
            card_area = action_obj.get('area', -1)
            sub_card_id = -1
            try:
                looking_cards = last_obs.get('current', {}).get('players', [{}])[self.my_index].get('looking', [])
                idx = action_obj.get('index', -1)
                if card_area == 12 and 0 <= idx < len(looking_cards):  # 12 = LOOKING
                    c = looking_cards[idx]
                    sub_card_id = c.get('id', -1) if isinstance(c, dict) else getattr(c, 'id', -1)
            except Exception:
                pass

            # Priority when picking from revealed/searched cards:
            if sub_card_id in ENERGY_IDS:
                intermediate_reward += 2.0    # #1: Always grab energy for Slowking
            elif sub_card_id in {163, 162}:   # #2: Slowking/Slowpoke — evolution line
                intermediate_reward += 1.8
            elif sub_card_id == 184:           # #3: Latias ex — retreat engine
                intermediate_reward += 1.2
            elif sub_card_id in {756, 272}:   # #4: Kangaskhan, Clefairy — secondary attackers
                intermediate_reward += 0.8
            elif sub_card_id == 140:           # Fezandipiti — only in emergency
                intermediate_reward += 0.3
            elif sub_card_id in POKEMON_IDS:
                intermediate_reward += 0.2

        elif action_type == 9:  # OptionType.EVOLVE — evolving a pokemon
            # played_card_id is already extracted above for action_type in [7, 9]
            if played_card_id == 163:   # Slowking — this is THE core strategic evolution
                intermediate_reward += 3.5
            elif played_card_id == 144: # Kyurem
                intermediate_reward += 0.5
            else:
                intermediate_reward += 0.5

        elif action_type == 10: # OptionType.ABILITY
            intermediate_reward -= 0.2

        elif select_context == 1:  # SelectContext.SETUP_ACTIVE_POKEMON (fires when action_type is NOT 7/3/9/10)
            # During normal play action_type==7 fires first for plays from hand.
            # But if setup sends action_type==3 (CARD), this branch catches it.
            if played_card_id in POKEMON_IDS:
                if played_card_id == 162:
                    intermediate_reward += 2.0
                elif played_card_id == 756:
                    intermediate_reward += 1.5
                elif played_card_id == 184:
                    intermediate_reward += 0.5
                elif played_card_id == 140:
                    intermediate_reward -= 2.0
                elif played_card_id in {115, 224, 880, 183, 1071}:
                    intermediate_reward -= 1.5
        elif action_type == 8: # OptionType.ATTACH — attach energy/tool
            target_card_id = -1
            energy_card_id = -1  # which energy/tool is being attached
            latias_on_bench = False
            try:
                current = last_obs.get('current', {})
                players = current.get('players', [])
                if len(players) > self.my_index:
                    my_state = players[self.my_index]

                    # Which pokemon is receiving the attachment
                    in_play_area = action_obj.get('inPlayArea', -1)
                    in_play_idx  = action_obj.get('inPlayIndex', 0)
                    if in_play_area == 4:  # Active
                        active_list = my_state.get('active', [])
                        if active_list:
                            target_card_id = active_list[0].get('id', -1)
                    elif in_play_area == 5:  # Bench
                        bench_list = my_state.get('bench', [])
                        if in_play_idx < len(bench_list):
                            target_card_id = bench_list[in_play_idx].get('id', -1)

                    # Which card is being attached (from hand, area=2=HAND)
                    src_area  = action_obj.get('area', -1)
                    src_index = action_obj.get('index', -1)
                    if src_area == 2:  # HAND
                        hand = my_state.get('hand', [])
                        if 0 <= src_index < len(hand):
                            c = hand[src_index]
                            energy_card_id = c.get('id', -1) if isinstance(c, dict) else getattr(c, 'id', -1)

                    # Check if Latias ex is already on bench (enables free-retreat combo)
                    bench_ids = {b.get('id') for b in my_state.get('bench', []) if isinstance(b, dict)}
                    latias_on_bench = 184 in bench_ids
            except Exception:
                pass

            # ---- Telepathic Psychic Energy (id=19) + Latias combo ----
            # Attaching Telepathic energy lets that pokemon retreat for FREE (via Latias ability)
            # This is THE key mobility tool of the deck
            if energy_card_id == 19:  # Telepathic Psychic Energy
                if latias_on_bench:
                    # Latias on bench → free retreat enabled! Very high value
                    intermediate_reward += 4.0
                else:
                    # No Latias yet — still decent, can still attach Latias later
                    intermediate_reward += 1.0
            else:
                # Regular energy or tool attachment — use target-based rewards
                GOOD_ENERGY_TARGETS = {163: 2.5, 162: 1.5}  # Slowking, Slowpoke
                BAD_ENERGY_TARGETS  = {144: -2.0, 140: -2.0, 115: -1.5, 224: -1.5, 880: -1.5}
                if target_card_id in GOOD_ENERGY_TARGETS:
                    intermediate_reward += GOOD_ENERGY_TARGETS[target_card_id]
                elif target_card_id in BAD_ENERGY_TARGETS:
                    intermediate_reward += BAD_ENERGY_TARGETS[target_card_id]
                else:
                    intermediate_reward += 0.5  # neutral fallback
        elif action_type == 12: # OptionType.RETREAT
            # Retreating is strategic when using Latias ex + Telepathic energy combo
            try:
                current = last_obs.get('current', {})
                players = current.get('players', [])
                my_state = players[self.my_index] if len(players) > self.my_index else {}
                bench_ids = {b.get('id') for b in my_state.get('bench', []) if isinstance(b, dict)}
                latias_on_bench = 184 in bench_ids

                # Check if active pokemon has Telepathic energy attached (id=19)
                active_energies = []
                active_list = my_state.get('active', [])
                if active_list:
                    active_energies = [e.get('id') for e in active_list[0].get('energies', []) if isinstance(e, dict)]
                has_telepathic = 19 in active_energies

                attack_available = any(o.get('type') == 13 for o in options)

                if latias_on_bench and has_telepathic:
                    # FREE retreat via Latias combo — strategic and okay
                    if attack_available:
                        # Even with free retreat, skipping an attack to retreat is mild penalty
                        intermediate_reward -= 1.0
                    else:
                        # No attack available, free retreat to bring up better pokemon = good!
                        intermediate_reward += 1.0
                elif attack_available:
                    # Had attack, chose to pay retreat cost instead — strongly penalize
                    intermediate_reward -= 3.0
                else:
                    # No attack, not free — small penalty (paying retreat cost unnecessarily)
                    intermediate_reward -= 0.5
            except Exception:
                intermediate_reward -= 0.5

        elif action_type == 13: # OptionType.ATTACK — HIGHEST PRIORITY
            # ATTACK IS THE HIGHEST PRIORITY ACTION — outweighs all setup combined
            intermediate_reward += 5.0  # Strong base: always rewarded for attacking
            try:
                if last_obs and 'current' in last_obs:
                    my_players = last_obs['current']['players']
                    if len(my_players) > 1:
                        opp_active = my_players[self.opp_index].get('active', [{}])[0]

                        # Look up damage from pre-cached all_attack() by attackId
                        attack_id = action_obj.get('attackId')
                        dmg = float(self._attack_damage_cache.get(attack_id, 0)) if attack_id is not None else 0.0

                        if opp_active and opp_active.get('hp'):
                            opp_hp = float(opp_active.get('hp', 0))
                            if opp_hp > 0 and dmg > 0:
                                dmg_ratio = min(dmg, opp_hp) / opp_hp
                                # Massive bonus proportional to damage — up to +15 more
                                intermediate_reward += 15.0 * dmg_ratio
            except Exception:
                pass
        elif action_type == 14: # ACTION_END_TURN
            if len(options) > 1:
                try:
                    attack_available = any(o.get('type') == 13 for o in options)
                    if attack_available:
                        # Skipping an attack is CATASTROPHICALLY penalized — never do this
                        intermediate_reward -= 20.0
                    else:
                        current = last_obs.get('current', {})
                        players = current.get('players', [])
                        if len(players) > self.my_index:
                            my_state = players[self.my_index]
                            hand = my_state.get('hand', [])
                            has_energy = any(c.get('id') in [5, 9, 19] for c in hand if isinstance(c, dict))
                            if has_energy:
                                intermediate_reward -= 2.0
                except:
                    intermediate_reward -= 1.0

        # ----------------------------------------------------------------
        # Prize-taken intermediate reward: reward each KO that takes a prize
        # ----------------------------------------------------------------
        def get_my_prize_count(st):
            """My remaining prize cards (starts at 6, decreases as opp KOs my pokemon? No...)
            Actually in TCG: when YOU KO an opponent's pokemon, YOU take a prize card from YOUR pile.
            So YOUR prize count goes DOWN when you KO opponent pokemon.
            """
            if st is None: return 6
            try:
                obs = st[self.my_index].observation
                if isinstance(obs, dict) and 'current' in obs and 'players' in obs['current']:
                    return len(obs['current']['players'][self.my_index].get('prize', []))
            except Exception:
                pass
            return 6

        def get_opp_prize_count(st):
            """Opponent's remaining prize cards (starts at 6, decreases when I KO their pokemon)."""
            if st is None: return 6
            try:
                obs = st[self.my_index].observation
                if isinstance(obs, dict) and 'current' in obs and 'players' in obs['current']:
                    return len(obs['current']['players'][self.opp_index].get('prize', []))
            except Exception:
                pass
            return 6

        my_prizes_before  = get_my_prize_count(self._last_state)
        my_prizes_after   = get_my_prize_count(state)
        opp_prizes_after  = get_opp_prize_count(state)

        # When MY prize count decreases, it means I KO'd an opponent pokemon → reward
        if my_prizes_after < my_prizes_before:
            intermediate_reward += 5.0 * (my_prizes_before - my_prizes_after)

        # Terminal reward
        final_reward = state[self.my_index].reward if state[self.my_index].reward is not None else 0.0

        if done:
            if final_reward > 0:
                # Won — but HOW? Check if opponent still has prizes (deckout) or not (prize win)
                if opp_prizes_after > 0:
                    # Won by deckout — penalize, we never attacked enough to take prizes
                    final_reward = -5.0
                else:
                    # Won by taking all opponent prizes — the RIGHT way to win
                    final_reward = 10.0
            elif final_reward < 0:
                final_reward = -10.0  # Amplify loss signal

        # Scale intermediate rewards — must stay clearly below terminal win/loss
        SHAPING_SCALE = 0.2
        intermediate_reward *= SHAPING_SCALE
        
        reward = final_reward + intermediate_reward
        
        if not done:
            state = self._fast_forward(state)
            # Recheck done after fast_forward
            done = self.env.done
            
        self._last_state = state
        vec = vectorize_state(self._get_obs_dict(state), self.my_index)
        self.current_action_mask = vec.pop("action_mask")
        info = {
            "is_success": final_reward > 0 if done else False,
            "opponent": getattr(self, 'current_opponent_type', 'unknown')
        }
        
        return vec, float(reward), done, False, info
