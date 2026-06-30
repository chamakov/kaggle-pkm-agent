import os
import sys
import json
import argparse
from tqdm import tqdm

# Aseguramos rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl.env_wrapper import CabtGymEnv
from sb3_contrib import MaskablePPO

def evaluate_model(model_path, num_games=50, opponent="random"):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"Error: Modelo no encontrado en {model_path}")
        return
        
    print(f"Cargando modelo desde: {model_path}")
    try:
        from src.rl.train import CardEmbeddingExtractor
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
            "policy_kwargs": dict(
                features_extractor_class=CardEmbeddingExtractor,
                features_extractor_kwargs=dict(embedding_dim=32),
                net_arch=[256, 256, 256]
            )
        }
        # It's crucial to load the env to correctly set up the extractor
        env = CabtGymEnv(opponent_agent=opponent)
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    

    
    wins = 0
    losses = 0
    invalid_moves_total = 0
    total_steps = 0
    
    print(f"Iniciando torneo de {num_games} partidas...")
    
    for i in tqdm(range(num_games), desc="Torneo en progreso"):
        obs, info = env.reset()
        done = False
        steps = 0
        invalid_moves_episode = 0
        
        while not done:
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if info.get("invalid_action"):
                invalid_moves_episode += 1
                
        total_steps += steps
        invalid_moves_total += invalid_moves_episode
        
        # Consideramos recompensa > 0 como victoria contra random
        if reward > 0:
            wins += 1
        else:
            losses += 1
            
    win_rate = (wins / num_games) * 100
    avg_steps = total_steps / num_games
    avg_invalid = invalid_moves_total / num_games
    
    print("\n" + "="*50)
    print(f"RESULTADOS DEL TORNEO (PPO vs {opponent.capitalize()})")
    print("="*50)
    print(f"Partidas Jugadas : {num_games}")
    print(f"Victorias        : {wins}")
    print(f"Derrotas         : {losses}")
    print(f"Tasa de Victorias: {win_rate:.1f}%")
    print(f"Promedio Pasos   : {avg_steps:.1f} por partida")
    print(f"Mov. Inválidos   : {invalid_moves_total} total ({avg_invalid:.2f} promedio por partida)")
    print("="*50)
    
    # Exportar la última partida para verla en el visualizador
    replay_data = env.env.toJSON()
    replay_file = "replay_eval.json"
    with open(replay_file, "w") as f:
        json.dump(replay_data, f)
    print(f"Última partida guardada como '{replay_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo PPO entrenado")
    parser.add_argument("--model", type=str, required=True, help="Ruta al archivo .zip del modelo")
    parser.add_argument("--games", type=int, default=50, help="Número de partidas a evaluar")
    parser.add_argument("--opponent", type=str, default="random", help="Oponente (random, etc)")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.games, args.opponent)
