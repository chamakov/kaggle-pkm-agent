import os
import sys

# Si corre dentro de docker o similar, aseguramos rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl.env_wrapper import CabtGymEnv

# Intentar importar SB3, si no está instalado, será solo un skeleton
try:
    from stable_baselines3 import PPO
    has_sb3 = True
except ImportError:
    has_sb3 = False
# Attempt to mount Google Drive if running in Colab
try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')
    base_save_dir = "/content/drive/MyDrive/PokemonAgentPPO"
    print(f"Colab detectado. Guardando datos en: {base_save_dir}")
except ImportError:
    base_save_dir = "."
    print("Entorno local detectado. Guardando datos localmente.")

def train_model():
    print("Inicializando entorno de entrenamiento vs Random...")
    env = CabtGymEnv(opponent_agent="random")
    
    if has_sb3:
        # Instantiate PPO with MultiInputPolicy (to handle the Dict observation space)
        # Using MultiInputPolicy is KEY because we have 'card_ids', 'scalars', and 'action_mask'
        
        # SB3 natively doesn't support action masks without sb3-contrib (MaskablePPO)
        # But we can either use sb3-contrib or apply a penalty. 
        # For this skeleton, we show standard PPO logic.
        # Directorios de guardado
        best_model_dir = os.path.join(base_save_dir, "best_models")
        tensorboard_dir = os.path.join(base_save_dir, "tensorboard_logs")
        results_dir = os.path.join(base_save_dir, "results")
        
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_dir)
        
        # Configurar el logger para exportar las métricas a un archivo progress.csv
        from stable_baselines3.common.logger import configure
        new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        from stable_baselines3.common.callbacks import EvalCallback
        
        # Entorno separado exclusivo para las evaluaciones
        eval_env = CabtGymEnv(opponent_agent="random")
        
        # Cada 10,000 pasos jugará 5 partidas en el eval_env.
        # Si obtiene una recompensa promedio mejor que el récord anterior, lo guarda en ./best_models/
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=results_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        print("Empezando el entrenamiento...")
        try:
            print("Entrenando... presiona Stop (⏹️) en Colab para pausar.")
            print("El mejor modelo se guardará automáticamente en la carpeta 'best_models'.")
            model.learn(total_timesteps=1000000000, callback=eval_callback)
        except KeyboardInterrupt:
            print("\nEntrenamiento detenido manualmente por el usuario.")
        finally:
            final_path = os.path.join(base_save_dir, "ppo_cabt_model_final")
            model.save(final_path)
            print(f"Último modelo (puede no ser el mejor) guardado en {final_path}.zip")
    else:
        print("StableBaselines3 no detectado. Modo Dummy:")
        obs, info = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f"Step {i}: Reward {reward}, Done {done}")
            if done:
                break

if __name__ == "__main__":
    train_model()
