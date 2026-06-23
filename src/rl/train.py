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

def train_model():
    print("Inicializando entorno de entrenamiento vs Random...")
    env = CabtGymEnv(opponent_agent="random")
    
    if has_sb3:
        # Instantiate PPO with MultiInputPolicy (to handle the Dict observation space)
        # Using MultiInputPolicy is KEY because we have 'card_ids', 'scalars', and 'action_mask'
        
        # SB3 natively doesn't support action masks without sb3-contrib (MaskablePPO)
        # But we can either use sb3-contrib or apply a penalty. 
        # For this skeleton, we show standard PPO logic.
        model = PPO("MultiInputPolicy", env, verbose=1)
        
        print("Empezando el entrenamiento...")
        try:
            print("Entrenando... presiona Stop (⏹️) en Colab para pausar y guardar el modelo.")
            model.learn(total_timesteps=1000000000)
        except KeyboardInterrupt:
            print("\nEntrenamiento detenido manualmente por el usuario.")
        finally:
            model.save("ppo_cabt_model")
            print("Modelo guardado exitosamente en ppo_cabt_model.zip")
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
