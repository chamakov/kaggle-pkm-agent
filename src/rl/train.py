import os
import sys

# Si corre dentro de docker o similar, aseguramos rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl.env_wrapper import CabtGymEnv

# Intentar importar sb3-contrib (requerido para MaskablePPO)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    has_sb3 = True
except ImportError:
    has_sb3 = False
# Detectar el entorno de ejecución para guardar los modelos
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working'):
    # Entorno de Kaggle Notebook
    base_save_dir = "/kaggle/working/PokemonAgentPPO"
    print(f"Kaggle Notebook detectado. Guardando datos en: {base_save_dir}")
else:
    # Intentar montar Google Drive si está en Colab
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        base_save_dir = "/content/drive/MyDrive/PokemonAgentPPO"
        print(f"Colab detectado. Guardando datos en: {base_save_dir}")
    except ImportError:
        # Entorno local
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
        
        # Learning rate schedule
        from typing import Callable
        def linear_schedule(initial_value: float) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func
            
        # Tuning hyperparameters for TCG
        model = MaskablePPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=tensorboard_dir,
            n_steps=4096,           # Más pasos para capturar partidas largas
            batch_size=512,         # Batch size más grande
            learning_rate=linear_schedule(0.0003),
            ent_coef=0.01,          # Fomentar exploración (prevenir que solo pase el turno)
            policy_kwargs=dict(net_arch=[256, 256, 256])  # Red más profunda para 52 escalares
        )
        
        # Configurar el logger para exportar las métricas a un archivo progress.csv
        from stable_baselines3.common.logger import configure
        new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Entorno separado exclusivo para las evaluaciones
        from stable_baselines3.common.monitor import Monitor
        eval_env = CabtGymEnv()
        eval_env = Monitor(eval_env)
        
        # Usamos MaskableEvalCallback para evaluar correctamente el modelo con la máscara
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=results_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Callback para guardar checkpoints cada N pasos
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
        import glob
        
        class RotatedCheckpointCallback(CheckpointCallback):
            def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", keep_num: int = 2, **kwargs):
                super().__init__(save_freq, save_path, name_prefix, **kwargs)
                self.keep_num = keep_num

            def _on_step(self) -> bool:
                result = super()._on_step()
                if self.n_calls % self.save_freq == 0:
                    search_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*steps.zip")
                    files = glob.glob(search_pattern)
                    
                    def extract_step(filepath):
                        try:
                            # format is prefix_1234_steps.zip
                            return int(filepath.split("_")[-2])
                        except:
                            return 0
                            
                    files.sort(key=extract_step)
                    if len(files) > self.keep_num:
                        for f in files[:-self.keep_num]:
                            try:
                                os.remove(f)
                                buffer_f = f.replace(".zip", "_replay_buffer.pkl")
                                if os.path.exists(buffer_f):
                                    os.remove(buffer_f)
                            except Exception as e:
                                print(f"Warning: could not delete {f}: {e}")
                return result
        
        checkpoint_dir = os.path.join(base_save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = RotatedCheckpointCallback(
            save_freq=503808,  # 4096 * 123 (Cerca de 500k, múltiplo de n_steps)
            save_path=checkpoint_dir,
            name_prefix="ppo_cabt_model",
            keep_num=2
        )
        
        # Agrupar los callbacks
        callbacks = CallbackList([eval_callback, checkpoint_callback])
        
        print("Empezando el entrenamiento...")
        try:
            print("Entrenando... presiona Stop (⏹️) en Colab para pausar.")
            print("El mejor modelo se guardará automáticamente en la carpeta 'best_models'.")
            model.learn(total_timesteps=10000000, callback=callbacks)
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
