import os
import sys

# Si corre dentro de docker o similar, aseguramos rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl.env_wrapper import CabtGymEnv

# Intentar importar sb3-contrib (requerido para MaskablePPO)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    has_sb3 = True
except ImportError:
    has_sb3 = False

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working'):
    base_save_dir = "/kaggle/working/PokemonAgentPPO"
    print(f"Kaggle Notebook detectado. Guardando datos en: {base_save_dir}")
else:
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        base_save_dir = "/content/drive/MyDrive/PokemonAgentPPO"
        print(f"Colab detectado. Guardando datos en: {base_save_dir}")
    except ImportError:
        base_save_dir = "."
        print("Entorno local detectado. Guardando datos localmente.")

if has_sb3:
    class CardEmbeddingExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Dict, embedding_dim: int = 32):
            card_ids_space = observation_space.spaces['card_ids']
            scalars_space = observation_space.spaces['scalars']
            
            self.num_cards = card_ids_space.shape[0] # 90
            self.num_scalars = scalars_space.shape[0] # 111
            
            features_dim = (self.num_cards * embedding_dim) + self.num_scalars
            
            super().__init__(observation_space, features_dim)
            
            # Card ID Embedding layer (max id is around 1300)
            self.embedding = nn.Embedding(num_embeddings=1300 + 10, embedding_dim=embedding_dim, padding_idx=0)
            
        def forward(self, observations: dict) -> torch.Tensor:
            card_ids = observations['card_ids'].long()
            # Clamp to prevent out-of-bounds indices
            card_ids = torch.clamp(card_ids, 0, self.embedding.num_embeddings - 1)
            
            scalars = observations['scalars'].float()
            
            embedded_cards = self.embedding(card_ids)
            embedded_cards_flat = embedded_cards.view(embedded_cards.size(0), -1)
            
            features = torch.cat([embedded_cards_flat, scalars], dim=1)
            return features

def make_env(opponent="random", rank=0, seed=0):
    def _init():
        env = CabtGymEnv(opponent_agent=opponent)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_model():
    print("Inicializando entorno de entrenamiento con SubprocVecEnv...")
    
    if has_sb3:
        num_envs = 4
        # Train against random for generic robustness or heuristic for challenge
        # Env wrapper internally chooses random, heuristic, lucario if passed random but available
        env = SubprocVecEnv([make_env(opponent="random", rank=i) for i in range(num_envs)])
        
        best_model_dir = os.path.join(base_save_dir, "best_models")
        tensorboard_dir = os.path.join(base_save_dir, "tensorboard_logs")
        results_dir = os.path.join(base_save_dir, "results")
        
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
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
            n_steps=4096 // num_envs, # Total batch is n_steps * num_envs
            batch_size=512,
            gamma=0.995,            # Better credit assignment for delayed win/loss
            learning_rate=linear_schedule(0.0003),
            ent_coef=0.01,
            policy_kwargs=dict(
                features_extractor_class=CardEmbeddingExtractor,
                features_extractor_kwargs=dict(embedding_dim=32),
                net_arch=[256, 256, 256]
            )
        )
        
        from stable_baselines3.common.logger import configure
        new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Eval Env - ALWAYS vs heuristic to benchmark progress properly
        eval_env = make_env(opponent="heuristic", rank=99)()
        
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=results_dir,
            eval_freq=10000 // num_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=30, # Aumentado a 30 episodios para evaluación robusta en TCG
            verbose=1
        )
        
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
        import glob
        
        class WinRateCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.wins = {"heuristic": 0, "random": 0, "lucario": 0, "unknown": 0}
                self.matches = {"heuristic": 0, "random": 0, "lucario": 0, "unknown": 0}

            def _on_step(self) -> bool:
                for idx, done in enumerate(self.locals.get("dones", [])):
                    if done:
                        info = self.locals.get("infos", [{}])[idx]
                        opponent = info.get("opponent", "unknown")
                        is_success = info.get("is_success", False)
                        
                        if opponent not in self.matches:
                            self.matches[opponent] = 0
                            self.wins[opponent] = 0
                            
                        self.matches[opponent] += 1
                        if is_success:
                            self.wins[opponent] += 1
                            
                        if sum(self.matches.values()) % 50 == 0:
                            for opp in self.matches.keys():
                                if self.matches[opp] > 0:
                                    win_rate = self.wins[opp] / self.matches[opp]
                                    self.logger.record(f"win_rate/{opp}", win_rate)
                return True
        
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
                                pass
                return result
        
        checkpoint_dir = os.path.join(base_save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = RotatedCheckpointCallback(
            save_freq=500000 // num_envs,
            save_path=checkpoint_dir,
            name_prefix="ppo_cabt_model",
            keep_num=2
        )
        
        winrate_callback = WinRateCallback()
        
        callbacks = CallbackList([eval_callback, checkpoint_callback, winrate_callback])
        
        print("Empezando el entrenamiento...")
        try:
            model.learn(total_timesteps=10000000, callback=callbacks)
        except KeyboardInterrupt:
            print("\nEntrenamiento detenido manualmente por el usuario.")
        finally:
            final_path = os.path.join(base_save_dir, "ppo_cabt_model_final")
            model.save(final_path)
            print(f"Último modelo guardado en {final_path}.zip")
    else:
        print("StableBaselines3 no detectado. Modo Dummy:")
        env = CabtGymEnv()
        obs, info = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f"Step {i}: Reward {reward}, Done {done}")
            if done:
                break

if __name__ == "__main__":
    train_model()
