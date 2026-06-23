try:
    from kaggle_environments.envs.cabt import cg
    print(dir(cg))
    
    # Try importing sim
    from kaggle_environments.envs.cabt.cg import sim
    print("sim:", dir(sim))
except ImportError as e:
    print(e)
