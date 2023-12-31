from metadrive.envs.metadrive_env import MetaDriveEnv


def local_test_close_and_restart():
    try:
        for m in ["X", "O", "C", "S", "R", "r", "T"]:
            env = MetaDriveEnv({"map": m, "use_render": True})
            o, _ = env.reset()
            for _ in range(300):
                assert env.observation_space.contains(o)
                o, r, tm, tc, i = env.step([1, 1])
                if tm or tc:
                    break
            env.close()
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()


if __name__ == '__main__':
    local_test_close_and_restart()
