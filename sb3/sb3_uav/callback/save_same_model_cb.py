from stable_baselines3.common.callbacks import CheckpointCallback
import os
import shutil

class SaveSameModelCB(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        overwrite: bool = False,
        verbose: int = 0,
        hist_checkpoint_len: int = 5,
        checkpoint_dir_name: str = "checkpoints",
    ):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.overwrite = overwrite
        self.hist_checkpoint_len = hist_checkpoint_len
        self.checkpoint_dir_name = checkpoint_dir_name

        # Create checkpoint directory for historical saves ## the checkpoint would saved under save_path/checkpoint_dir_name
        self.checkpoint_dir = os.path.join(save_path, checkpoint_dir_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Current checkpoint path (directly under save_path, e.g., ./logs/)
        self.current_save_file = os.path.join(save_path, f"{name_prefix}.zip")
        self.current_replay_buffer_path = self.current_save_file.replace(".zip", "_replay_buffer.pkl")
        self.current_vecnormalize_path = self.current_save_file.replace(".zip", "_vecnormalize.pkl")

        # List to track historical checkpoint files
        self.checkpoint_history = []

        if self.overwrite:
            # 🧹 Remove existing folder if it exists
            conflicting_dir = os.path.join(save_path, name_prefix)
            if os.path.isdir(conflicting_dir):
                shutil.rmtree(conflicting_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.overwrite:
                self._save_overwrite()
            else:
                super()._on_step()
        return True
    
    def _save_overwrite(self):
        timesteps = self.num_timesteps

        # Save current checkpoint directly under save_path (e.g., ./logs/)
        if self.verbose > 0:
            print(f"Saving current checkpoint to {self.current_save_file}")
        self.model.save(self.current_save_file)
        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            self.model.save_replay_buffer(self.current_replay_buffer_path)
        if self.save_vecnormalize and hasattr(self.model, "get_vec_normalize_env"):
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(self.current_vecnormalize_path)

        # Save historical checkpoint in checkpoint directory
        hist_save_file = os.path.join(self.checkpoint_dir, f"{self.name_prefix}_{timesteps}_steps.zip")
        hist_replay_buffer_path = hist_save_file.replace(".zip", "_replay_buffer.pkl")
        hist_vecnormalize_path = hist_save_file.replace(".zip", "_vecnormalize.pkl")

        if self.verbose > 0:
            print(f"Saving historical checkpoint to {hist_save_file}")
        self.model.save(hist_save_file)
        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            self.model.save_replay_buffer(hist_replay_buffer_path)
        if self.save_vecnormalize and hasattr(self.model, "get_vec_normalize_env"):
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(hist_vecnormalize_path)

        # Track checkpoint files
        checkpoint_files = [hist_save_file]
        if os.path.exists(hist_replay_buffer_path):
            checkpoint_files.append(hist_replay_buffer_path)
        if os.path.exists(hist_vecnormalize_path):
            checkpoint_files.append(hist_vecnormalize_path)

        self.checkpoint_history.append(checkpoint_files)

        # Remove oldest checkpoint if exceeding hist_checkpoint_len
        if len(self.checkpoint_history) > self.hist_checkpoint_len:
            oldest_checkpoint = self.checkpoint_history.pop(0)
            for file_path in oldest_checkpoint:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if self.verbose > 0:
                        print(f"Removed old checkpoint: {file_path}")