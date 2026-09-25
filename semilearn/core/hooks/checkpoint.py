



import os

from .hook import Hook

class CheckpointHook(Hook):
    """
    Checkpoint Hook for saving checkpoint
    """
    def after_train_step(self, algorithm):

        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)

            if (not algorithm.distributed) or \
               (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.save_model('latest_model.pth', save_path)

                if algorithm.it == algorithm.best_it:
                    algorithm.save_model('model_best.pth', save_path)

