import tqdm


class ProgressBar:
    def __init__(self, num_iterations: int, verbose: bool = True, separate_scalars: bool = True):
        if verbose:  # create a nice little progress bar
            progress_bar_format = '{desc} {n_fmt:' + str(
                len(str(num_iterations))) + '}/{total_fmt}|{bar}|{elapsed}<{remaining}'
            if separate_scalars:
                self.scalar_tracker = tqdm.tqdm(total=num_iterations, desc="Scalars", bar_format="{desc}",
                                                position=0, leave=True)
                self.progress_bar = tqdm.tqdm(total=num_iterations, desc='Iteration', bar_format=progress_bar_format,
                                              position=1, leave=True)
            else:
                self.progress_bar = tqdm.tqdm(total=num_iterations, desc='Iteration', bar_format=progress_bar_format,
                                              position=0, leave=True)
                self.scalar_tracker = self.progress_bar
        else:
            self.scalar_tracker = None
            self.progress_bar = None

    def __call__(self, **kwargs):
        if self.progress_bar is not None:
            formatted_scalars = {key: "{:.3e}".format(value[-1] if isinstance(value, list) else value)
                                 for key, value in kwargs.items()}
            description = ("Scalars: " + "".join([str(key) + "=" + value + ", "
                                                  for key, value in formatted_scalars.items()]))[:-2]
            self.scalar_tracker.set_description(description)
            self.progress_bar.update(1)
