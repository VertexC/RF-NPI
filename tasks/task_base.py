import abc


class TaskBase(abc.ABC):
    """
    abstract base class for tasks
    for each task, inherit and overwrite the methods
    note the type of the return value depends on the state dimension
    """

    @abc.abstractmethod
    def f_enc(self, env, args):
        """encoder fn:: (Env, args: Tensor[args_dim]) -> Tensor[state_dim]"""

    @abc.abstractmethod
    def f_env(self, env, prog_id, args):
        """env fn:: (Env, args: Tensor[args_dim], prog_id: int) -> Env"""