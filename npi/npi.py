import torch
from torch import nn
import torch.nn.functional as F
from npi_core import NPICore


class PKeyMem(nn.Module):
    def __init__(self, n_progs, pkey_dim):
        super(PKeyMem, self).__init__()
        self.n_progs = n_progs
        self.pkey_mem = nn.Parameter(torch.randn(n_progs, pkey_dim))

    def is_act(self, prog_id):
        return prog_id == 0

    def calc_correlation_scores(self, pkey):
        return (self.pkey_mem @ pkey.unsqueeze(1)).view(-1)


class NPI(nn.Module):
    def __init__(self,
                 core,
                 task,
                 pkey_mem,
                 ret_threshold,
                 n_progs,
                 prog_dim):
        super(NPI, self).__init__()
        self.core = core
        self.task = task
        self.ret_threshold = ret_threshold
        self.n_progs = n_progs
        self.prog_dim = prog_dim
        self.pkey_mem = pkey_mem
        self.prog_mem = nn.Parameter(torch.randn(n_progs, prog_dim))

    def forward(self, env, prog_id, args):
        state = self.task.f_enc(env, args)
        prog = self.prog_mem[prog_id]
        ret, pkey, new_args = self.core(state, prog)
        scores = self.pkey_mem.calc_correlation_scores(pkey)
        prog_id_log_probs = F.log_softmax(scores, dim=0)  # log softmax is more numerically stable
        return ret, prog_id_log_probs, new_args

    def run(self, env, prog_id, args):
        ret = 0
        while ret < self.ret_threshold:
            ret, prog_id_log_probs, args = self.forward(env, prog_id, args)
            prog_id = torch.argmax(prog_id_log_probs)

            yield ret, env, prog_id, args
            if self.pkey_mem.is_act(prog_id):
                env = self.task.f_env(env, prog_id, args)
            else:
                yield from self.run(env, prog_id, args)  # todo: change to iteration


def npi_factory(task,
                state_dim,  # state tensor dimension
                n_progs,  # number of programs
                prog_dim,  # program embedding dimension
                hidden_dim,  # LSTM hidden dimension (in core)
                n_lstm_layers,  # number of LSTM layers (in core)
                ret_threshold,  # return probability threshold
                pkey_dim,  # program key dimension
                args_dim):  # argument vector dimension
    core = NPICore(state_dim=state_dim,
                   prog_dim=prog_dim,
                   hidden_dim=hidden_dim,
                   n_lstm_layers=n_lstm_layers,
                   pkey_dim=pkey_dim,
                   args_dim=args_dim)

    pkey_mem = PKeyMem(n_progs=n_progs,
                       pkey_dim=pkey_dim)

    npi = NPI(core=core,
              task=task,
              pkey_mem=pkey_mem,
              ret_threshold=ret_threshold,
              n_progs=n_progs,
              prog_dim=prog_dim)

    return npi


if __name__ == '__main__':
    import random
    import sys
    from task_base import TaskBase

    seed = random.randrange(sys.maxsize)
    print('seed= {}'.format(seed))
    torch.manual_seed(seed)
    # good seeds: 1528524055033086069, 8996695485408183525, 603660310440929170, 7859767191706266139

    state_dim = 2
    args_dim = 3


    class DummyTask(TaskBase):
        def __init__(self, state_dim):
            self.state_dim = state_dim

        def f_enc(self, env, args):
            return torch.randn(self.state_dim)

        def f_env(self, env, prog_id, args):
            return torch.randn(1)


    dummy_task = DummyTask(state_dim)

    npi = npi_factory(task=dummy_task,
                      state_dim=state_dim,
                      n_progs=4,
                      prog_dim=5,
                      hidden_dim=3,
                      n_lstm_layers=2,
                      ret_threshold=0.2,
                      pkey_dim=3,
                      args_dim=args_dim)

    ret, prog_id_probs, new_args = npi(42, 1, torch.randn(args_dim))
    it = npi.run(42, 1, torch.randn(args_dim))
    for x in it:
        print(x)
    # or run with next(it)