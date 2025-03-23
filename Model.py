import torch.nn, torchdiffeq
from Kernel import LiKernel

class LTCODE(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.log_tau = torch.nn.Parameter(torch.randn(hidden_size))
        self.input_map = torch.nn.Linear(input_size, hidden_size)
        self.recurrent_map = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.BLOCK_SIZE = 64
        assert hidden_size % self.BLOCK_SIZE == 0, f"Hidden size must be multiple of {self.BLOCK_SIZE}"
        
    def forward(self, t, h):
        batch_size = h.shape[0]
        output = torch.empty_like(h)
        current_input = self._current_input(t)
        grid = (batch_size * (self.hidden_size // self.BLOCK_SIZE),)
        LiKernel[grid](
            h,
            current_input,
            output,
            self.recurrent_map.weight,
            self.input_map.weight,
            self.gate.weight,
            self.log_tau,
            batch_size,
            self.hidden_size,
            self.input_size,
            h.stride(0),
            current_input.stride(0),
            output.stride(0),
            self.BLOCK_SIZE,
        )
        
        return self.norm(output)
    
    def set_input(self, input_func):
        self._current_input = input_func

class MAINMODEL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.odefunc = LTCODE(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, x, t_span=None):
        batch_size, seq_len, num_features = x.size()
        if t_span is None:
            t_span = torch.linspace(0.0, seq_len, seq_len + 1, device=x.device)
        else:
            t_span = torch.linspace(t_span[0], t_span[-1], seq_len + 1, device=x.device)
        
        def input_func(t):
            t_idx = torch.clamp(t.long(), 0, seq_len - 1)
            return x[:, t_idx, :].squeeze(1)
        self.odefunc.set_input(input_func)
        h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        solution = torchdiffeq.odeint(
            self.odefunc,
            h0,
            t_span,
            method='rk4',
            atol=1e-4,
            rtol=1e-4,
            options={'step_size': 1.0}
        )
      
        h_final = solution[-1]
        return self.fc(h_final), h_final
