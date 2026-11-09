"""Lead attention: a small neural corrector that learns, at every origin, which other locations (and which part of
their recent lag profile) lead the target. It generalizes the two hand-built forms of cross-regional lag
information tested in cell24/25:
  * uniform pooling over locations ("coll" features)     -> mode="mean"  (attention weights fixed and equal)
  * hard, frozen pairwise leaders at fixed lags ("pair")  -> replaced by soft, state-dependent attention
Tokens for location j (seen from target i), all dated <= s:
  growth lag profile g_j(s-l), l = 0..5 ; level gap z(y_j) - z(y_i) ; drawdown of j ; ED growth of j and availability
Query: the target's own features (backbone predicted change, own momentum, ED profile, gaps) and own lag profile.
Output: the backbone's log-median error at one horizon. Inputs are standardized on the training rows."""
import numpy as np, torch, torch.nn as nn

LP = 6          # lag profile length (weeks)


def tokens(F):
    """R x (R-1) x 10 token array for one origin (Features object of halo_lite)."""
    from covidfc.trackp.halo_lite import _z
    y, g = F.y, F.g; R = y.shape[1]
    prof = np.nan_to_num(np.stack([g[-1 - l] for l in range(LP)], 1))             # R x LP
    zl = np.array([_z(y[:, j]) for j in range(R)])
    dd = y[-1] - np.nanmax(y[-9:], 0)
    if F.e is not None:
        eg = np.nan_to_num(F.ge[-1]); ea = np.isfinite(F.e[-1]).astype(float)
    else:
        eg = np.zeros(R); ea = np.zeros(R)
    T = np.zeros((R, R - 1, LP + 4), np.float32)
    for i in range(R):
        js = [j for j in range(R) if j != i]
        T[i] = np.c_[prof[js], (zl[js] - zl[i])[:, None], dd[js][:, None], eg[js][:, None], ea[js][:, None]]
    return np.nan_to_num(T)


class _Net(nn.Module):
    def __init__(self, ft, fk, d=32, heads=4, mode="attn"):
        super().__init__()
        self.mode = mode
        self.q = nn.Sequential(nn.Linear(ft, d), nn.GELU(), nn.Linear(d, d))
        self.k = nn.Sequential(nn.Linear(fk, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.lin = nn.Linear(ft, 1)                                   # linear skip (ridge-like part)

    def forward(self, xt, xk, return_w=False):
        q = self.q(xt); k = self.k(xk)
        if self.mode == "attn":
            ctx, w = self.att(q[:, None], k, k, need_weights=True, average_attn_weights=True)
            ctx = ctx[:, 0]
        else:
            ctx, w = k.mean(1), None
        y = self.out(torch.cat([q, ctx], 1)).squeeze(-1) + self.lin(xt).squeeze(-1)
        return (y, w) if return_w else y


class LeadAttn:
    def __init__(self, mode="attn", epochs=80, lr=3e-3, wd=1e-3, seeds=(0, 1, 2), device=None):
        self.mode, self.epochs, self.lr, self.wd, self.seeds = mode, epochs, lr, wd, seeds
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, Xt, Tk, y):
        def ms(A):
            m, s = A.mean(0), A.std(0)
            const = s < 1e-3                    # constant on the training rows (e.g. ED not yet available)
            return np.where(const, 0.0, m), np.where(const, 1.0, s)
        self.mt, self.st = ms(Xt)
        self.mk, self.sk = ms(Tk.reshape(-1, Tk.shape[-1]))
        xt = torch.tensor(np.clip((Xt - self.mt) / self.st, -8, 8), dtype=torch.float32, device=self.dev)
        xk = torch.tensor(np.clip((Tk - self.mk) / self.sk, -8, 8), dtype=torch.float32, device=self.dev)
        yy = torch.tensor(y, dtype=torch.float32, device=self.dev)
        self.nets = []
        n = len(y); bs = 256
        for sd in self.seeds:
            torch.manual_seed(sd); net = _Net(Xt.shape[1], Tk.shape[-1], mode=self.mode).to(self.dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            g = torch.Generator(device="cpu").manual_seed(sd)
            for ep in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for b in range(0, n, bs):
                    idx = perm[b:b + bs].to(self.dev)
                    loss = nn.functional.huber_loss(net(xt[idx], xk[idx]), yy[idx], delta=0.3)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, Xt, Tk, return_w=False):
        xt = torch.tensor(np.clip((Xt - self.mt) / self.st, -8, 8), dtype=torch.float32, device=self.dev)
        xk = torch.tensor(np.clip((Tk - self.mk) / self.sk, -8, 8), dtype=torch.float32, device=self.dev)
        outs = [net(xt, xk, return_w=True) for net in self.nets]
        p = torch.stack([o[0] for o in outs]).mean(0).cpu().numpy()
        if return_w and outs[0][1] is not None:
            return p, torch.stack([o[1][:, 0] for o in outs]).mean(0).cpu().numpy()
        return p
