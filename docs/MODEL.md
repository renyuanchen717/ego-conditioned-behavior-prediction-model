# D_m2i architecture
Frozen NDPNet encodes history and map and returns six native trajectories, probabilities, agent embeddings and map embeddings. Its weights never receive gradients.
D_m2i adapts these features, encodes the full supplied ego future, applies two agent-interaction layers, and decodes six **direct** proposals in each target-local frame. These are not CV-based residual trajectories and there is no impact router.

`FrozenContextAdapter -> EgoPlanEncoder -> AgentInteraction x2 -> DirectDecoder`

Outputs: `new [B,A,6,80,2]`, `pool [B,A,12,80,2]`. `pool[:,:,:6]` equals the input native trajectories exactly. The six new proposals in D_m2i itself have **no learned mode probabilities**.

The objective is group-balanced `L_pool + L_new + L_displacement` (endpoint weight zero). New-mode winner is minimum masked ADE; displacement supervision applies to consecutive valid timestamps. Selection is validation group-balanced 8s Pool12 minADE. See `losses.py` for the exact reductions.
Training: seed 0, AdamW lr 2e-4, weight decay .01, clipping 2, global batch 4, 30 epochs; best epoch 26. Frozen encoder caches use 1s history and 8s future. 61,975 training and 7,776 independent validation samples.

Optional D_m2i-P learns one distribution over all 12 modes with frozen trajectories (best epoch 1). D_m2i-P2 preserves native6 probabilities and separately normalizes new6 probabilities (best epoch 2). The two P2 distributions cannot be interpreted as a single normalized Pool12 distribution without an additional group weight.
