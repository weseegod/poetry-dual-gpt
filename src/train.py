# train.py
# =========
# Purpose: The training loop — where the model learns from data.
#
# =========================================================================
# CONCEPT: How does a language model learn?
# =========================================================================
#
# 1. Take a chunk of text tokens: [1, 5, 23, 89, 45, 2, 7, ...]
# 2. Split into input and target (shifted by 1):
#      input:  [1, 5, 23, 89, 45, 2, 7]
#      target: [5, 23, 89, 45, 2, 7, ?]
#    The model learns: "given all previous tokens, predict the next one"
# 3. Forward pass: model sees input, predicts logits (probabilities per token)
# 4. Loss = cross_entropy(predicted_logits, actual_target_tokens)
# 5. Backward pass: compute gradients of loss w.r.t every parameter
# 6. Optimizer step: update parameters to reduce loss
# 7. Repeat millions of times
#
# =========================================================================
# CONCEPT: Cross-Entropy Loss
# =========================================================================
# Measures how "wrong" the model's predictions are.
#   Loss = -log(P(correct_token))
#
# If model is 100% confident in correct token → loss = 0
# If model gives 0% probability to correct token → loss = infinity
# Random guessing among 12,000 tokens → loss ≈ ln(12000) ≈ 9.4
#
# Initial loss ~9.4 → final validation ~1.42 (per README)
#
# =========================================================================
# CONCEPT: Mixed Precision Training
# =========================================================================
# Uses lower-precision floats (bfloat16 or float16) for forward/backward
# passes to save GPU memory and run faster, while keeping master weights
# in float32 for precision.
#
# bfloat16 (recommended for L4 GPU):
#   - Same exponent range as float32 (no overflow issues)
#   - Less mantissa precision (7 bits vs 23)
#   - Perfect for training — dynamic range matters more than precision
#
# Usage:
#   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#       logits, loss = model(x, y)    # forward in bf16
#   loss.backward()                    # backward in bf16
#   optimizer.step()                   # update in float32
#
# For float16 (older GPUs like T4), you also need GradientScaler:
#   scaler = torch.cuda.amp.GradScaler()
#   scaler.scale(loss).backward()
#   scaler.step(optimizer)
#   scaler.update()
#
# =========================================================================
# CONCEPT: AdamW Optimizer
# =========================================================================
# Adam = Adaptive Moment Estimation (keeps running averages of gradients)
# AdamW = Adam with decoupled Weight Decay (the "correct" version)
#
# Weight decay: adds a small penalty for large weights → regularization
#   - Should apply to weight matrices (nn.Linear.weight, nn.Embedding.weight)
#   - Should NOT apply to biases and LayerNorm parameters
#
# How to separate parameters:
#   decay_params = [p for n,p in model.named_parameters() if p.dim() >= 2]
#   no_decay_params = [p for n,p in model.named_parameters() if p.dim() < 2]
#   optimizer = AdamW([
#       {'params': decay_params, 'weight_decay': 0.1},
#       {'params': no_decay_params, 'weight_decay': 0.0},
#   ], lr=2e-4, betas=(0.9, 0.95))
#
# betas=(0.9, 0.95):
#   - beta1=0.9: momentum (smoothing for gradient)
#   - beta2=0.95: variance smoothing (slightly lower than default 0.999,
#     common for Transformers — the loss landscape changes fast)
#
# =========================================================================
# CONCEPT: Cosine Learning Rate Schedule with Warmup
# =========================================================================
# Training large models is unstable at the start if LR is too high.
# Solution: gradually increase LR (warmup), then gradually decrease.
#
#   During warmup:  lr = max_lr × (step / warmup_steps)
#   After warmup:   lr = min_lr + 0.5 × (max_lr - min_lr) ×
#                        (1 + cos(π × progress))
#   where progress = (step - warmup) / (total_steps - warmup)
#
# Visual shape:
#
#   lr  ^
#   2e-4|         /‾‾‾‾‾‾‾‾‾‾‾‾‾\
#       |        /                   \
#       |       /                     \___________
#   2e-5|______/                                 →
#       +------|---------|------------|--------→ steps
#           warmup     50%          100%
#
# =========================================================================
# CONCEPT: Gradient Clipping
# =========================================================================
# If gradients become too large (exploding gradients), parameter updates
# can destroy what the model learned. Clipping caps the gradient norm.
#
#   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
# This scales down ALL gradients if their total norm exceeds 1.0.
#
# =========================================================================
# DATA PIPELINE PREVIEW
# =========================================================================
# The corpus file (data/poetry_corpus.txt) has one example per line:
#   <|start|> [LUC_BAT] Câu thơ sáu chữ, <|reply|> Câu thơ tám chữ nối theo. <|end|>
#
# We tokenize all lines and concatenate into one giant tensor of token IDs.
# Then each batch picks a random contiguous chunk:
#   x = data[start : start+block_size]
#   y = data[start+1 : start+block_size+1]
#
# =========================================================================
# IMPLEMENTATION PLAN
# =========================================================================
#
# 1. TrainingConfig dataclass — keep all hyperparameters in one place
#
# 2. load_and_tokenize(corpus_path, tokenizer_path, block_size)
#    - Load tokenizer, read file, encode each line, concatenate
#
# 3. get_batch(data, batch_size, block_size, device)
#    - Pick random indices, extract x and y, move to GPU
#
# 4. get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
#    - Cosine schedule with warmup formula
#
# 5. train(config)
#    - Main loop: load data → init model → for step in range(max_steps):
#        get batch → autocast forward → backward → clip → step → log
#
# 6. evaluate(model, data, config)
#    - Average loss on held-out data (no gradients)
#
# 7. save_checkpoint / load_checkpoint
#    - Save: model weights + optimizer state + step number
#    - Load: restore everything to resume training
#
# 8. main() — CLI entry point: parse --epochs, --batch_size, --device, etc.

# --- YOUR CODE BELOW ---
