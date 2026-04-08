<div align="center">

<div style="max-width: 820px; margin: 0 auto 28px; padding: 28px 32px; border-radius: 20px; background: linear-gradient(145deg, rgba(18, 18, 22, 0.94) 0%, rgba(8, 8, 12, 0.98) 100%); border: 1px solid rgba(255, 255, 255, 0.10); box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);">

<p style="margin: 0 0 8px; font-size: 11px; letter-spacing: 0.35em; text-transform: uppercase; color: rgba(255, 255, 255, 0.38); font-family: ui-sans-serif, system-ui, sans-serif;">NULLXES</p>

<h1 style="margin: 0 0 12px; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; color: #f4f4f6; font-family: ui-sans-serif, system-ui, sans-serif;">VERIDIAN</h1>

<p style="margin: 0; font-size: 15px; line-height: 1.55; color: rgba(232, 232, 238, 0.78); font-family: ui-sans-serif, system-ui, sans-serif;">Sparse Mixture-of-Experts language model stack — built from scratch inside NULLXES.</p>

</div>

</div>

<div style="max-width: 820px; margin: 0 auto 18px; padding: 20px 24px; border-radius: 16px; background: rgba(10, 10, 14, 0.88); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04); color: #d4d4dc; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px; line-height: 1.6;">

<p style="margin: 0 0 14px; color: rgba(228, 228, 234, 0.9);">VERIDIAN is a decoder-only sparse Mixture-of-Experts language model stack built from scratch.<br/>The repository is structured around three hard constraints:</p>

<ul style="margin: 0; padding-left: 1.25rem; color: rgba(200, 202, 210, 0.95);">
<li style="margin-bottom: 6px;">no borrowed weights</li>
<li style="margin-bottom: 6px;">no fine-tuning of external checkpoints</li>
<li>tokenizer, model, data mixture, and training pipeline are owned inside one project</li>
</ul>

</div>

<div style="max-width: 820px; margin: 0 auto 18px; padding: 20px 24px; border-radius: 16px; background: rgba(10, 10, 14, 0.88); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04); color: #d4d4dc; font-family: ui-sans-serif, system-ui, sans-serif;">

<h2 style="margin: 0 0 14px; font-size: 1.05rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px;">Repository layout</h2>

<ul style="margin: 0; padding-left: 1.25rem; font-size: 14px; line-height: 1.65; color: rgba(200, 202, 210, 0.95);">
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">configs/model/</code> — model hyperparameters</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">configs/tokenizer/</code> — tokenizer spec and corpus mix</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">configs/data/</code> — Hugging Face dataset mixtures</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">configs/train/</code> — training recipes</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">src/veridian/modeling/</code> — transformer, RoPE, attention, and MoE blocks</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">src/veridian/tokenizer/</code> — tokenizer spec and trainer</li>
<li style="margin-bottom: 6px;"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">src/veridian/data/</code> — streaming dataset mixture utilities</li>
<li><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">src/veridian/train/</code> — pretraining and SFT entrypoints</li>
</ul>

</div>

<div style="max-width: 820px; margin: 0 auto 18px; padding: 20px 24px; border-radius: 16px; background: rgba(10, 10, 14, 0.88); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04); color: #d4d4dc; font-family: ui-sans-serif, system-ui, sans-serif;">

<h2 style="margin: 0 0 14px; font-size: 1.05rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px;">Quick start</h2>

<p style="margin: 0 0 8px; font-size: 14px; color: rgba(228, 228, 234, 0.85);">Install:</p>

<pre style="margin: 0 0 16px; padding: 14px 16px; overflow-x: auto; border-radius: 12px; background: rgba(0, 0, 0, 0.42); border: 1px solid rgba(255, 255, 255, 0.07); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); font-size: 13px; line-height: 1.5; color: #b8ecf4; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;"><code>pip install -e .</code></pre>

<p style="margin: 0 0 8px; font-size: 14px; color: rgba(228, 228, 234, 0.85);">Train tokenizer:</p>

<pre style="margin: 0 0 16px; padding: 14px 16px; overflow-x: auto; border-radius: 12px; background: rgba(0, 0, 0, 0.42); border: 1px solid rgba(255, 255, 255, 0.07); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); font-size: 13px; line-height: 1.5; color: #b8ecf4; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;"><code>veridian-train-tokenizer ^
  --tokenizer-config configs/tokenizer/bpe_128k.yaml ^
  --model-config configs/model/veridian_60b.yaml ^
  --output-dir artifacts/tokenizer</code></pre>

<p style="margin: 0 0 12px; font-size: 14px; line-height: 1.55; color: rgba(200, 202, 210, 0.92);"><code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">--model-config</code> is the source of truth for <code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">vocab_size</code> and model-side special token ids, so the tokenizer is built around the model contract instead of drifting independently.</p>

<p style="margin: 0 0 8px; font-size: 14px; color: rgba(228, 228, 234, 0.85);">Smoke test the debug model:</p>

<pre style="margin: 0 0 16px; padding: 14px 16px; overflow-x: auto; border-radius: 12px; background: rgba(0, 0, 0, 0.42); border: 1px solid rgba(255, 255, 255, 0.07); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); font-size: 12px; line-height: 1.45; color: #b8ecf4; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;"><code>python -c &quot;import torch; from veridian.config import load_model_config; from veridian.modeling import VeridianForCausalLM; cfg = load_model_config(&#39;configs/model/veridian_debug.yaml&#39;); m = VeridianForCausalLM(cfg); x = torch.randint(0, cfg.vocab_size, (2, 128)); y = m(x); print(y.logits.shape)&quot;</code></pre>

<p style="margin: 0 0 8px; font-size: 14px; color: rgba(228, 228, 234, 0.85);">Start pretraining:</p>

<pre style="margin: 0 0 0; padding: 14px 16px; overflow-x: auto; border-radius: 12px; background: rgba(0, 0, 0, 0.42); border: 1px solid rgba(255, 255, 255, 0.07); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); font-size: 13px; line-height: 1.5; color: #b8ecf4; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;"><code>veridian-pretrain ^
  --model-config configs/model/veridian_60b.yaml ^
  --data-config configs/data/pretrain_2t.yaml ^
  --train-config configs/train/pretrain_h200.yaml</code></pre>

</div>

<div style="max-width: 820px; margin: 0 auto 18px; padding: 20px 24px; border-radius: 16px; background: rgba(10, 10, 14, 0.88); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04); color: #d4d4dc; font-family: ui-sans-serif, system-ui, sans-serif;">

<h2 style="margin: 0 0 14px; font-size: 1.05rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px;">Notes</h2>

<ul style="margin: 0; padding-left: 1.25rem; font-size: 14px; line-height: 1.65; color: rgba(200, 202, 210, 0.95);">
<li style="margin-bottom: 8px;">The implementation is a PyTorch reference stack. It is suitable for local development, smoke tests, and as a clean base for a Megatron-Core production port.</li>
<li style="margin-bottom: 8px;">Attention uses <code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">torch.nn.functional.scaled_dot_product_attention</code>, which can route to Flash Attention kernels when the runtime supports them.</li>
<li>The tokenizer is trained from scratch with Hugging Face <code style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); color: #aeefff;">tokenizers</code>; no external vocabulary is reused.</li>
</ul>

</div>

<div align="center" style="max-width: 820px; margin: 28px auto 0; padding: 22px 24px; border-radius: 16px; background: linear-gradient(165deg, rgba(22, 22, 28, 0.9) 0%, rgba(6, 6, 10, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.09); box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05); font-family: ui-sans-serif, system-ui, sans-serif;">

<p style="margin: 0 0 6px; font-size: 12px; letter-spacing: 0.22em; text-transform: uppercase; color: rgba(255, 255, 255, 0.35);">NULLXES</p>

<p style="margin: 0; font-size: 14px;">
<a href="mailto:ceo@nullxes.com" style="color: #7ee8ff; text-decoration: none; font-weight: 500;">ceo@nullxes.com</a>
</p>

</div>
