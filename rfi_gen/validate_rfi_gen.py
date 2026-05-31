"""Validate rfi_gen: knee placement and obvious-to-none contrast range."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rfi_gen
from rfi_gen import get_generator, estimate_knee

rng = np.random.default_rng(7)
M, K = 32, 128

# Expected nominal knees per style for a sanity check.
expect = {"cw_tone":1, "multitone":3, "wideband":None, "pulsed":None,
          "subtle":1, "chirp":None}

print("style       inr_db  knee_truth  nominal  contrast_dB  est_knee  est_drop")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
order = ["cw_tone","multitone","wideband","pulsed","subtle","chirp"]
for ax, style in zip(axes.ravel(), order):
    kw = {}
    if style == "multitone": kw = {"n_tones":3}
    for inr in (30, 18, 8, 2):
        gen = get_generator(style, inr_db=inr, **kw)
        r = gen.generate(M=M, K=K, rng=np.random.default_rng(int(inr)+hash(style)%1000))
        prof = r.profile_db()
        ek, ed = estimate_knee(prof)
        print("%-11s %6d  %10d  %7s  %10.1f  %8d  %7.1f" % (
            style, inr, r.knee_truth, str(r.nominal_knee),
            r.contrast_db, ek, ed))
        ax.plot(np.arange(1, M+1), prof, marker=".", ms=3,
                label="INR %d dB (knee@%d)" % (inr, r.knee_truth))
    # mark the truth knee from the strongest case
    ax.axvline(r.knee_truth + 0.5, color="k", ls=":", lw=0.8)
    ax.set_title(style)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (dB)")
    ax.legend(fontsize=6)
fig.suptitle("rfi_gen: eigenvalue profiles by style and INR "
             "(obvious knee -> none, left to right / top to bottom)",
             fontsize=11)
fig.tight_layout()
fig.savefig("rfi_gen_validation.png", dpi=110, bbox_inches="tight")
print("\nwrote rfi_gen_validation.png")

# Assertions on the deterministic-rank styles.
cw = get_generator("cw_tone", inr_db=30).generate(M=M,K=K,rng=np.random.default_rng(1))
assert cw.knee_truth == 1, cw.knee_truth
mt = get_generator("multitone", n_tones=4, inr_db=24).generate(M=M,K=K,rng=np.random.default_rng(2))
assert mt.knee_truth == 4, mt.knee_truth
mt2 = get_generator("multitone", n_tones=2, inr_db=24).generate(M=M,K=K,rng=np.random.default_rng(3))
assert mt2.knee_truth == 2, mt2.knee_truth
# subtle should be faint, cw should be obvious -> contrast ordering
cw_lo = get_generator("cw_tone", inr_db=2).generate(M=M,K=K,rng=np.random.default_rng(4))
assert cw.contrast_db > 15, cw.contrast_db
assert cw_lo.contrast_db < cw.contrast_db
print("ASSERTIONS PASSED: knee indices and contrast ordering correct")
