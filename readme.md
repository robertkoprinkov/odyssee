An implementation of the [EGMDO](https://hal.science/hal-02904829/file/Article_EGMDO.pdf) algorithm.

See `img/` for results. See `test_sample.py` and `test_sellar.py` for use.

Dependencies:
- `numpy`, `scipy` and `matplotlib`
- [`smt`](https://github.com/SMTorg/smt) for Gaussian Process regression
- [`chaospy`](https://github.com/jonathf/chaospy) for the Uncertainty Quantification and Polynomial Chaos Expansion
- [`tqdm`](https://github.com/tqdm/tqdm) is not strictly necessary, but is used on occasion for convenience
