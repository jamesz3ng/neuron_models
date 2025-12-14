import argparse
import statistics
import time


def _time_call(fn, *, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def bench_hh_cable(*, n_x: int, dx: float, **kwargs):
    from hh_cable import simulate_hh_cable

    L = n_x * dx

    def run():
        simulate_hh_cable(L=L, dx=dx, store_history=False, **kwargs)

    return _time_call(run, warmup=kwargs.pop("warmup"), repeats=kwargs.pop("repeats"))


def bench_hh_model(*, n_spatial: int, **kwargs):
    from hh_model import simulate_hh_model

    def run():
        simulate_hh_model(n_spatial=n_spatial, store_history=False, **kwargs)

    return _time_call(run, warmup=kwargs.pop("warmup"), repeats=kwargs.pop("repeats"))


def bench_wave_model(*, nx: int, **kwargs):
    from wave_model import simulate_wave_model

    def run():
        simulate_wave_model(nx=nx, store_history=False, **kwargs)

    return _time_call(run, warmup=kwargs.pop("warmup"), repeats=kwargs.pop("repeats"))


def main():
    parser = argparse.ArgumentParser(description="Quick timing benchmarks for models.")
    parser.add_argument(
        "model",
        choices=["hh_cable", "hh_model", "wave_model"],
        help="Which model to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)

    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)

    parser.add_argument("--n-x", type=int, default=500, dest="n_x")
    parser.add_argument("--dx", type=float, default=10.0)

    parser.add_argument("--n-spatial", type=int, default=100, dest="n_spatial")

    parser.add_argument("--nx", type=int, default=100)

    args = parser.parse_args()

    common = {"repeats": args.repeats, "warmup": args.warmup}
    if args.T is not None:
        common["T"] = args.T
    if args.dt is not None:
        common["dt"] = args.dt

    if args.model == "hh_cable":
        times = bench_hh_cable(n_x=args.n_x, dx=args.dx, **common)
        n_t = int((args.T if args.T is not None else 40.0) / (args.dt if args.dt is not None else 0.001))
        work = args.n_x * n_t
        sim_T = args.T if args.T is not None else 40.0
    elif args.model == "hh_model":
        times = bench_hh_model(n_spatial=args.n_spatial, **common)
        n_t = int((args.T if args.T is not None else 50.0) / (args.dt if args.dt is not None else 0.01))
        work = args.n_spatial * n_t
        sim_T = args.T if args.T is not None else 50.0
    else:
        times = bench_wave_model(nx=args.nx, **common)
        n_t = int((args.T if args.T is not None else 0.5) / (args.dt if args.dt is not None else (1.0 / (args.nx - 1)) / 10.0))
        work = args.nx * n_t
        sim_T = args.T if args.T is not None else 0.5

    median = statistics.median(times)
    sorted_times = sorted(times)
    p90_idx = int(0.9 * (len(sorted_times) - 1))
    p90 = sorted_times[p90_idx]
    rtf = sim_T / median if median > 0 else float("inf")
    updates_per_sec = work / median if median > 0 else float("inf")

    print(f"model={args.model}")
    print(f"repeats={args.repeats} warmup={args.warmup}")
    print(f"median_s={median:.6f} p90_s={p90:.6f}")
    print(f"rtf={rtf:.3f}")
    print(f"updates_per_sec={updates_per_sec:.3e} (approx {work} updates/run)")


if __name__ == "__main__":
    main()
