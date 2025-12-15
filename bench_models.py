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


def _summarize_times(times: list[float]) -> tuple[float, float]:
    median = statistics.median(times)
    sorted_times = sorted(times)
    p90_idx = int(0.9 * (len(sorted_times) - 1))
    p90 = sorted_times[p90_idx]
    return median, p90


def _print_summary(*, model: str, times: list[float], work: int, sim_T_s: float, dt_s: float):
    median, p90 = _summarize_times(times)
    rtf = sim_T_s / median if median > 0 else float("inf")
    updates_per_sec = work / median if median > 0 else float("inf")

    print(f"model={model}")
    print(f"median_s={median:.6f} p90_s={p90:.6f}")
    print(f"T_s={sim_T_s:g} dt_s={dt_s:g} rtf={rtf:.3f}")
    print(f"updates_per_sec={updates_per_sec:.3e} (approx {work} updates/run)")


def main():
    parser = argparse.ArgumentParser(description="Quick timing benchmarks for models.")
    parser.add_argument(
        "model",
        choices=["hh_cable", "hh_model", "wave_model", "common"],
        help="Which model to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)

    parser.add_argument("--T", type=float, default=None, help="Simulated time (seconds).")
    parser.add_argument("--dt", type=float, default=None, help="Time step (seconds).")

    parser.add_argument(
        "--N",
        type=int,
        default=500,
        help="Common spatial points used by `model=common`.",
    )

    parser.add_argument("--n-x", type=int, default=500, dest="n_x")
    parser.add_argument("--dx", type=float, default=10.0)

    parser.add_argument("--n-spatial", type=int, default=100, dest="n_spatial")

    parser.add_argument("--nx", type=int, default=100)

    args = parser.parse_args()

    print(f"repeats={args.repeats} warmup={args.warmup}")

    if args.model == "common":
        T_s = 1.0 if args.T is None else args.T

        from hh_cable import _default_params as _hh_cable_defaults
        from hh_cable import simulate_hh_cable
        from hh_model import _default_params as _hh_model_defaults
        from hh_model import simulate_hh_model
        from wave_model import _default_params as _wave_defaults
        from wave_model import simulate_wave_model

        # hh_model: use default dt unless forced
        hh_model_dt_s = args.dt if args.dt is not None else _hh_model_defaults()["dt_s"]
        meta = simulate_hh_model(
            n_spatial=args.N, T_s=T_s, dt_s=hh_model_dt_s, store_history=False
        )
        times = _time_call(
            lambda: simulate_hh_model(
                n_spatial=args.N, T_s=T_s, dt_s=hh_model_dt_s, store_history=False
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _print_summary(
            model="hh_model",
            times=times,
            work=meta["n_spatial"] * meta["n_t"],
            sim_T_s=meta["T_s"],
            dt_s=meta["dt_s"],
        )
        print()

        # hh_cable: choose dt from alpha <= 0.45 unless forced
        hh_cable = _hh_cable_defaults()
        if args.dt is not None:
            hh_cable_dt_s = args.dt
        else:
            alpha_target = 0.45
            dt_ms_max = (
                alpha_target * hh_cable["tau_ms"] * (args.dx**2) / (hh_cable["lam"]**2)
            )
            hh_cable_dt_s = dt_ms_max / 1e3

        L = args.N * args.dx
        meta = simulate_hh_cable(
            L=L, dx=args.dx, T_s=T_s, dt_s=hh_cable_dt_s, store_history=False
        )
        times = _time_call(
            lambda: simulate_hh_cable(
                L=L, dx=args.dx, T_s=T_s, dt_s=hh_cable_dt_s, store_history=False
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _print_summary(
            model="hh_cable",
            times=times,
            work=meta["n_x"] * meta["n_t"],
            sim_T_s=meta["T_s"],
            dt_s=meta["dt_s"],
        )
        print()

        # wave_model: choose dt from CFL < 1 unless forced
        wave = _wave_defaults()
        if args.dt is not None:
            wave_dt_s = args.dt
        else:
            if args.N < 2:
                raise SystemExit("--N must be >= 2 for wave_model")
            dx_wave = wave["L"] / (args.N - 1)
            wave_dt_s = 0.9 * dx_wave / wave["c"]

        meta = simulate_wave_model(nx=args.N, T_s=T_s, dt_s=wave_dt_s, store_history=False)
        times = _time_call(
            lambda: simulate_wave_model(
                nx=args.N, T_s=T_s, dt_s=wave_dt_s, store_history=False
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _print_summary(
            model="wave_model",
            times=times,
            work=meta["nx"] * meta["n_t"],
            sim_T_s=meta["T_s"],
            dt_s=meta["dt_s"],
        )
        return

    if args.model == "hh_cable":
        from hh_cable import simulate_hh_cable

        L = args.n_x * args.dx

        def run_once():
            return simulate_hh_cable(
                L=L,
                dx=args.dx,
                T_s=args.T,
                dt_s=args.dt,
                store_history=False,
            )

        meta = run_once()
        times = _time_call(run_once, warmup=args.warmup, repeats=args.repeats)
        work = meta["n_x"] * meta["n_t"]
        sim_T_s = meta["T_s"]
        dt_s = meta["dt_s"]
    elif args.model == "hh_model":
        from hh_model import simulate_hh_model

        def run_once():
            return simulate_hh_model(
                n_spatial=args.n_spatial,
                T_s=args.T,
                dt_s=args.dt,
                store_history=False,
            )

        meta = run_once()
        times = _time_call(run_once, warmup=args.warmup, repeats=args.repeats)
        work = meta["n_spatial"] * meta["n_t"]
        sim_T_s = meta["T_s"]
        dt_s = meta["dt_s"]
    else:
        from wave_model import simulate_wave_model

        def run_once():
            return simulate_wave_model(
                nx=args.nx,
                T_s=args.T,
                dt_s=args.dt,
                store_history=False,
            )

        meta = run_once()
        times = _time_call(run_once, warmup=args.warmup, repeats=args.repeats)
        work = meta["nx"] * meta["n_t"]
        sim_T_s = meta["T_s"]
        dt_s = meta["dt_s"]

    _print_summary(
        model=args.model,
        times=times,
        work=work,
        sim_T_s=sim_T_s,
        dt_s=dt_s,
    )


if __name__ == "__main__":
    main()
