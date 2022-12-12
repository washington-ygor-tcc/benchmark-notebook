import time
import numpy as np
import matplotlib.pyplot as plt

from benchmark.application import app, types
from benchmark.application.types import BenchmarkParams


def get_result(param):
    result = app.run_benchmark(param)

    return result


benchmark_types = [types.BenchmarkTypes.API, types.BenchmarkTypes.MSG]
mo_list = [100 * (i + 1) for i in range(20)]

bench_param_list = [
    BenchmarkParams(
        benchmark_type=bench_type,
        complexity_factor=3,
        memory_overhead=mo,
        requests_number=100,
        batch_size=2,
        total_progress=True,
    )
    for mo in mo_list
    for bench_type in benchmark_types
]

results = [get_result(param) for param in bench_param_list]

elapsed_times = [[t.end - t.start for t in result.response_list] for result in results]


def get_median(results):
    return [np.median(result) for result in results]


def get_standard_deviation(elapsed_times):
    return [np.std(elapsed_time) for elapsed_time in elapsed_times]


def plot_total_time_by_mo():
    plt.plot(
        mo_list,
        [result.elapsed_time for result in results[::2]],
        ":r",
        label="API RPC",
    )
    plt.plot(
        mo_list,
        [result.elapsed_time for result in results[1::2]],
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Fator de consumo de memória")
    plt.ylabel("Tempo total para realizar 100 requisições (s)")

    plt.grid(True)

    plt.savefig("plots/total_time_by_mo.png")
    plt.show()


def plot_median():
    plt.plot(mo_list, get_median(elapsed_times[::2]), ":r", label="API RPC")
    plt.plot(mo_list, get_median(elapsed_times[1::2]), "--b", label="Mensageria")

    plt.legend()
    plt.xlabel("Fator de consumo de memória")
    plt.ylabel("Média tempo de resposta (s)")

    plt.grid(True)

    plt.savefig("plots/median_by_mo.png")
    plt.show()


def plot_standard_deviation():
    plt.plot(
        mo_list,
        get_standard_deviation(elapsed_times[::2]),
        ":r",
        label="API RPC",
    )
    plt.plot(
        mo_list,
        get_standard_deviation(elapsed_times[1::2]),
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Fator de consumo de memória")
    plt.ylabel("Desvio padrão tempo de resposta (s)")

    plt.grid(True)

    plt.savefig("plots/standard_deviation_by_mo.png")
    plt.show()


plot_total_time_by_mo()
plot_median()
plot_standard_deviation()
