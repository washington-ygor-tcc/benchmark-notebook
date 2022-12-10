import click
import numpy as np
import matplotlib.pyplot as plt

from benchmark.application import app, types, config
from benchmark.application.types import BenchmarkParams


flatten = lambda lst: [item for sublist in lst for item in sublist]
timings = lambda results: [
    [t.end - t.start for t in result.response_list]
    for result in results
]

def plot_total_bloxplot_timings(
    api_results,
    msg_results,
    x_values,
    x_legend,
    y_legend,
    title,
    file_prefix,
    show=False,
    save=True,
):
    api_elapsed_times = timings(api_results)
    msg_elapsed_times = timings(msg_results)

    fig, axs = plt.subplots(nrows=2, figsize=(12, 16))

    axs[0].boxplot(api_elapsed_times, labels=x_values)
    axs[0].set_title("API RPC", fontsize=10)
    axs[0].plot(
        axs[0].get_xticks(),
        [result.elapsed_time for result in api_results],
        "o--r",
        label="Tempo total",
    )
    axs[0].legend()
    axs[0].grid(True)

    axs[1].boxplot(msg_elapsed_times, labels=x_values)
    axs[1].set_title("Mensageria", fontsize=10)
    axs[1].plot(
        axs[1].get_xticks(),
        [result.elapsed_time for result in msg_results],
        "o--b",
        label="Tempo total",
    )
    axs[1].legend()
    axs[1].grid(True)

    fig.supxlabel(x_legend)
    fig.supylabel(y_legend)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.93, hspace=0.2, wspace=0.2)

    if save:
        plt.savefig(f"plots/{file_prefix}_test_total_bloxplot.png")

    if show:
        plt.show()
    else:
        plt.close()


def plot_total_timings(
    api_results,
    msg_results,
    x_values,
    x_legend,
    y_legend,
    title,
    file_prefix,
    show=False,
    save=True,
):
    def get_result_timings(results):
        return [result.elapsed_time for result in results]

    plt.plot(x_values, get_result_timings(api_results), ":r", label="API RPC")
    plt.plot(x_values, get_result_timings(msg_results), "--b", label="Mensageria")
    plt.legend()
    plt.grid(True)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.title(title)
    plt.yscale("log", base=10)
    plt.xscale("log", base=2)
    if save:
        plt.savefig(f"plots/{file_prefix}_test_total.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_timings(
    api_results,
    msg_results,
    x_values,
    x_legend,
    y_legend,
    title,
    file_prefix,
    show=False,
    save=True,
):
    def get_mean_request_timings(results):
        return [
            np.mean([t.end - t.start for t in result.response_list])
            if len(result.response_list) > 0
            else 0
            for result in results
        ]

    plt.plot(
        x_values, get_mean_request_timings(api_results), ":r", label="API RPC"
    )
    plt.plot(
        x_values, get_mean_request_timings(msg_results), "--b", label="Mensageria"
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.title(title)
    plt.yscale("log", base=10)
    plt.xscale("log", base=2)

    if save:
        plt.savefig(f"plots/{file_prefix}_test_mean.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_throughput(
    api_results,
    msg_results,
    x_values,
    x_legend,
    y_legend,
    title,
    file_prefix,
    show=False,
    save=True,
):
    def get_requests_per_second(results):
        return [
            len(result.response_list) / result.elapsed_time
            if result.elapsed_time > 0
            else 0
            for result in results
        ]

    plt.plot(
        x_values, get_requests_per_second(api_results), ":r", label="API RPC"
    )
    plt.plot(
        x_values, get_requests_per_second(msg_results), "--b", label="Mensageria"
    )
    plt.legend()
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.title(title)
    plt.grid(True)
    plt.yscale("log", base=10)
    plt.xscale("log", base=2)
    plt.xticks(x_values)

    if save:
        plt.savefig(f"plots/{file_prefix}_test_throughput.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_std_timings(
    api_results,
    msg_results,
    x_values,
    x_legend,
    y_legend,
    title,
    file_prefix,
    show=False,
    save=True,
):
    def get_standard_deviation(results):
        return [np.std(np.asarray(timing)) for timing in timings(results)]

    plt.plot(
        x_values, get_standard_deviation(api_results), ":r", label="API RPC"
    )
    plt.plot(
        x_values, get_standard_deviation(msg_results), "--b", label="Mensageria"
    )

    plt.legend()
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.title(title)
    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(x_values)

    if save:
        plt.savefig(f"plots/{file_prefix}_test_std.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_wrapper(plot_func, x_legend, y_legend, title):
    def wrapper(
        api_results,
        msg_results,
        x_values,
        file_prefix,
        show=False,
        save=True,
    ):
        return plot_func(
            api_results,
            msg_results,
            x_values,
            x_legend,
            y_legend,
            title,
            file_prefix,
            show,
            save,
        )

    return wrapper


def get_bechmark_params(benchmark_type, parameter, range_values, fixed_requests_number):
    default_params = dict(
        benchmark_type=benchmark_type,
        complexity_factor=1,
        memory_overhead=1,
        requests_number=fixed_requests_number,
        batch_size=fixed_requests_number,
        batch_progress=False,
        total_progress=True,
    )

    def update(parameter, value):
        default_params[parameter] = value
        return default_params

    return [
        BenchmarkParams(**update(parameter, value)) for value in range_values
    ]


def run(params, env):
    return [app.run_benchmark(param, env) for param in params]


def plot_results(
    plots, api_results, msg_results, benchmark_range, parameter, show=False, save=True
):
    for plot in plots:
        plot(api_results, msg_results, benchmark_range, parameter, show=show, save=save)


ANALYSIS_TYPE = {
    "requests_number": (
        plot_wrapper(
            plot_total_timings,
            x_legend="Volume de requisições",
            y_legend="Duração (s)",
            title="Duração total por volume de requisições",
        ),
        plot_wrapper(
            plot_mean_timings,
            x_legend="Volume de requisições",
            y_legend="Duração média (s)",
            title="Duração média por volume de requisições",
        ),
        plot_wrapper(
            plot_std_timings,
            x_legend="Volume de requisições",
            y_legend="Desvio padrão",
            title="Desvio padrão por volume de requisições",
        ),
        plot_wrapper(
            plot_throughput,
            x_legend="Volume de requisições",
            y_legend="Req/s",
            title="Requisições por segundo",
        ),
        plot_wrapper(
            plot_total_bloxplot_timings,
            x_legend="Volume de requisições",
            y_legend="Duração total (s)",
            title="Duração total por volume de requisições",
        ),
    ),
    "memory_overhead": (
        plot_wrapper(
            plot_total_timings,
            x_legend="Complexidade de memória",
            y_legend="Duração total(s)",
            title="Duração total das requisições por complexidade de memória",
        ),
        plot_wrapper(
            plot_mean_timings,
            x_legend="Complexidade de memória",
            y_legend="Duração média (s)",
            title="Duração média das requisições por complexidade de memória",
        ),
        plot_wrapper(
            plot_std_timings,
            x_legend="Complexidade de memória",
            y_legend="Desvio padrão",
            title="Desvio padrão por complexidade de memória",
        ),
        plot_wrapper(
            plot_throughput,
            x_legend="Complexidade de memória",
            y_legend="Req/s",
            title="Requisições por segundo por complexidade de memória",
        ),
        plot_wrapper(
            plot_total_bloxplot_timings,
            x_legend="Complexidade de memória",
            y_legend="Duração total (s)",
            title="Duração total das requisições por complexidade de memória",
        ),
    ),
    "complexity_factor": (
        plot_wrapper(
            plot_total_timings,
            x_legend="Complexidade computacional",
            y_legend="Duração total(s)",
            title="Duração total das requisições por complexidade computacional",
        ),
        plot_wrapper(
            plot_mean_timings,
            x_legend="Complexidade computacional",
            y_legend="Duração média (s)",
            title="Duração média das requisições por complexidade computacional",
        ),
        plot_wrapper(
            plot_std_timings,
            x_legend="Complexidade computacional",
            y_legend="Desvio padrão",
            title="Desvio padrão por complexidade computacional",
        ),
        plot_wrapper(
            plot_throughput,
            x_legend="Complexidade computacional",
            y_legend="Req/s",
            title="Requisições por segundo por complexidade computacional",
        ),
        plot_wrapper(
            plot_total_bloxplot_timings,
            x_legend="Complexidade computacional",
            y_legend="Duração total (s)",
            title="Requisições por segundo por complexidade computacional",
        ),
    ),
}


def main(benchmark_range_callback, env, fixed_requests_number, show_plots=False, save_plots=True):
    results = []
    ttests = []
    for parameter, plots in ANALYSIS_TYPE.items():
        print("parameter", parameter)
        api_params = get_bechmark_params(
            types.BenchmarkTypes.API,
            parameter,
            benchmark_range_callback(parameter),
            fixed_requests_number
        )
        msg_params = get_bechmark_params(
            types.BenchmarkTypes.MSG,
            parameter,
            benchmark_range_callback(parameter),
            fixed_requests_number
        )
        api_results = run(api_params, env)
        msg_results = run(msg_params, env)

        results.append(
            [
                plots,
                api_results,
                msg_results,
                benchmark_range_callback(parameter),
                parameter,
                show_plots,
                save_plots
            ]
        )

    for result in results:
        plot_results(*result)



@click.command()
@click.option(
    "--env",
    "-e",
    "env",
    type=click.Choice(
        list(config.Env),
        case_sensitive=False,
    ),
    default=config.Env.DOCKER,
)
@click.option("--memory_overhead", "-mr", type=int, default=11)
@click.option("--complexity_factor", "-cf", type=int, default=11)
@click.option("--requests_number", "-rn", type=int, default=15)
@click.option("--fixed_requests_number", "-fr", type=int, default=4096)
@click.option("--save", is_flag=True)
@click.option("--show", is_flag=True)
def cli(env, memory_overhead, complexity_factor, requests_number, fixed_requests_number, save, show):

    def range_callback(parameter):
        match parameter:
            case "memory_overhead":
                return [2**i for i in range(memory_overhead)]
            case "complexity_factor":
                return [2**i for i in range(complexity_factor)]
            case "requests_number":
                return [2**i for i in range(requests_number)]

    main(range_callback, env, fixed_requests_number, show_plots=show, save_plots=save)

if __name__ == "__main__":
    cli()
