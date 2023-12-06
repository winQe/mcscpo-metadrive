import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import glob

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(
    data,
    no_std=True,
    xaxis="Epoch",
    value="AverageEpRet",
    condition="Condition1",
    smooth=1,
    **kwargs,
):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
            datum[value] = smoothed_x
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci="sd", **kwargs)

    # Calculate mean and std dev
    if not no_std:
        grouped_data = data.groupby(xaxis)[value].agg(["mean", "std"])
        mean = grouped_data["mean"]
        std = grouped_data["std"]

        # Get current plot's axes for fill_between
        ax = plt.gca()

        # Plot the std deviation as a shaded area
        ax.fill_between(
            grouped_data.index,
            mean - 1.0 * std,
            mean + 1.0 * std,
            alpha=0.3,
        )

    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc="best").set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    # xscale_diff = np.max(np.asarray(data[xaxis])) - np.min(np.asarray(data[xaxis]))
    # if xscale_diff > 5e3:
    #     # Just some formatting niceness: x-axis scale in scientific notation if difference is large
    #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # plt.tight_layout(pad=0.5)


def get_datasets(exp_type, scpo_exp_name, mcscpo_exp_name, rows=0, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    mcscpo_datasets = []
    scpo_datasets = []
    mcscpo_base_dir = "/home/avt/random_projects/guard/safe_rl_lib/StateWise_Constrained_Policy_Optimization/logs/"

    task_type = exp_type + "_Multi333_noconti"
    # Construct the search pattern
    mcscpo_pattern = f"{mcscpo_base_dir}{task_type}_{mcscpo_exp_name}_*"

    mcscpo_logdir = []
    # Use glob.glob to find all directories that match the pattern
    mcscpo_directories = [d for d in glob.glob(mcscpo_pattern) if os.path.isdir(d)]
    for directory in mcscpo_directories:
        # Use next to get the first batch of subdirectories from os.walk
        subdirs = next(os.walk(directory))[1]
        # Construct the full path for subdirectories and append to logdir
        mcscpo_logdir.extend([os.path.join(directory, d) for d in subdirs])
    for logdir in mcscpo_logdir:
        for root, _, files in os.walk(logdir):
            if "progress.txt" in files:
                exp_name = None
                try:
                    config_path = open(os.path.join(root, "config.json"))
                    config = json.load(config_path)
                    if "exp_name" in config:
                        exp_name = config["exp_name"]
                except:
                    print("No file named config.json")
                condition1 = condition or exp_name or "exp"
                condition2 = condition1 + "-" + str(exp_idx)
                exp_idx += 1
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1

                try:
                    exp_data = pd.read_table(os.path.join(root, "progress.txt"))
                    if rows != 0:
                        exp_data = exp_data.head(rows)
                    # if "mcscpo" in logdir:
                    #     exp_data['EpisodeCost'] = exp_data['CumulativeCost'].diff().fillna(exp_data['CumulativeCost'])
                    #     exp_data['EpisodeCost'] = exp_data['EpisodeCost']/30

                except:
                    print("Could not read from %s" % os.path.join(root, "progress.txt"))
                    continue

                exp_data["EpMaxCost"] = exp_data.filter(regex="^EpMaxCost", axis=1).max(
                    axis=1
                )

                reward_performance = "EpRet" if "EpRet" in exp_data else "AverageEpRet"
                cost_performance = "EpCost" if "EpCost" in exp_data else "AverageEpCost"
                cost_rate_performance = (
                    "AverageTestCostRate"
                    if "AverageTestCostRate" in exp_data
                    else "CostRate"
                )
                max_cost = "EpMaxCost"
                exp_data.insert(len(exp_data.columns), "Unit", unit)
                exp_data.insert(len(exp_data.columns), "Condition1", condition1)
                exp_data.insert(len(exp_data.columns), "Condition2", condition2)
                exp_data.insert(
                    len(exp_data.columns),
                    "Reward_Performance",
                    exp_data[reward_performance],
                )
                # if exp_data[cost_performance]:
                if cost_performance in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Cost_Performance",
                        exp_data[cost_performance],
                    )
                if cost_rate_performance in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Cost_Rate_Performance",
                        exp_data[cost_rate_performance],
                    )
                if max_cost in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Maximum_Statewise_Cost",
                        exp_data[max_cost],
                    )
                mcscpo_datasets.append(exp_data)

    scpo_base_dir = "/home/avt/random_projects/guard/safe_rl_lib/old_scpo/logs/"
    task_type = exp_type + "_Multi333_noconti"
    # Construct the search pattern
    scpo_pattern = f"{scpo_base_dir}{task_type}_{scpo_exp_name}_*"

    scpo_logdir = []
    # Use glob.glob to find all directories that match the pattern
    scpo_directories = [d for d in glob.glob(scpo_pattern) if os.path.isdir(d)]
    for directory in scpo_directories:
        # Use next to get the first batch of subdirectories from os.walk
        subdirs = next(os.walk(directory))[1]
        # Construct the full path for subdirectories and append to logdir
        scpo_logdir.extend([os.path.join(directory, d) for d in subdirs])
    for logdir in scpo_logdir:
        for root, _, files in os.walk(logdir):
            if "progress.txt" in files:
                exp_name = None
                try:
                    config_path = open(os.path.join(root, "config.json"))
                    config = json.load(config_path)
                    if "exp_name" in config:
                        exp_name = config["exp_name"]
                except:
                    print("No file named config.json")
                condition1 = condition or exp_name or "exp"
                condition2 = condition1 + "-" + str(exp_idx)
                exp_idx += 1
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1

                try:
                    exp_data = pd.read_table(os.path.join(root, "progress.txt"))
                    if rows != 0:
                        exp_data = exp_data.head(rows)
                    # if "scpo" in logdir:
                    #     exp_data['EpisodeCost'] = exp_data['CumulativeCost'].diff().fillna(exp_data['CumulativeCost'])
                    #     exp_data['EpisodeCost'] = exp_data['EpisodeCost']/30

                except:
                    print("Could not read from %s" % os.path.join(root, "progress.txt"))
                    continue

                exp_data["EpMaxCost"] = exp_data.filter(regex="^EpMaxCost", axis=1).max(
                    axis=1
                )

                reward_performance = "EpRet" if "EpRet" in exp_data else "AverageEpRet"
                cost_performance = "EpCost" if "EpCost" in exp_data else "AverageEpCost"
                cost_rate_performance = (
                    "AverageTestCostRate"
                    if "AverageTestCostRate" in exp_data
                    else "CostRate"
                )
                max_cost = "EpMaxCost"
                exp_data.insert(len(exp_data.columns), "Unit", unit)
                exp_data.insert(len(exp_data.columns), "Condition1", condition1)
                exp_data.insert(len(exp_data.columns), "Condition2", condition2)
                exp_data.insert(
                    len(exp_data.columns),
                    "Reward_Performance",
                    exp_data[reward_performance],
                )
                # if exp_data[cost_performance]:
                if cost_performance in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Cost_Performance",
                        exp_data[cost_performance],
                    )
                if cost_rate_performance in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Cost_Rate_Performance",
                        exp_data[cost_rate_performance],
                    )
                if max_cost in exp_data:
                    exp_data.insert(
                        len(exp_data.columns),
                        "Maximum_Statewise_Cost",
                        exp_data[max_cost],
                    )
                scpo_datasets.append(exp_data)
    breakpoint()
    return [mcscpo_datasets, scpo_datasets]


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print("Plotting from...\n" + "=" * DIV_LINE_WIDTH + "\n")
    for logdir in logdirs:
        print(logdir)
    print("\n" + "=" * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (
        len(legend) == len(logdirs)
    ), "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(
    exp_type,
    scpo_exp_name,
    mcscpo_exp_name,
    legend=None,
    xaxis=None,
    values=None,
    count=False,
    font_scale=1.5,
    smooth=1,
    select=None,
    exclude=None,
    estimator="mean",
    results_dir=None,
    title="reward",
    reward_flag=True,
    cost_flag=False,
    max_cost_flag=False,
    rows=0,
):
    # create a separate folder for each plot
    # results_dir = osp.join(results_dir, title)
    data = get_datasets(exp_type, scpo_exp_name, mcscpo_exp_name, rows)
    # values = values if isinstance(values, list) else [values]
    values = []
    if reward_flag:
        values.append("Reward_Performance")
    if cost_flag:
        values.append("Cost_Performance")
        values.append("Cost_Rate_Performance")

    if max_cost_flag:
        values.append("Maximum_Statewise_Cost")

    condition = "Condition2" if count else "Condition1"
    estimator = getattr(
        np, estimator
    )  # choose what to show on main curve: mean? max? min?
    no_std = False
    if len(data[0]) == 1 and len(data[1]) == 1:
        no_std = True
    group1 = pd.concat(data[0], ignore_index=True)
    group1[condition] = "Multi-constrained SCPO"

    group2 = pd.concat(data[1], ignore_index=True)
    group2[condition] = "Vanilla SCPO"
    # Concatenate the two groups into a single DataFrame
    combined_data = pd.concat([group1, group2], ignore_index=True)

    for value in values:
        subdir = title + "/"
        plt.figure()
        try:
            plot_data(
                combined_data,
                no_std,
                xaxis=xaxis,
                value=value,
                condition=condition,
                smooth=smooth,
                estimator=estimator,
            )
        except:
            print(f"this key {value} is not in the data")
            break
        # make direction for save figure
        final_dir = osp.join(results_dir, subdir)
        existence = os.path.exists(final_dir)
        if not existence:
            os.makedirs(final_dir)
        plt.show()
        plt.savefig(final_dir + value, dpi=400, bbox_inches="tight")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("logdir", nargs="*")
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--scpo_exp_name", type=str)
    parser.add_argument("--mcscpo_exp_name", type=str)

    parser.add_argument(
        "--results_dir", default="./results/", help="plot results dir (default: ./)"
    )
    parser.add_argument(
        "--title", default="reward", help="the title for the saved plot"
    )
    parser.add_argument("--legend", "-l", nargs="*")
    parser.add_argument("--xaxis", "-x", default="TotalEnvInteracts")
    parser.add_argument("--value", "-y", default="Performance", nargs="*")
    parser.add_argument("--reward", action="store_true")
    parser.add_argument("--cost", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--smooth", "-s", type=int, default=1)
    parser.add_argument("--select", nargs="*")
    parser.add_argument("--exclude", nargs="*")
    parser.add_argument("--est", default="mean")
    parser.add_argument("--max_cost", action="store_true")
    parser.add_argument("--rows", type=int, default=0)
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    reward_flag = True
    cost_flag = True
    max_cost_flag = True if args.max_cost else False

    make_plots(
        args.task_type,
        args.scpo_exp_name,
        args.mcscpo_exp_name,
        args.legend,
        args.xaxis,
        args.value,
        args.count,
        smooth=args.smooth,
        select=args.select,
        exclude=args.exclude,
        estimator=args.est,
        results_dir=args.results_dir,
        title=args.title,
        reward_flag=reward_flag,
        cost_flag=cost_flag,
        max_cost_flag=max_cost_flag,
        rows=args.rows,
    )


if __name__ == "__main__":
    main()
