import pathlib
import yaml

global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/LeNet_CIFAR_UnifAvg_02_many_tests/")
#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/BigMLP_MNIST_UnifAvg_05_many_tests/")
#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/VGG_CIFAR_UnifAvg_02_many_tests/")
lst_metrics = global_path.glob("*/*/metrics/metrics.json")
dct_expes = {}

# Gather existing experiments with "metrics" file
for fmetrics in lst_metrics:
    s = str(fmetrics)
    s = s[:s.rfind("/")]
    s = s[:s.rfind("/")]
    path = pathlib.Path(s)
    fconfig = list(path.glob(".hydra/config.yaml"))[0]
    dct_expes[s] = {"path": path,
            "fmetrics": fmetrics,
            "fconfig": fconfig}

lst_results = []
for expe_name, dct_expe in dct_expes.items():
    # Get the corresponding config files
    fmetrics = dct_expe["fmetrics"]
    fconfig = dct_expe["fconfig"]
    with open(fconfig) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Exclude unfinished experiments
    epochs = config["optimizer"]["epochs"]
    with open(fmetrics) as f:
        lst_metrics = f.readlines()
    lst_metrics = [eval(l) for l in lst_metrics]

    if len(lst_metrics) == 0 or lst_metrics[-1]["epoch"] != epochs - 1:
        continue

    # Exclude experiments with instabilities
    exclude = False
    for m1, m2 in zip(lst_metrics[:-1], lst_metrics[1:]):
        if m2["tr_nll"] >= 2 * m1["tr_nll"]:
            exclude = True
            break
    if exclude:
        continue

    # Store the final loss and the config
    lst_results.append({"expe_name": expe_name, 
        "loss": lst_metrics[-1]["tr_nll"],
        "config": config})

lst_results = sorted(lst_results, key = lambda dct: dct["loss"])

for dct in lst_results:
    loss = dct["loss"]
    if loss <= .1:
        print(dct["expe_name"])
        print(f"Loss = {loss:.3e}")
        print(dct["config"])
    else:
        break

