import pathlib
import yaml

#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/MLP_MNIST_UnifAvg_04_many_tests/")
#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/LeNet_CIFAR_UnifAvg_02_many_tests/")
#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/BigMLP_MNIST_UnifAvg_05_many_tests/")
#global_path = pathlib.Path("../../_Experiments/GroupedNewton_Results/VGG_CIFAR_UnifAvg_02_many_tests/")

global_path = pathlib.Path("/gpfswork/rech/tza/uki35ex/_Experiments/GroupedNewton_Results/LeNet_CIFAR_NS_01_new_tests/")
#global_path = pathlib.Path("/gpfswork/rech/tza/uki35ex/_Experiments/GroupedNewton_Results/VGG_CIFAR_NS_01_new_tests/")
#global_path = pathlib.Path("/gpfswork/rech/tza/uki35ex/_Experiments/GroupedNewton_Results/BigMLP_MNIST_iNS_01_tests/")

lst_metrics = global_path.glob("*/*/metrics/metrics.json")

loss_threshold = 2
rate_instability = 3
only_finished_expes = False

dct_expes = {}

# Gather existing experiments with "metrics" file
for fmetrics in lst_metrics:
    s = str(fmetrics)
    s = s[:s.rfind("/")]
    s = s[:s.rfind("/")]
    path = pathlib.Path(s)
    fconfig = list(path.glob(".hydra/config.yaml"))[0]

    subexp_number = s[s.rfind("/")+1:]
    exp_path = s[:s.rfind("/")]
    exp_path = pathlib.Path(exp_path + "/.submitit")
    lst_paths = list(exp_path.glob("*_" + subexp_number))
    if len(lst_paths) > 0:
        sub_path = str(lst_paths[0])
    else:
        lst_paths = list(exp_path.glob("*"))
        for p in lst_paths:
            if str(p)[-3:] == ".sh":
                continue
            else:
                sub_path = str(p)
                break

    expe_id = sub_path[sub_path.rfind("/")+1:]

    dct_expes[str(path)] = {"path": path,
            "fmetrics": fmetrics,
            "fconfig": fconfig,
            "expe_id": expe_id}

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

    if len(lst_metrics) == 0:
        continue
    elif only_finished_expes and lst_metrics[-1]["epoch"] != epochs - 1:
        continue

    # Exclude experiments with instabilities
    exclude = False
    for m1, m2 in zip(lst_metrics[:-1], lst_metrics[1:]):
        if m2["tr_nll"] >= rate_instability * m1["tr_nll"]:
            exclude = True
            break
    if exclude:
        continue

    # Store the final loss and the config
    lst_results.append({"expe_name": expe_name, 
        "expe_id": dct_expe["expe_id"],
        "epoch": lst_metrics[-1]["epoch"],
        "loss": lst_metrics[-1]["tr_nll"],
        "config": config})

lst_results = sorted(lst_results, key = lambda dct: dct["loss"])

print("#"*60)
print("#"*60)
print("#"*60)
for dct in lst_results:
    loss = dct["loss"]
    if loss <= loss_threshold:
        print("")
        print(dct["expe_name"] + "    " + dct["expe_id"])
        print(f"Epoch {dct['epoch']}")
        print(f"Loss = {loss:.3e}")
        print(dct["config"])
    else:
        break

