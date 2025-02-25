import itertools
import pandas as pd
import typing as tp

# Task 1: Create a CART

def calc_mean(configs):
    return sum(c['performance'] for c in configs) / len(configs)

def calc_sqerr(configs, mean):
    return sum((c['performance'] - mean) ** 2 for c in configs)

def leaf_node(name, mean):
    return {
        "name": name,
        "mean": mean,
        "split_by_feature": None,
        "error_of_split": None,
        "successor_left": None,
        "successor_right": None
    }

def best_split(configs, features):
    f_best = None
    l_best = None
    r_best = None
    err_best = None

    for f in sorted(features):
        l_configs = [c for c in configs if c[f] == 1]
        r_configs = [c for c in configs if c[f] == 0]
        
        if not l_configs or not r_configs:
            continue

        l_mean = calc_mean(l_configs)
        r_mean = calc_mean(r_configs)
        err = calc_sqerr(l_configs, l_mean) + calc_sqerr(r_configs, r_mean)

        if err_best is None or err < err_best:
            err_best = err
            f_best = f
            l_best = l_configs
            r_best = r_configs
    
    result_err = err_best if err_best is not None else 0

    return f_best, l_best, r_best, result_err

def build_cart(configs, features, name="X"):
        if not configs:
            return None
            
        mean = calc_mean(configs)
        
        initial_err = calc_sqerr(configs, mean)
        if initial_err < 10:
            return leaf_node(name, mean)
            
        f_best, l_configs, r_configs, err = best_split(configs, features)
        if not f_best:
            return leaf_node(name, mean)
        
        cart = {
            "name": name,
            "mean": mean,
            "split_by_feature": f_best,
            "error_of_split": err,
            "successor_left": build_cart(l_configs, features, name+"L"),
            "successor_right": build_cart(r_configs, features, name+"R")
        }
        
        return cart


def get_cart(sample_set_csv: str) -> tp.Dict[str, tp.Union[str, float, tp.Dict]]:
    """
    Calculate the CART based on a set of measurements

    @param sample_set_csv: Path to a CSV file to use as a basis for the CART
    @return: The CART in a format as specified in the "General Information" section in the Assignment sheet
    """
    # The sample_set_csv is a file path to a csv data, this can be imported into a dataframe
    df = pd.read_csv(sample_set_csv)

    configs = df.to_dict('records')
    features = [c for c in df.columns if c not in ['Id', 'performance']]

    return build_cart(configs, features)


# Task 2: Create a Performance Influence Model

def base_perf(df):
    return df.iloc[0]['performance']

def get_features(df):
    return [c for c in df.columns 
            if c not in ['Id', 'performance']]

def single_infl(df, features, base_perf):
   infls = {}
   
   for f1 in features:
       for _, row in df.iterrows():
           enabled = 0
           for f2 in features:
               if row[f2] == 1:
                   enabled += 1
           
           if row[f1] == 1 and enabled == 1:
               perf_f = row['performance']
               infl_f = perf_f - base_perf
               infls[f1] = infl_f
               break
               
   return infls

def double_infl(df, features, base_perf, single_infl):
   dbl_infls = {}

   for f1, f2 in itertools.combinations(sorted(features), 2):
       for _, row in df.iterrows():
           enabled = 0
           for f in features:
               if row[f] == 1:
                   enabled += 1
           
           if row[f1] == 1 and row[f2] == 1 and enabled == 2:
               perf_true = row['performance']
               
               exp = base_perf
               exp += single_infl.get(f1, 0)
               exp += single_infl.get(f2, 0)
               
               interact = perf_true - exp
               if abs(interact) > 0:
                   dbl_infls[f"{f1}*{f2}"] = interact
               break
               
   return dbl_infls

def triple_infl(df, features, base_perf, sng_infl, dbl_infl):
   trp_infls = {}
   
   for f1, f2, f3 in itertools.combinations(sorted(features), 3):
       for _, row in df.iterrows():
           enabled = 0
           for f in features:
               if row[f] == 1:
                   enabled += 1
           
           if row[f1] == 1 and row[f2] == 1 and row[f3] == 1 and enabled == 3:
               perf_true = row['performance']
               exp = base_perf
               for f in [f1, f2, f3]:
                   exp += sng_infl.get(f, 0)

               for pair in itertools.combinations([f1, f2, f3], 2):
                   key = "*".join(sorted(pair))
                   exp += dbl_infl.get(key, 0)
               
               interact = perf_true - exp
               if abs(interact) > 0:
                   trp_infls[f"{f1}*{f2}*{f3}"] = interact
               break
               
   return trp_infls


def get_pim(sample_set_csv: str) -> tp.Dict[str, float]:
    """
    Calculate the Performance Influence Model (PIM) of a system based on a set of performance measurements

    @param sample_set_csv: Path to a CSV file to use as a basis for the PIM
    @return: PIM for the given set of measurements
    """
    # The sample_set_csv is a file path to a csv data, this can be imported into a dataframe
    df = pd.read_csv(sample_set_csv)
    pim = {"": base_perf(df)}
    
    features = get_features(df)
    single = single_infl(df, features, pim[""])
    
    double = double_infl(df, features, pim[""], single)
    triple = triple_infl(df, features, pim[""], single, double)
    
    pim.update(single)
    pim.update(double)
    pim.update(triple)
    
    return pim


# Task 3a: Predicted performance




def get_performance(cart: tp.Dict[str, tp.Union[str, float, tp.Dict]], configuration: tp.Set[str]) -> float:
    """
    Calculate the performance of a given configuration for a specific CART

    @param cart: CART to use for the performance prediction
    @param configuration: The configuration for which to predict the performance
    @return: The expected performance for a specific configuration, according to a given CART
    """
    node = cart
    
    while node['split_by_feature'] is not None:
        f = node['split_by_feature']

        if f in configuration:
            node = node['successor_left']
        else:
            node = node['successor_right']
    
    return node['mean']


# Task 3b: Calculate the error rate



def get_error_rate(cart: tp.Dict[str, tp.Union[str, float, tp.Dict]]
                   , sample_set_csv: str) -> float:
    """
    Calculate the error rate of a CART

    @param cart: A CART of a software system
    @param sample_set_csv: Path to a CSV file with a set of performance measurements.
    @return: The error rate of the CART with respect to the actual performance measurements
    """
    # The sample_set_csv is a file path to a csv data, this can be imported into a dataframe
    df = pd.read_csv(sample_set_csv)
    err_sum = 0.0
    n = len(df)
    for _, row in df.iterrows():
        config = set()
        for c in df.columns:
            if c not in ['Id', 'performance'] and row[c] == 1:
                config.add(c)
        
        pred = get_performance(cart, config)
        actual = row['performance']
        err = abs(pred - actual)
        err_sum += err
    
    return err_sum / n


# Task 3c: Generate optimal configuration

def calc_cost(pim, config):
    cost = pim[""]
    
    for f in config:
        if f in pim:
            cost += pim[f]
    for i in range(2, 4):
        for f in itertools.combinations(sorted(config), i):
            key = "*".join(f)
            if key in pim:
                cost += pim[key]

    return cost

def get_all_feature(model):
    f = {model["name"]}
    if "children" in model:
        for child in model["children"]:
            f.update(get_all_feature(child))
    return f

def get_valid_configs(features):
    configs = []
    for i in range(len(features) + 1):
        for c in itertools.combinations(features, i):
            cset = set(c)
            configs.append(cset)
    return configs

def is_valid(config, model):
    if model["name"] in config:
        if "children" in model:
            if model["groupType"] == "And":
                return check_and(config, model)
            elif model["groupType"] == "Or":
                return check_or(config, model)
            elif model["groupType"] == "Xor":
                return check_xor(config, model)
    return True

def check_and(config, model):
   if model["name"] not in config:
       return True

   for child in model["children"]:
       if "featureType" in child and child["featureType"] == "Mand":
           if child["name"] not in config:
               return False
   return True

def check_or(config, model):
   if model["name"] not in config:
       return True
       
   has_child = False
   for child in model["children"]:
       if child["name"] in config:
           has_child = True
           break
   return has_child

def check_xor(config, model):
   if model["name"] not in config:
       return True
       
   select = 0
   for child in model["children"]:
       if child["name"] in config:
           select += 1
   return select == 1

def get_optimal_configuration(pim: tp.Dict[str, float]
                              , feature_model : tp.Dict[str, tp.Union[str,tp.Dict, None]]
                              , partial_configuration: tp.Set[str])\
        -> tp.Tuple[tp.Set[str], float]:
    """
    Find the optimal full configuration according to a specific PIM while adhering to a specific partial configuration.
    Assumes that the performance is the cost of a system. Thus, the optimal configuration is the cheapest one.

    @param pim: The PIM to use as a basis for performance prediction
    @param feature_model: The feature model of the system to find the best configuration for.
    @param partial_configuration: Partial configuration to find the optimal configuration for.
    @return: A tuple which contains the set of selected features as its first entry and the corresponding performance value as its second entry
    """
    all_f = get_all_feature(feature_model)
    p_configs = get_valid_configs(all_f)
    
    valid_configs = []
    for config in p_configs:
        if is_valid(config, feature_model) and partial_configuration.issubset(config):
            valid_configs.append(config)
    
    first_config = valid_configs[0]
    lowest_cost = calc_cost(pim, first_config)
    best_config = first_config

    for config in valid_configs[1:]:
        cost = calc_cost(pim, config)
        if cost < lowest_cost:
            lowest_cost = cost
            best_config = config

    return best_config, lowest_cost
