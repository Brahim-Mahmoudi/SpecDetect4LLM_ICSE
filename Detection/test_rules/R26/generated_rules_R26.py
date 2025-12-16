import ast
import sys
import re
from typing import Optional, Set

# Noms/fonctions indiquant un prétraitement d'image (utilisé par générateurs R25-R31)

torch_imported = False
torch_used = False
deterministic_used = False
def reset_torch_flags():
    global torch_imported, torch_used, deterministic_used
    torch_imported = False
    torch_used = False
    deterministic_used = False

def track_torch_import(node):
    global torch_imported
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == 'torch':
                torch_imported = True
    elif isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith('torch'):
            torch_imported = True

def useDeterministic(node):
    global deterministic_used
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'use_deterministic_algorithms':
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is True:
            deterministic_used = True
            return True
    return False

def isRelevantTorchCall(node):
    global torch_used
    if not isinstance(node, ast.Call):
        return False
    full_path = get_full_attr_path(node.func)
    if full_path and full_path.startswith("torch."):
        torch_used = True
        return True
    return False

def get_full_attr_path(expr):
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.insert(0, expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.insert(0, expr.id)
    return '.'.join(parts)

def customCheckTorchDeterminism(ast_node):
    reset_torch_flags()
    for node in ast.walk(ast_node):
        track_torch_import(node)
        isRelevantTorchCall(node)
        useDeterministic(node)
    #log(f"torch_imported={torch_imported}, torch_used={torch_used}, deterministic_used={deterministic_used}")
    return torch_imported and torch_used and not deterministic_used

def report(message):
    print('REPORT:', message, flush=True)

def log(message):
    print('LOG:', message, flush=True)

def report_with_line(message, node):
    line = getattr(node, 'lineno', '?')
    report(message.format(lineno=line))

def add_parent_info(node, parent=None):
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        add_parent_info(child, node)
    if parent is None:
       init_train_lines(node)

def gather_scale_sensitive_ops(ast_node):
    scale_sensitive_ops = {
        'PCA', 'SVC', 'SGDClassifier', 'SGDRegressor', 'MLPClassifier',
        'ElasticNet', 'Lasso', 'Ridge', 'KMeans', 'KNeighborsClassifier',
        'LogisticRegression'
    }
    ops = {}
    for stmt in ast.walk(ast_node):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                value = stmt.value
                if isinstance(value, ast.Call):
                    func = value.func
                    if isinstance(func, ast.Name) and func.id in scale_sensitive_ops:
                        ops[var_name] = func.id
                    elif isinstance(func, ast.Attribute) and func.attr in scale_sensitive_ops:
                        ops[var_name] = func.attr
    return ops

def isScaleSensitiveFit(call, variable_ops):
    if not isinstance(call, ast.Call):
        return False
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'fit'):
        return False
    callee = call.func.value
    if isinstance(callee, ast.Name):
        return callee.id in variable_ops
    return False

def isChainedIndexingBase(node):
    """
    Détecte les patterns chain indexing :
      - df[...][...]
      - df[...][...].method()
    Exclut .values[...] et .to_numpy()[...], .str[...] et .apply[...]
    """
    if not isinstance(node, ast.Subscript):
        return False

    value = node.value

    # Exclure accès .values, .to_numpy, .str, .apply, .dt, .map, etc.
    skip_attrs = {'values', 'to_numpy', 'str', 'dt', 'apply', 'map', 'squeeze'}

    # Traverse la chaîne d'attributs/méthodes entre le Subscript actuel et le précédent Subscript
    while isinstance(value, (ast.Attribute, ast.Call)):
        # Si on rencontre .str, .apply, etc. => on NE FLAG PAS
        if isinstance(value, ast.Attribute) and value.attr in skip_attrs:
            return False
        # Si c'est un appel de méthode, on check la fonction puis continue sur .value
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "func"):
            value = value.func
        else:
            break

    # Si on atteint un Subscript (df[...][...])
    if isinstance(value, ast.Subscript):
        return True

    return False
def get_scope_dataframe_vars(node):
    current = node
    while current is not None and not isinstance(current, (ast.FunctionDef, ast.Module)):
        current = getattr(current, 'parent', None)

    local_vars = set()
    series_vars = set()

    dataframe_creators = {
        'DataFrame', 'from_dict', 'from_records',
        'read_csv', 'read_json', 'read_excel',
        'read_sql', 'read_parquet', 'read_feather',
        'read_table', 'concat', 'merge'
    }
    series_creators = {'Series'}

    # 1. Détection explicite des DataFrames et Series Pandas
    for stmt in ast.walk(current):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id
            val = stmt.value

            if isinstance(val, ast.Call):
                func = val.func
                if isinstance(func, ast.Attribute):
                    # Ex: pd.read_csv(...) ou pd.Series(...)
                    if isinstance(func.value, ast.Name) and func.value.id == 'pd':
                        if func.attr in dataframe_creators:
                            local_vars.add(var_name)
                        elif func.attr in series_creators:
                            series_vars.add(var_name)
                    # Ex: pd.DataFrame.from_dict(...) ou pd.Series.from_array(...)
                    elif isinstance(func.value, ast.Attribute):
                        if func.value.attr == 'DataFrame' and getattr(func.value.value, 'id', '') == 'pd':
                            if func.attr in dataframe_creators:
                                local_vars.add(var_name)
                        if func.value.attr == 'Series' and getattr(func.value.value, 'id', '') == 'pd':
                            if func.attr in series_creators:
                                series_vars.add(var_name)
                elif isinstance(func, ast.Name):
                    if func.id == 'DataFrame':
                        local_vars.add(var_name)
                    elif func.id == 'Series':
                        series_vars.add(var_name)

    # 2. Propagation du statut DataFrame/Series via alias, accès colonne ou méthode pandas
    for stmt in ast.walk(current):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id
            val = stmt.value

            # df2 = df.method(...) ou df2 = df[...]
            if isinstance(val, (ast.Call, ast.Subscript)):
                base = get_base_name(val)
                # Cas DataFrame
                if base in local_vars:
                    local_vars.add(var_name)
                # Cas Series : accès à une colonne d'un DataFrame connu
                if base in local_vars and isinstance(val, ast.Subscript):
                    series_vars.add(var_name)
                # Cas alias de Series
                if base in series_vars:
                    series_vars.add(var_name)

    # 3. On EXCLUT explicitement les dict, defaultdict, list, set...
    for stmt in ast.walk(current):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id
            val = stmt.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
                if val.func.id in {'dict', 'defaultdict', 'list', 'set'}:
                    if var_name in local_vars:
                        local_vars.remove(var_name)
                    if var_name in series_vars:
                        series_vars.remove(var_name)

    # 4. Propagation simple des alias (alias = df ou alias = series)
    for stmt in ast.walk(current):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id
            val = stmt.value
            if isinstance(val, ast.Name):
                if val.id in local_vars:
                    local_vars.add(var_name)
                if val.id in series_vars:
                    series_vars.add(var_name)

    # Retourne l'union
    return local_vars | series_vars
def get_base_name(expr):
    while isinstance(expr, (ast.Subscript, ast.Attribute, ast.Call)):
        if hasattr(expr, "value"):
            expr = expr.value
        elif hasattr(expr, "func"):
            expr = expr.func
        else:
            break
    if isinstance(expr, ast.Name):
        return expr.id
    return None
def isDataFrameVariable(var, node):
    if isinstance(var, str):
        base = var
    else:
        base = get_base_name(var)
    if base is None:
        return False
    scope_vars = get_scope_dataframe_vars(node)
    return base in scope_vars

def gather_scaled_vars(ast_node):
    scaled_vars = set()
    known_scalers = {
        'StandardScaler','MinMaxScaler','RobustScaler','Normalizer',
        'MaxAbsScaler','PowerTransformer','QuantileTransformer'
    }

    scaler_map = {}
    for stmt in ast.walk(ast_node):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                value = stmt.value
                if isinstance(value, ast.Call):
                    func = value.func
                    # ex: scaler = StandardScaler()
                    if isinstance(func, ast.Name) and func.id in known_scalers:
                        scaler_map[var_name] = func.id
                    elif isinstance(func, ast.Attribute) and func.attr in known_scalers:
                        scaler_map[var_name] = func.attr

    for stmt in ast.walk(ast_node):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                if isinstance(stmt.value, ast.Call):
                    if isinstance(stmt.value.func, ast.Attribute):
                        # ex: scaler.fit_transform(X) ou StandardScaler().fit_transform(X)
                        if stmt.value.func.attr == 'fit_transform':
                            base = stmt.value.func.value  # ex. ast.Name(id='scaler') ou ast.Call(...)
                            if isinstance(base, ast.Name):
                                # case: scaler.fit_transform(X)
                                if base.id in scaler_map:
                                    scaled_vars.add(target.id)
                                else:
                                    base_func = get_base_name(base)
                                    if base_func in known_scalers:
                                        scaled_vars.add(target.id)
                            else:
                                # case: StandardScaler().fit_transform(X)
                                base_func = get_base_name(base)
                                if base_func in known_scalers:
                                    scaled_vars.add(target.id)
    return scaled_vars

def call_uses_scaled_data(call_node, scaled_vars):
    if not isinstance(call_node, ast.Call):
        return False
    for arg in call_node.args:
        if isinstance(arg, ast.Name) and arg.id in scaled_vars:
            return True
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Name) and kw.value.id in scaled_vars:
            return True
    return False

def hasPrecedingScaler(call, scaled_vars=None):
    if scaled_vars:
        if call_uses_scaled_data(call, scaled_vars):
            return True
    scalers = {
        'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer',
        'MaxAbsScaler', 'PowerTransformer', 'QuantileTransformer'
    }
    current = call
    while current:
        current = getattr(current, 'parent', None)
        if isinstance(current, ast.Assign):
            value = current.value
            if isinstance(value, ast.Call):
                if isinstance(value.func, ast.Name) and value.func.id in scalers:
                    return True
                elif isinstance(value.func, ast.Attribute) and value.func.attr in scalers:
                    return True
    return False

def parse_pipeline_steps(node):
    funcs = []
    if isinstance(node, ast.Call):
        base_name = None
        if isinstance(node.func, ast.Name):
            base_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            base_name = node.func.attr
        if base_name:
            funcs.append(base_name)
        for arg in node.args:
            funcs.extend(parse_pipeline_steps(arg))
        for kw in node.keywords:
            funcs.extend(parse_pipeline_steps(kw.value))
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            funcs.extend(parse_pipeline_steps(elt))
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            funcs.extend(parse_pipeline_steps(value))
    elif isinstance(node, ast.keyword):
        funcs.extend(parse_pipeline_steps(node.value))
    return funcs

def isPartOfValidatedPipeline(call):
    scalers = {
        'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer',
        'MaxAbsScaler', 'PowerTransformer', 'QuantileTransformer'
    }
    sensitive_ops = {
        'PCA', 'SVC', 'SGDClassifier', 'SGDRegressor', 'MLPClassifier',
        'ElasticNet', 'Lasso', 'Ridge', 'KMeans', 'KNeighborsClassifier',
        'LogisticRegression'
    }
    parent = call
    while parent:
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
            if parent.func.id in {'Pipeline', 'make_pipeline'}:
                all_funcs = []
                for arg in parent.args:
                    all_funcs.extend(parse_pipeline_steps(arg))
                for kw in parent.keywords:
                    all_funcs.extend(parse_pipeline_steps(kw.value))
                has_scaler = any(func in scalers for func in all_funcs)
                has_sensitive_op = any(func in sensitive_ops for func in all_funcs)
                return (has_scaler and has_sensitive_op)
        parent = getattr(parent, 'parent', None)
    return False

def isDataFrameColumnAssignment(node):
    if not isinstance(node, ast.Assign):
        return False
    if len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Subscript):
        return False
    base_name = get_base_name(target.value)
    if not isDataFrameVariable(base_name, target.value):
        return False
    return True

def isAssignedLiteral(node, val):
    if not isinstance(node, ast.Assign):
        return False
    assigned_value = node.value
    if not isinstance(assigned_value, ast.Constant):
        return False
    return assigned_value.value == val

# Fonctions ajoutées (absentes du premier header initial)
def isDataFrameMerge(node):
    return (isinstance(node, ast.Call) and
            hasattr(node, 'func') and
            getattr(node.func, 'attr', '') == 'merge' and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id in get_scope_dataframe_vars(node))

def singleParam(node):
    return (len(node.args) + len(node.keywords)) == 1

def isApiMethod(node):
    """
    Détecte les appels d’API qui doivent :
      • soit être exécutés avec « inplace=True » (DataFrame Pandas),
      • soit ré‑affecter leur résultat (NumPy ou Pandas).

    Renvoie True si l’appel entre dans l’un de ces deux cas.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False

    attr = node.func.attr
    base = get_base_name(node.func.value)

    # ----------  NumPy -------------------------------------------------
    numpy_methods_requiring_assignment = {'clip', 'sort', 'argsort'}
    if base == 'np' and attr in numpy_methods_requiring_assignment:
        return True

    # ----------  DataFrame Pandas -------------------------------------
    api_methods = {
        'drop', 'dropna', 'sort_values', 'replace',
        'clip', 'sort', 'argsort',
        'detach', 'cpu', 'clone', 'numpy',
        'transform', 'fit_transform',
        'traverse', 'strip', 'rstrip', 'lstrip', 'lower', 'upper'
    }
    if attr in api_methods and base in get_scope_dataframe_vars(node):
        return True

    return False


def hasInplaceTrue(node):
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == 'inplace' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False

def isResultUsed(node):
    parent = getattr(node, 'parent', None)
    if isinstance(parent, ast.Expr):
        return False
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return True
    if isinstance(parent, ast.Return):
        return True
    if isinstance(parent, ast.Call):
        return True
    if isinstance(parent, (ast.Attribute, ast.Subscript)):
        return isResultUsed(parent)
    if isinstance(parent, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return True
    return False

def isBinOp(node):
    return isinstance(node, ast.BinOp)

def isTfTile(node):
    return (
    hasattr(node, 'func') and
    getattr(node.func, 'attr', '') == 'tile' and
    isinstance(node.func.value, ast.Name) and
    node.func.value.id == 'tf')

def isSubscript(node):
    return isinstance(node, ast.Subscript)

def extract_metric_name(node):
    if isinstance(node, ast.Call):
        if hasattr(node.func, 'attr') and node.func.attr == 'make_scorer':
            for arg in node.args:
                candidate = extract_metric_name(arg)
                if candidate is not None:
                    return candidate
            return None
        else:
            return extract_metric_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None

def isMetricCall(node):
    return isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'attr')

def isThresholdDependent(node):
    metric_name = extract_metric_name(node)
    return metric_name in {
        'f1_score', 'precision_score', 'recall_score', 'accuracy_score',
        'specificity', 'balanced_accuracy', 'jaccard_score',
        'confusion_matrix', 'brier_score_loss'
    }

def isThresholdIndependent(node):
    metric_name = extract_metric_name(node)
    return metric_name in {
        'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',
        'r2_score', 'max_error', 'mean_absolute_percentage_error',
        'roc_auc_score', 'roc_curve', 'pr_auc_score',
        'precision_recall_curve', 'log_loss', 'hinge_loss', 'auc'
    }

def isCompare(node):
    return isinstance(node, ast.Compare)

def hasNpNanComparator(node):
    if not isinstance(node, ast.Compare):
        return False
    for comparator in node.comparators:
        if isinstance(comparator, ast.Attribute):
            if (isinstance(comparator.value, ast.Name) and
                comparator.value.id == 'np' and
                comparator.attr == 'nan'):
                return True
    return False

def isNumpyVariable(node):
    return isinstance(node, ast.Name) and node.id == 'np'

def isValuesAccess(node):
    """
    Retourne True si le nœud est un accès d'attribut sur '.values'
    """
    return isinstance(node, ast.Attribute) and node.attr == 'values'

def isPandasReadCall(node):
    pandas_read_methods = {'read_csv', 'read_json', 'read_sql', 'read_table', 'read_excel', 'read_parquet'}
    if isinstance(node, ast.Call):
        # Cas pd.read_csv(...) ou pandas.read_csv(...)
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                and node.func.value.id in {'pd', 'pandas'}
                and node.func.attr in pandas_read_methods):
                return True
        # Cas import direct : read_csv(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in pandas_read_methods:
                return True
    return False
def hasKeyword(node, keyword_name):
   if isinstance(node, ast.Call):
       return any(kw.arg == keyword_name for kw in node.keywords)
   return False
def isDotCall(node):
   if not isinstance(node, ast.Call):
       return False
   if not isinstance(node.func, ast.Attribute):
       return False
   if node.func.attr != "dot":
       return False
   if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "np"):
       return False
   return True
def isMatrix2D(node):
   if not isinstance(node, ast.Call):
       return False
   if len(node.args) != 2:
       return False
   return True
def isForLoop(node):
   return isinstance(node, ast.For)

def isFunctionDef(node):
   return isinstance(node, ast.FunctionDef)
def usesIterrows(node):
   return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'iterrows')
def usesItertuples(node):
   return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'itertuples')
def usesPythonLoopOnTensorFlow(loop_node):
    if not isinstance(loop_node, ast.For):
        return False
    
    # 1. Collecter toutes les variables assignées à un tenseur TensorFlow
    tf_vars = set()
    root = loop_node
    while hasattr(root, 'parent') and root.parent:
        root = root.parent

    for stmt in ast.walk(root):
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute):
                    base_name = get_base_name(func.value)
                    if base_name == 'tf':
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                tf_vars.add(target.id)

    # 2. Vérifier si la boucle fait des opérations sur ces variables TensorFlow
    for stmt in ast.walk(loop_node):
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add):
            if isinstance(stmt.value, ast.Subscript):
                var_name = get_base_name(stmt.value)
                if var_name in tf_vars:
                    return True
    return False
def isTensorFlowTensor(node):
   if isinstance(node, ast.Name):
       var_name = node.id.lower()
       return 'tf' in var_name or 'tensor' in var_name
   return False
def isRandomCall(call):
    if not isinstance(call, ast.Call):
        return False
    if isinstance(call.func, ast.Name) and call.func.id == 'DataLoader':
        for kw in call.keywords:
           if kw.arg == 'shuffle' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
               return True
    if not isinstance(call.func, ast.Attribute):
        return False

    rand_funcs = [
        ('np', 'random', {'random', 'rand', 'randn', 'randint', 'normal',
                          'uniform', 'sample', 'choice', 'shuffle', 'permutation'}),
        ('torch', None, {'rand', 'randn', 'randint', 'random'}),
        ('tf', 'random', {'normal', 'uniform', 'shuffle'}),
        ('random', None, {'randint', 'choice', 'shuffle', 'random', 'uniform'}),
        ('sklearn', 'utils', {'shuffle'}),
        ('sklearn', 'model_selection', {'train_test_split'}),
        ('sklearn', 'metrics', {'make_scorer'}),
        ('df', None, {'randomSplit'}),
    ]

    for lib, submodule, funcs in rand_funcs:
        if submodule:
            if (isinstance(call.func.value, ast.Attribute) and
                isinstance(call.func.value.value, ast.Name) and
                call.func.value.value.id == lib and
                call.func.value.attr == submodule and
                call.func.attr in funcs):
                return True
        else:
            if (isinstance(call.func.value, ast.Name) and
                call.func.value.id == lib and
                call.func.attr in funcs):
                return True

    return False

def seedSet(call):
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Attribute):
        return False

    # np.random.seed(...)
    if (isinstance(call.func.value, ast.Attribute) and
        isinstance(call.func.value.value, ast.Name) and
        call.func.value.value.id == 'np' and
        call.func.value.attr == 'random' and
        call.func.attr == 'seed'):
        return True

    # tf.random.set_seed(...)
    if (isinstance(call.func.value, ast.Attribute) and
        isinstance(call.func.value.value, ast.Name) and
        call.func.value.value.id == 'tf' and
        call.func.value.attr == 'random' and
        call.func.attr == 'set_seed'):
        return True

    # torch.manual_seed(...)
    if (isinstance(call.func.value, ast.Name) and
        call.func.value.id == 'torch' and
        call.func.attr == 'manual_seed'):
        return True

    # random.seed(...)
    if (isinstance(call.func.value, ast.Name) and
        call.func.value.id == 'random' and
        call.func.attr == 'seed'):
        return True

    return False
    # np.random.seed(...)
    if (isinstance(call.func.value, ast.Attribute) and
        isinstance(call.func.value.value, ast.Name) and
        call.func.value.value.id == 'np' and
        call.func.value.attr == 'random' and
        call.func.attr == 'seed'):
        return True

    # tf.random.set_seed(...)
    if (isinstance(call.func.value, ast.Attribute) and
        isinstance(call.func.value.value, ast.Name) and
        call.func.value.value.id == 'tf' and
        call.func.value.attr == 'random' and
        call.func.attr == 'set_seed'):
        return True

    # torch.manual_seed(...)
    if (isinstance(call.func.value, ast.Name) and
        call.func.value.id == 'torch' and
        call.func.attr == 'manual_seed'):
        return True

    # random.seed(...)
    if (isinstance(call.func.value, ast.Name) and
        call.func.value.id == 'random' and
        call.func.attr == 'seed'):
        return True

    return False
def hasRandomState(call):
    if not isinstance(call, ast.Call):
        return False
    for kw in call.keywords:
        if kw.arg == 'random_state':
            if isinstance(kw.value, ast.Constant):
                return kw.value.value is not None
            return True  
    return False
def global_seed_set(ast_node, lib):
    seeds = set()
    for stmt in ast.walk(ast_node):
        if isinstance(stmt, ast.Call):
            if isinstance(stmt.func, ast.Attribute):
                # np.random.seed(...)
                if (isinstance(stmt.func.value, ast.Attribute) and
                    isinstance(stmt.func.value.value, ast.Name) and
                    stmt.func.value.value.id == 'np' and
                    stmt.func.value.attr == 'random' and
                    stmt.func.attr == 'seed'):
                    seeds.add('numpy')
                # torch.manual_seed(...)
                elif (isinstance(stmt.func.value, ast.Name) and
                      stmt.func.value.id == 'torch' and
                      stmt.func.attr == 'manual_seed'):
                    seeds.add('torch')
                # tf.random.set_seed(...)
                elif (isinstance(stmt.func.value, ast.Attribute) and
                      isinstance(stmt.func.value.value, ast.Name) and
                      stmt.func.value.value.id == 'tf' and
                      stmt.func.value.attr == 'random' and
                      stmt.func.attr == 'set_seed'):
                    seeds.add('tensorflow')
                # random.seed(...)
                elif (isinstance(stmt.func.value, ast.Name) and
                      stmt.func.value.id == 'random' and
                      stmt.func.attr == 'seed'):
                    seeds.add('random')
    return lib in seeds

def get_random_lib(call):
    if is_random_numpy_call(call):
        return 'numpy'
    if is_random_torch_call(call):
        return 'torch'
    if is_dataloader_with_shuffle(call):
        return 'torch'
    if is_random_tf_call(call):
        return 'tensorflow'
    if is_random_python_call(call):
        return 'random'
    return None

def is_random_numpy_call(stmt):
    if not isinstance(stmt, ast.Call):
        return False
    if not isinstance(stmt.func, ast.Attribute):
        return False
    return (isinstance(stmt.func.value, ast.Attribute) and
            isinstance(stmt.func.value.value, ast.Name) and
            stmt.func.value.value.id == 'np' and
            stmt.func.value.attr == 'random' and
            stmt.func.attr in {
                'random', 'rand', 'randn', 'randint', 'normal',
                'uniform', 'sample', 'choice', 'shuffle', 'permutation'
            })
def is_random_python_call(stmt):
    if not isinstance(stmt, ast.Call):
        return False
    if not isinstance(stmt.func, ast.Attribute):
        return False
    return (isinstance(stmt.func.value, ast.Name) and
            stmt.func.value.id == 'random' and
            stmt.func.attr in {'randint', 'choice', 'shuffle', 'random', 'uniform'})
def is_random_torch_call(stmt):
    if is_dataloader_with_shuffle(stmt):
       return True
    if not isinstance(stmt, ast.Call):
        return False
    if not isinstance(stmt.func, ast.Attribute):
        return False
    return (isinstance(stmt.func.value, ast.Name) and
            stmt.func.value.id == 'torch' and
            stmt.func.attr in {'rand', 'randn', 'randint', 'random'})
def is_random_tf_call(stmt):
    if not isinstance(stmt, ast.Call):
        return False
    if not isinstance(stmt.func, ast.Attribute):
        return False
    return (isinstance(stmt.func.value, ast.Attribute) and
            isinstance(stmt.func.value.value, ast.Name) and
            stmt.func.value.value.id == 'tf' and
            stmt.func.value.attr == 'random')
def isSklearnRandomAlgo(call):
    if not isinstance(call, ast.Call):
        return False
    if isinstance(call.func, ast.Name):
        return call.func.id in {
            'RandomForestClassifier', 'RandomForestRegressor',
            'KMeans', 'train_test_split', 'RandomizedSearchCV',
            'StratifiedKFold', 'ShuffleSplit', 'GridSearchCV','CatBoostregressor','SGD','Linear'
        }
    return False
def is_dataloader_with_shuffle(stmt):
    if not isinstance(stmt, ast.Call):
        return False
    if isinstance(stmt.func, ast.Name) and stmt.func.id == 'DataLoader':
        for kw in stmt.keywords:
            if kw.arg == 'shuffle' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    if isinstance(stmt.func, ast.Attribute) and stmt.func.attr == 'DataLoader':
        for kw in stmt.keywords:
            if kw.arg == 'shuffle' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False

def hasConstantAndConcatIntersection(block):
    import ast

    TF_INIT_FUNCS = {'Variable', 'ones', 'zeros', 'random_normal', 'random_uniform', 'fill'}
    MODIFICATION_FUNCS = {'concat', 'stack'}

    tf_constant_vars = set()
    ignore_vars = set()
    tensorarray_write_vars = set()

    # Vérifie si le block est dans une boucle (nécessaire pour ce smell)
    def is_inside_loop(node):
        while node:
            if isinstance(node, (ast.For, ast.While)):
                return True
            node = getattr(node, 'parent', None)
        return False

    # 1. Collecte les tf.constant assignés
    for node in ast.walk(block):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # assurer la référence vers parent
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute)
                and hasattr(call.func.value, 'id')
                and call.func.value.id == 'tf'):
                if call.func.attr == 'constant':
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        tf_constant_vars.add(node.targets[0].id)
                elif call.func.attr in TF_INIT_FUNCS:
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        ignore_vars.add(node.targets[0].id)

    # 2. Ignore si uniquement utilisé dans TensorArray.write
    for node in ast.walk(block):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "write" and isinstance(node.func.value, ast.Name):
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in tf_constant_vars:
                        tensorarray_write_vars.add(arg.id)

    # 3. Détecte les modifications suspectes dans le bloc, UNIQUEMENT DANS UNE BOUCLE
    for node in ast.walk(block):
        # Assurer parent
        for child in ast.iter_child_nodes(node):
            child.parent = node

        # tf.concat / tf.stack
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute)
                and hasattr(call.func.value, 'id')
                and call.func.value.id == 'tf'
                and call.func.attr in MODIFICATION_FUNCS
                and is_inside_loop(node)):  # Seulement dans boucle
                involved_vars = set()
                for arg in call.args:
                    if isinstance(arg, ast.List):
                        involved_vars |= set(elt.id for elt in arg.elts if isinstance(elt, ast.Name))
                    elif isinstance(arg, ast.Name):
                        involved_vars.add(arg.id)
                smell_vars = (involved_vars & tf_constant_vars) - ignore_vars - tensorarray_write_vars
                if smell_vars:
                    return True

        # Opération arithmétique (+, *, etc.) sur tf.constant dans une boucle seulement
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp) and is_inside_loop(node):
            for side in [node.value.left, node.value.right]:
                if isinstance(side, ast.Name):
                    if side.id in tf_constant_vars and side.id not in ignore_vars and side.id not in tensorarray_write_vars:
                        return True

    # Aucune intersection détectée
    return False
def isMLMethodCall(call):
    if not isinstance(call, ast.Call):
        return False
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr
    else:
        return False
    hyperparameter_functions = {
        # Scikit-Learn
        'KMeans', 'DBSCAN', 'AgglomerativeClustering',
        'RandomForestClassifier','RandomForestRegressor', 'GradientBoostingClassifier', 'AdaBoostClassifier',
        'LogisticRegression', 'LinearRegression', 'Lasso', 'Ridge',
        'SVC', 'SVR', 'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'MLPClassifier', 'MLPRegressor',
        # PyTorch Optimizers
        'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'RMSprop', 'Net',
        # TensorFlow Optimizers (et éventuellement des layers si pertinent)
        'Adam', 'Ftrl', 'Nadam', 'Adamax', 'Dense', 'Conv2D', 'LSTM',
        # XGBoost
        'XGBClassifier', 'XGBRegressor',
        # LightGBM
        'LGBMClassifier', 'LGBMRegressor'
        # LightGBM
        'Sequential'
    }
    return func_name in hyperparameter_functions
def hasExplicitHyperparameters(call):
    return len(call.keywords) > 0

def isLog(call):
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Attribute):
        return False
    if hasattr(call.func.value, 'id') and call.func.value.id == 'tf' and call.func.attr == 'log':
        return True
    return False

def hasMask(call):
    if not isinstance(call, ast.Call):
        return False
    if not isLog(call):
        return False
    if len(call.args) == 0:
        return False
    arg = call.args[0]
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        if hasattr(arg.func.value, 'id') and arg.func.value.id == 'tf' and arg.func.attr == 'clip_by_value':
            return True
    return False

def isForwardCall(call):
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != 'forward':
        return False

    # (NOUVEAU) Vérifie si on est dans la méthode __call__ d'une classe : autorisé => pas de smell
    node = call
    while hasattr(node, 'parent') and node.parent is not None:
        node = node.parent
        if isinstance(node, ast.FunctionDef) and node.name == '__call__':
            return False  # Autorisé dans __call__, donc pas un smell

    # Base de l'appel (self, self.block, model, etc)
    base = call.func.value
    while isinstance(base, ast.Attribute):
        base = base.value

    if isinstance(base, ast.Name):
        base_id = base.id
        # Accepte self
        if base_id == 'self':
            return True
        # Accepte toute variable qui ressemble à un modèle dans le code
        root = call
        while hasattr(root, 'parent') and root.parent is not None:
            root = root.parent
        for node in ast.walk(root):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == base_id:
                    val = node.value
                    if isinstance(val, ast.Call):
                        func = val.func
                        if isinstance(func, ast.Attribute):
                            if (
                                (isinstance(func.value, ast.Name) and func.value.id in {'torch', 'nn'})
                                or (isinstance(func.value, ast.Attribute) and func.value.attr == 'nn')
                            ):
                                return True
    # Optionnel : flag toute utilisation de .forward dans un fichier PyTorch
    root = call
    while hasattr(root, 'parent') and root.parent is not None:
        root = root.parent
    torch_present = False
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'torch':
                    torch_present = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('torch'):
                torch_present = True
    if torch_present:
        return True
    return False

def isRelevantLibraryCall(node):
    if not isinstance(node, ast.Call):
       return False
    base = get_base_name(node.func)
    return base in ['torch', 'numpy', 'random', 'transformers']

def hasManualSeed(node):
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr in {'manual_seed', 'set_seed', 'seed'}:
        return bool(node.args and isinstance(node.args[0], ast.Constant))
    return False

def isEvalCall(node):
    return (isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'eval')

def isTrainCall(node):
    return (isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'train')

def isOptimizerStep(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and getattr(node.func.value, 'id', None) == 'optimizer'
        and node.func.attr == 'step'
    )

train_lines = []

def init_train_lines(ast_node):
    global train_lines
    train_lines = []
    for stmt in ast.walk(ast_node):
        if (isTrainCall(stmt) or isOptimizerStep(stmt)) and hasattr(stmt, 'lineno'):
            train_lines.append(stmt.lineno)
    train_lines.sort()

def hasLaterTrainCall(node):
    if not hasattr(node, 'lineno'):
        return False

    node_line = node.lineno
    for tline in train_lines:
        if tline > node_line:
            return True
    return False

def isLossBackward(node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr == 'backward'

def isZeroGradCall(node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr == 'zero_grad'

def isClearGradCall(node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr == 'clear_grad'

def isPaddleEnvironment(root_node):
    """
    Détecte si l'AST contient un import de paddle,
    donc si l'on est dans un environnement Paddle.
    """
    import ast
    for stmt in ast.walk(root_node):
        # Case: import paddle
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == 'paddle':
                    return True
        # Case: from paddle import <X>
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module and 'paddle' in stmt.module:
                return True
    return False

def isInsideNoGrad(node):
    """
    Retourne True si le noeud se trouve dans un bloc `with torch.no_grad():`
    """
    current = node
    while getattr(current, 'parent', None) is not None:
        current = current.parent
        if isinstance(current, ast.With):
            for item in current.items:
                if isinstance(item.context_expr, ast.Call):
                    called = item.context_expr.func
                    if (isinstance(called, ast.Attribute)
                        and isinstance(called.value, ast.Name)
                        and called.value.id == 'torch'
                        and called.attr == 'no_grad'):
                        return True
    return False

def hasPrecedingZeroGrad(call):
    """
    Vérifie si backward() est précédé d'un appel zero_grad(), ou si on est
    dans un bloc no_grad(), ou si on est en environnement Paddle et qu'un
    clear_grad() (paddle) est présent *après* la ligne du backward.
    """
    import ast
    if isInsideNoGrad(call):
        return True

    if not hasattr(call, 'lineno'):
        return False
    node_line = call.lineno

    root_node = call
    while getattr(root_node, 'parent', None) is not None:
        root_node = root_node.parent

    if not isPaddleEnvironment(root_node):
        for stmt in ast.walk(root_node):
            if isZeroGradCall(stmt) and hasattr(stmt, 'lineno'):
                if stmt.lineno < node_line:
                    return True
        return False
    else:

        for stmt in ast.walk(root_node):
            if isZeroGradCall(stmt) and hasattr(stmt, 'lineno'):
                if stmt.lineno < node_line:
                    return True

        for stmt in ast.walk(root_node):
            if isClearGradCall(stmt) and hasattr(stmt, 'lineno'):
                if stmt.lineno > node_line:
                    return True

        return False

tracked_tensors = set()

pytorch_tensors = set()

def isPytorchTensorDefinition(node):
    """Register variables created via torch tensor creation functions."""
    global pytorch_tensors
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute):
            # Catch torch.tensor(...) and torch.Tensor(...)
            if isinstance(call.func.value, ast.Name) and call.func.value.id == 'torch':
                if call.func.attr in {'tensor', 'Tensor'}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            pytorch_tensors.add(target.id)
                            return True
    return False

def isPytorchTensorUsage(node):
    # Limite la détection aux variables connues comme tenseurs torch
    if not isinstance(node, ast.Call): return False
    if not isinstance(node.func, ast.Attribute): return False
    ops = {'matmul', 'add', 'mul', 'sub', 'div', 'mm'}
    if node.func.attr not in ops: return False
    if isinstance(node.func.value, ast.Name):
        var_name = node.func.value.id
        return var_name in pytorch_tensors
    return False



def isModelCreation(node):
    """
    Détecte la création d'un modèle ou d'une couche (Keras/PyTorch) même sans assignation explicite,
    y compris passé comme argument à une fonction/méthode (ex: append(Model(...))).
    """
    if not isinstance(node, ast.Call):
        return False
    # Liste élargie : tous les objets potentiellement coûteux en mémoire
    model_layer_names = {
        "Sequential", "Model",
        "Conv1D", "Conv2D", "Conv3D", "Dense", "LSTM", "GRU", "RNN",
        "LeakyReLU", "ReLU", "MaxPooling2D", "Flatten", "Dropout"
    }
    # Cas appel direct (ex: Sequential(...), Model(...))
    if isinstance(node.func, ast.Name) and node.func.id in model_layer_names:
        return True
    # Cas appel qualifié (ex: tf.keras.Sequential(...), tf.keras.Model(...))
    if isinstance(node.func, ast.Attribute) and node.func.attr in model_layer_names:
        return True
    return False


def isMemoryFreeCall(node):
    """Return True if the node represents a memory-freeing API call."""
    if isinstance(node, ast.Call):
        # Handle method calls like tensor.detach() or backend.clear_session()
        if isinstance(node.func, ast.Attribute):
            # PyTorch: tensor.detach()
            if node.func.attr == 'detach':
                return True
            # TensorFlow/Keras: clear_session() called as an attribute
            if node.func.attr == 'clear_session':
                # Check if this call is inside a loop (for memory freeing in loops)
                current = node
                in_loop = False
                while hasattr(current, "parent") and current.parent is not None:
                    if isinstance(current.parent, ast.For):
                        in_loop = True
                        break
                    current = current.parent
                return in_loop  # True if in a loop, False otherwise
        # Handle function calls like clear_session() imported directly
        elif isinstance(node.func, ast.Name):
            if node.func.id == 'clear_session':
                # Similar loop check for standalone clear_session()
                current = node
                in_loop = False
                while hasattr(current, "parent") and current.parent is not None:
                    if isinstance(current.parent, ast.For):
                        in_loop = True
                        break
                    current = current.parent
                return in_loop
    # Handle explicit deletions: `del var`
    if isinstance(node, ast.Delete):
        return True
    # Handle assigning a variable to None as a form of manual cleanup
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if (isinstance(target, ast.Name) 
                and isinstance(node.value, ast.Constant) 
                and node.value.value is None):
            return True
    return False


def isFitTransform(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'fit_transform'
    )

def pipelineUsed(node):
    current = node
    while current:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Name) and current.func.id in {'make_pipeline', 'Pipeline'}:
                return True
            if isinstance(current.func, ast.Attribute) and current.func.attr in {'fit', 'predict'}:
                base = get_base_name(current.func.value)
                if base in {'pipeline', 'clf'}:
                    return True
        current = getattr(current, 'parent', None)
    return False

def usedBeforeTrainTestSplit(node):
    if not hasattr(node, 'lineno'):
        return False
    fit_line = node.lineno
    root = node
    while getattr(root, 'parent', None):
        root = root.parent
    for sub in ast.walk(root):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name) and sub.func.id == 'train_test_split':
                if hasattr(sub, 'lineno') and sub.lineno > fit_line:
                    return True
    return False

def pipelineUsedGlobally(ast_node):
    for node in ast.walk(ast_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {'Pipeline', 'make_pipeline'}:
                return True
    return False

def isModelFitPresent(ast_node):
    for node in ast.walk(ast_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                return True
    return False

def isFitCall(node):
   return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fit"

def reportFitLine(msg, node):   report_with_line(msg, node)

def report_line(message, node):
   report_with_line(message, node)

def hasEarlyStoppingCallback(call):
   # return True if 'callbacks' exists and contains 'EarlyStopping'
   if not (isinstance(call, ast.Call) and hasKeyword(call, "callbacks")):
       return False
   for kw in call.keywords:
       if kw.arg == "callbacks" and "EarlyStopping" in ast.unparse(kw.value):
           return True
   return False

def isLLMCall(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    # Reconstitue le chemin de l'appel
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)
    #print("DBG_R25.isLLMCall:", call_tuple)
    if call_tuple and call_tuple[-1] in {"with_options"}:
         return False

    # 1) Appels LLM explicites (OpenAI/Anthropic/…)
    EXACTS = {
        ("openai", "Completion", "create"),
        ("openai", "ChatCompletion", "create"),
        ("OpenAI", "completions", "create"),
        ("OpenAI", "chat", "completions", "create"),
        ("anthropic", "completions", "create"),
        ("anthropic", "messages", "create"),
        ("Anthropic", "completions", "create"),
        ("vertexai", "generative_models", "generate_content"),
        ("GenerativeModel", "generate_content"),
        ("cohere", "generate"),
        ("Client", "generate"),
    }
    if call_tuple in EXACTS:
        return True

    # 2) Suffixes client.*.create (inclut responses.create)
    SUFFIXES = {
        ("completions", "create"),
        ("messages", "create"),
        ("responses", "create"),
        ("chat", "completions", "create"),
    }
    for suf in SUFFIXES:
        if len(call_tuple) >= len(suf) and call_tuple[-len(suf):] == suf:
            return True

    # 3) LangChain : considérer les *constructeurs de modèles* comme LLM calls
    #    (ils portent temperature). On évite 'OpenAI()' non qualifié pour ne pas
    #    confondre avec le client OpenAI SDK.
    LANGCHAIN_LLM_CTORS = {
        "OpenAI",             # langchain.llms.OpenAI
        "ChatOpenAI",         # langchain_openai
        "ChatAnthropic",      # langchain_anthropic
        "ChatCohere",         # langchain_cohere
        "ChatVertexAI",       # langchain_google
        "HuggingFacePipeline" # langchain_huggingface
    }

    # a) Nom simple: ChatOpenAI(...)
    if isinstance(node.func, ast.Name) and node.func.id in LANGCHAIN_LLM_CTORS:
        return True

    # b) Chemin qualifié: langchain_openai.ChatOpenAI(...) ou x.y.ChatOpenAI(...)
    if isinstance(node.func, ast.Attribute) and node.func.attr in LANGCHAIN_LLM_CTORS:
        return True

    # 4) transformers.pipeline("text-generation", ...)
    if isinstance(f, ast.Name) and f.id == "pipeline":
        if node.args and isinstance(node.args[0], ast.Constant) and str(node.args[0].value) == "text-generation":
            return True

    # 5) variable = pipeline('text-generation'); variable(...)
    if isinstance(node.func, ast.Name):
        varname = node.func.id
        root = node
        while isinstance(getattr(root, "parent", None), ast.AST):
            root = root.parent
        if not isinstance(root, ast.AST):
            root = node
        call_line = getattr(node, "lineno", float("inf"))
        last_line = -1
        found_pipeline = False
        for n in ast.walk(root):
            if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < call_line:
                if any(isinstance(t, ast.Name) and t.id == varname for t in n.targets):
                    v = n.value
                    if isinstance(v, ast.Call):
                        vf = v.func
                        is_pipeline = (isinstance(vf, ast.Name) and vf.id == "pipeline") or \
                                      (isinstance(vf, ast.Attribute) and vf.attr == "pipeline")
                        if is_pipeline and v.args and isinstance(v.args[0], ast.Constant):
                            if str(v.args[0].value) == "text-generation" and getattr(n, "lineno", 0) > last_line:
                                last_line = getattr(n, "lineno", 0)
                                found_pipeline = True
        if found_pipeline:
            return True

    # 6) Gemini var-based
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"generate_content", "start_chat"}:
        return True

    return False


def hasNoTemperatureParameter(node: ast.AST) -> bool:
    """
    Retourne True IFF la température est ABSENTE (missing).
    """
    if not isinstance(node, ast.Call):
        return False  # pas un call

    # 1) Présence directe
    for kw in node.keywords:
        if kw.arg == "temperature":
            return False  # présente -> pas "missing"

    # 2) **kwargs
    for kw in node.keywords:
        if kw.arg is None:  # **kwargs
            val = kw.value

            # **{ ... }
            if isinstance(val, ast.Dict):
                has_temp = any(isinstance(k, ast.Constant) and k.value == "temperature" for k in val.keys)
                return not has_temp  # True si absent

            # **params (variable dict)
            if isinstance(val, ast.Name):
                varname = val.id

                # Racine sûre
                root = node
                while isinstance(getattr(root, "parent", None), ast.AST):
                    root = root.parent
                if not isinstance(root, ast.AST):
                    root = node

                call_line = getattr(node, "lineno", float("inf"))
                last_dict = None
                last_line = -1
                for n in ast.walk(root):
                    if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < call_line:
                        if any(isinstance(t, ast.Name) and t.id == varname for t in n.targets):
                            if isinstance(n.value, ast.Dict) and getattr(n, "lineno", 0) > last_line:
                                last_dict = n.value
                                last_line = getattr(n, "lineno", 0)

                if last_dict is not None:
                    has_temp = any(isinstance(k, ast.Constant) and k.value == "temperature" for k in last_dict.keys)
                    return not has_temp  # True si absent

                # On ne sait pas ce qu'il y a dans **params -> conservateur: missing
                return True

    # 3) Aucun indice de température -> missing
    return True

# --- Regex pour juger si un ID de modèle est "pinné" ---
_OPENAI_OK   = re.compile(r"-20\d{2}-\d{2}-\d{2}$")     # ...-YYYY-MM-DD (ex: gpt-4o-2024-11-20)
_ANTHROPIC_OK = re.compile(r"-20\d{6}$")                # ...-YYYYMMDD   (ex: claude-3-5-haiku-20241022)
_GEMINI_OK   = re.compile(r"-00\d$")                    # ...-001 / -002
_LATEST      = re.compile(r"(?:^|:)latest$")            # 'latest' ou ':latest'

def _is_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)

def _model_is_pinned(s: str) -> bool:
    s = s.strip()
    if _LATEST.search(s):
        return False
    if _OPENAI_OK.search(s) or _ANTHROPIC_OK.search(s) or _GEMINI_OK.search(s):
        return True
    return False

def _get_root(node: ast.AST) -> ast.AST:
    root = node
    while isinstance(getattr(root, "parent", None), ast.AST):
        root = root.parent
    return root if isinstance(root, ast.AST) else node

def _find_last_dict_assignment(root: ast.AST, varname: str, before_line: int) -> ast.Dict:
    last_dict, last_line = None, -1
    for n in ast.walk(root):
        if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < before_line:
            if any(isinstance(t, ast.Name) and t.id == varname for t in n.targets):
                if isinstance(n.value, ast.Dict) and getattr(n, "lineno", 0) > last_line:
                    last_dict, last_line = n.value, getattr(n, "lineno", 0)
    return last_dict

# --------------------------------------------------------------------
# R26 helpers (NOUVEAUX) — pas de recouvrement avec tes fonctions existantes
# --------------------------------------------------------------------

def isModelVersionedLLMCall(node: ast.AST) -> bool:
    """
    True si 'node' est un appel où le "model pinning" a un sens :
      - OpenAI/Anthropic: *.create(..., model=...)
      - Gemini: GenerativeModel("<model-id>")
      - HF Transformers: Auto*.from_pretrained(...)
      - Ollama: subprocess.run([... 'ollama','run','<model[:tag]>' ...]) ou shell string
    """
    if not isinstance(node, ast.Call):
        return False

    # Construire le chemin: openai.ChatCompletion.create -> ('openai','ChatCompletion','create')
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # A) OpenAI / Anthropic
    exacts = {
        ("openai", "Completion", "create"),
        ("openai", "ChatCompletion", "create"),
        ("OpenAI", "completions", "create"),
        ("OpenAI", "chat", "completions", "create"),
        ("anthropic", "completions", "create"),
        ("anthropic", "messages", "create"),
        ("Anthropic", "completions", "create"),
    }
    if call_tuple in exacts:
        return True
    for suf in {("completions", "create"), ("messages", "create"), ("chat", "completions", "create")}:
        if len(call_tuple) >= len(suf) and call_tuple[-len(suf):] == suf:
            return True

    # B) Gemini — GenerativeModel("gemini-...")
    if call_tuple and call_tuple[-1] == "GenerativeModel":
        return True

    # C) HF Transformers — Auto*.from_pretrained(...)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in {
            "AutoModel", "AutoTokenizer", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
            "AutoModelForTokenClassification", "AutoModelForMaskedLM"
        }:
            return True

    # D) Ollama via subprocess
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"run", "check_call"}:
        return True

    return False


def hasNoModelVersionPinning(node: ast.AST) -> bool:
    """
    True si l'appel n'est PAS pinné (alias, 'latest', pas de revision, etc.)
    """
    if not isinstance(node, ast.Call):
        return False

    # Reconstituer chemin
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # 1) OpenAI / Anthropic — via model="..." ou **params
    # a) model="..."
    for kw in node.keywords:
        if kw.arg == "model" and _is_str(kw.value):
            return not _model_is_pinned(kw.value.value)

    # b) **params (dict litéral ou variable)
    for kw in node.keywords:
        if kw.arg is None:
            if isinstance(kw.value, ast.Dict):
                # **{...}
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and k.value == "model" and _is_str(v):
                        return not _model_is_pinned(v.value)
            elif isinstance(kw.value, ast.Name):
                # **params (variable dict)
                root = _get_root(node)
                call_line = getattr(node, "lineno", float("inf"))
                d = _find_last_dict_assignment(root, kw.value.id, call_line)
                if isinstance(d, ast.Dict):
                    for k, v in zip(d.keys, d.values):
                        if isinstance(k, ast.Constant) and k.value == "model" and _is_str(v):
                            return not _model_is_pinned(v.value)

    # 2) Gemini — GenerativeModel("gemini-...")
    if call_tuple and call_tuple[-1] == "GenerativeModel":
        if node.args and _is_str(node.args[0]):
            return not _model_is_pinned(node.args[0].value)
        return True  # arg non-const → conservateur

    # 3) HF Transformers — from_pretrained(..., revision=?)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in {
            "AutoModel", "AutoTokenizer", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
            "AutoModelForTokenClassification", "AutoModelForMaskedLM"
        }:
            for kw in node.keywords:
                if kw.arg == "revision":
                    if _is_str(kw.value):
                        v = kw.value.value.strip().lower()
                        return v in ("", "main", "latest")
                    return False  # revision présent mais non-const → ne pas signaler
            return True  # pas de revision → non pinné

    # 4) Ollama — subprocess.run([... "ollama","run","llama3[:tag]"])
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"run", "check_call"}:
        # a) Liste/tuple d'args
        for arg in node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                items = [e.value for e in arg.elts if _is_str(e)]
                if len(items) >= 3 and items[0] == "ollama" and items[1] == "run":
                    model_tok = items[2]
                    if ":" not in model_tok:
                        return True
                    return _LATEST.search(model_tok) is not None
            # b) Chaîne shell
            if _is_str(arg):
                s = arg.value.strip()
                if s.startswith("ollama run "):
                    model_tok = s.split("ollama run ", 1)[1].split()[0]
                    if ":" not in model_tok:
                        return True
                    return _LATEST.search(model_tok) is not None

    return False

def _kw_value(node: ast.AST, name: str):
    """Retourne la valeur du mot-clé `name` si présent dans l'appel AST."""
    if not isinstance(node, ast.Call):
        return None
    for kw in node.keywords:
        # Cas normal : kw.arg est une string
        if kw.arg == name:
            return kw.value
        # Cas **kwargs : kw.arg est None → on ne sait pas encore
    return None

def _dict_has_key_str(d: ast.Dict, wanted: str) -> bool:
    """True si le dict AST a une clé constante == wanted."""
    if not isinstance(d, ast.Dict):
        return False
    for k in d.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value == wanted:
            return True
    return False

def _list_has_system_message(lst: ast.List) -> bool:
    """
    Retourne True si la liste AST contient un dict ou tuple dont le rôle est 'system'.
    """
    if not isinstance(lst, ast.List):
        return False

    for elt in lst.elts:
        # Cas {"role": "system", "content": "..."}
        if isinstance(elt, ast.Dict):
            for k, v in zip(elt.keys, elt.values):
                if isinstance(k, ast.Constant) and k.value == "role":
                    if isinstance(v, ast.Constant) and v.value == "system":
                        return True

        # Cas ("system", ...)
        if isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
            first = elt.elts[0]
            if isinstance(first, ast.Constant) and first.value == "system":
                return True

    return False



def isRoleBasedLLMChat(node: ast.AST) -> bool:
    """
    True si l'appel est un "chat" role-based (où un 'system' est attendu) :
      - OpenAI Chat Completions: openai.ChatCompletion.create / client.chat.completions.create
      - Anthropic Messages: anthropic.messages.create / client.messages.create
      - OpenAI Responses: client.responses.create (quand input est de type chat)
      - Gemini / Vertex: model.generate_content(...) / model.start_chat(...) où model provient d'un GenerativeModel(...)
    """
    if not isinstance(node, ast.Call):
        return False

    # Reconstruit le chemin ('openai','ChatCompletion','create'), ('client','responses','create'), etc.
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # --- OpenAI Chat ---
    if call_tuple in {("openai", "ChatCompletion", "create"), ("OpenAI", "chat", "completions", "create")}:
        return True
    if len(call_tuple) >= 3 and call_tuple[-3:] == ("chat", "completions", "create"):
        return True

    # --- Anthropic Messages ---
    if call_tuple in {("anthropic", "messages", "create"), ("Anthropic", "messages", "create")}:
        return True
    if len(call_tuple) >= 2 and call_tuple[-2:] == ("messages", "create"):
        return True

    # --- OpenAI Responses ---
    if len(call_tuple) >= 2 and call_tuple[-2:] == ("responses", "create"):
        return True  # on vérifiera "instructions" côté hasNoSystemMessage

    # --- Gemini / Vertex generate_content / start_chat ---
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"generate_content", "start_chat"}:
        # On ne vérifie pas ici le constructeur; hasNoSystemMessage s'en charge (recherche GenerativeModel)
        return True

    return False


def _find_last_list_assignment(root: ast.AST, varname: str, before_line: int):
    last_val, last_line = None, -1
    for n in ast.walk(root):
        ln = getattr(n, "lineno", 0)
        if ln >= before_line:
            continue
        if isinstance(n, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == varname for t in n.targets):
                if isinstance(n.value, (ast.List, ast.Tuple)) and ln > last_line:
                    last_val, last_line = n.value, ln
        if isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and n.target.id == varname:
                v = n.value
                if isinstance(v, (ast.List, ast.Tuple)) and ln > last_line:
                    last_val, last_line = v, ln
    return last_val

def hasNoSystemMessage(node: ast.AST) -> bool:
    """
    True si, avec suffisamment d'évidence statique, aucun "system message/instructions" n'est fourni.
    Couvre :
      - OpenAI Chat: 'messages' ne contient pas {'role':'system',...}
      - Anthropic Messages: kw 'system' absent (même via **kwargs)
      - OpenAI Responses: 'instructions' absent ET input est de type chat (liste de dicts avec 'role')
      - Gemini / Vertex: GenerativeModel(...) sans system_instruction (ou ""), puis generate_content/start_chat
    Conservative: si indéterminé, NE PAS signaler.
    """
    if not isinstance(node, ast.Call) or not isRoleBasedLLMChat(node):
        return False

    # Recrée call_tuple pour distinguer les familles
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # -------------------- OpenAI Chat --------------------
    # Inspecte messages=[...] / **kwargs{"messages": ...}
    if (call_tuple in {("openai", "ChatCompletion", "create"), ("OpenAI", "chat", "completions", "create")}) or \
       (len(call_tuple) >= 3 and call_tuple[-3:] == ("chat", "completions", "create")):
        msgs_kw = _kw_value(node, "messages")

        def _messages_has_system_from_value(val: ast.AST):
            # True/False si déterminé, None sinon
            if isinstance(val, ast.List):
                return _list_has_system_message(val)
            if isinstance(val, ast.Name):
                root = _get_root(node)
                call_line = getattr(node, "lineno", 10**9)
                lst = _find_last_list_assignment(root, val.id, call_line)
                if isinstance(lst, ast.List):
                    return _list_has_system_message(lst)
                return None
            return None

        if msgs_kw is not None:
            has = _messages_has_system_from_value(msgs_kw)
            if has is True:
                return False
            if has is False:
                return True
            return False  # indéterminé

        # Cherche via **kwargs
        for kw in node.keywords:
            if kw.arg is None:
                val = kw.value
                # **{...}
                if isinstance(val, ast.Dict):
                    # Anthropic-style 'system' n'est pas pertinent ici; on cherche 'messages'
                    for k, v in zip(val.keys, val.values):
                        if isinstance(k, ast.Constant) and k.value == "messages":
                            if isinstance(v, ast.List):
                                return not _list_has_system_message(v)
                            if isinstance(v, ast.Name):
                                root = _get_root(node)
                                call_line = getattr(node, "lineno", 10**9)
                                lst = _find_last_list_assignment(root, v.id, call_line)
                                if isinstance(lst, ast.List):
                                    return not _list_has_system_message(lst)
                            return False  # indéterminé
                    return False
                # **params (variable dict)
                if isinstance(val, ast.Name):
                    root = _get_root(node)
                    call_line = getattr(node, "lineno", 10**9)
                    d = _find_last_dict_assignment(root, val.id, call_line)
                    if isinstance(d, ast.Dict):
                        for k, v in zip(d.keys, d.values):
                            if isinstance(k, ast.Constant) and k.value == "messages":
                                if isinstance(v, ast.List):
                                    return not _list_has_system_message(v)
                                if isinstance(v, ast.Name):
                                    lst = _find_last_list_assignment(root, v.id, call_line)
                                    if isinstance(lst, ast.List):
                                        return not _list_has_system_message(lst)
                        return False
                    return False
        return False  # rien de concluant → ne pas signaler

    # -------------------- Anthropic Messages --------------------
    if (call_tuple in {("anthropic", "messages", "create"), ("Anthropic", "messages", "create")}) or \
       (len(call_tuple) >= 2 and call_tuple[-2:] == ("messages", "create")):
        # kw 'system' direct ?
        if _kw_value(node, "system") is not None:
            return False
        # via **kwargs
        for kw in node.keywords:
            if kw.arg is None:
                v = kw.value
                if isinstance(v, ast.Dict):
                    if _dict_has_key_str(v, "system"):
                        return False
                    return True  # dict présent mais sans 'system' → missing
                if isinstance(v, ast.Name):
                    root = _get_root(node)
                    call_line = getattr(node, "lineno", 10**9)
                    d = _find_last_dict_assignment(root, v.id, call_line)
                    if isinstance(d, ast.Dict):
                        if _dict_has_key_str(d, "system"):
                            return False
                        return True
                    return False  # indéterminé
        # Aucun 'system' visible → signaler
        return True

    # -------------------- OpenAI Responses --------------------
    if len(call_tuple) >= 2 and call_tuple[-2:] == ("responses", "create"):
        # Si 'instructions' est présent (même via **kwargs) → OK
        if _kw_value(node, "instructions") is not None:
            return False
        for kw in node.keywords:
            if kw.arg is None:
                v = kw.value
                if isinstance(v, ast.Dict):
                    if _dict_has_key_str(v, "instructions"):
                        return False
                    inp = None
                    for k, val in zip(v.keys, v.values):
                        if isinstance(k, ast.Constant) and k.value == "input":
                            inp = val
                            break
                    # input chat-like ?
                    chat_like = False
                    if isinstance(inp, ast.List):
                        chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in inp.elts)
                    elif isinstance(inp, ast.Name):
                        root = _get_root(node)
                        call_line = getattr(node, "lineno", 10**9)
                        lst = _find_last_list_assignment(root, inp.id, call_line)
                        if isinstance(lst, ast.List):
                            chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in lst.elts)
                    return chat_like  # report seulement si clairement chat-like
                if isinstance(v, ast.Name):
                    root = _get_root(node)
                    call_line = getattr(node, "lineno", 10**9)
                    d = _find_last_dict_assignment(root, v.id, call_line)
                    if isinstance(d, ast.Dict):
                        if _dict_has_key_str(d, "instructions"):
                            return False
                        # inspect input
                        inp = None
                        for k, val in zip(d.keys, d.values):
                            if isinstance(k, ast.Constant) and k.value == "input":
                                inp = val
                                break
                        chat_like = False
                        if isinstance(inp, ast.List):
                            chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in inp.elts)
                        elif isinstance(inp, ast.Name):
                            lst = _find_last_list_assignment(root, inp.id, call_line)
                            if isinstance(lst, ast.List):
                                chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in lst.elts)
                        return chat_like
                    return False  # indéterminé
        # input=... direct ?
        inp = _kw_value(node, "input")
        chat_like = False
        if isinstance(inp, ast.List):
            chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in inp.elts)
        elif isinstance(inp, ast.Name):
            root = _get_root(node)
            call_line = getattr(node, "lineno", 10**9)
            lst = _find_last_list_assignment(root, inp.id, call_line)
            if isinstance(lst, ast.List):
                chat_like = any(isinstance(e, ast.Dict) and _dict_has_key_str(e, "role") for e in lst.elts)
        return chat_like  # True ⇒ report (instructions manquantes)

    # -------------------- Gemini / Vertex generate_content / start_chat --------------------
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"generate_content", "start_chat"}:
        root = _get_root(node)
        call_line = getattr(node, "lineno", 10**9)

        # Helper inline: vérifie si un Call est un constructeur GenerativeModel(...)
        def _is_gm_ctor(call_obj: ast.AST) -> bool:
            if not isinstance(call_obj, ast.Call):
                return False
            ff = call_obj.func
            return (isinstance(ff, ast.Name) and ff.id == "GenerativeModel") or \
                   (isinstance(ff, ast.Attribute) and ff.attr == "GenerativeModel")

        # Helper inline: teste si system_instruction est manquant ou vide ("") dans un ctor
        def _gm_ctor_missing_sysinstr(ctor: ast.Call) -> bool:
            si = _kw_value(ctor, "system_instruction")
            if si is not None:
                # vide => report
                if isinstance(si, ast.Constant) and isinstance(si.value, str) and si.value == "":
                    return True
                return False
            # **kwargs
            for kw in ctor.keywords:
                if kw.arg is None:
                    v = kw.value
                    if isinstance(v, ast.Dict):
                        val = None
                        for k, dv in zip(v.keys, v.values):
                            if isinstance(k, ast.Constant) and k.value == "system_instruction":
                                val = dv
                                break
                        if val is None:
                            return True
                        if isinstance(val, ast.Constant) and isinstance(val.value, str) and val.value == "":
                            return True
                        return False
                    if isinstance(v, ast.Name):
                        d = _find_last_dict_assignment(root, v.id, getattr(ctor, "lineno", 10**9))
                        if isinstance(d, ast.Dict):
                            val = None
                            for k, dv in zip(d.keys, d.values):
                                if isinstance(k, ast.Constant) and k.value == "system_instruction":
                                    val = dv
                                    break
                            if val is None:
                                return True
                            if isinstance(val, ast.Constant) and isinstance(val.value, str) and val.value == "":
                                return True
                            return False
                        return False
            # aucun kw pertinent → manquant
            return True

        # Cas A: appel chaîné GenerativeModel(...).generate_content(...)
        base_expr = node.func.value
        if isinstance(base_expr, ast.Call) and _is_gm_ctor(base_expr):
            return _gm_ctor_missing_sysinstr(base_expr)

        # Cas B: var.generate_content(...), retrouver dernière affectation var = GenerativeModel(...)
        # Trouver le dernier Assign à cette variable avant l'appel
        base_name = None
        b = node.func.value
        while isinstance(b, ast.Attribute):
            b = b.value
        if isinstance(b, ast.Name):
            base_name = b.id

        if base_name:
            last_ctor = None
            last_line = -1
            for n in ast.walk(root):
                if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < call_line:
                    if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == base_name:
                        if isinstance(n.value, ast.Call) and _is_gm_ctor(n.value):
                            if getattr(n, "lineno", 0) > last_line:
                                last_ctor, last_line = n.value, getattr(n, "lineno", 0)
            if isinstance(last_ctor, ast.Call):
                return _gm_ctor_missing_sysinstr(last_ctor)

        return False  # indéterminé

    return False

def hasNoBoundedMetrics(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isLLMCall(node):
        print("DBG_R28.hasNoBoundedMetrics: not a Call or not LLM -> False")
        return False

    # bornes directes (max_output_tokens / timeout)
    for kw in node.keywords:
        if kw.arg in { "max_tokens","max_output_tokens", "timeout"}:
            return False

    # Gemini: generate_content / start_chat
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"generate_content", "start_chat"}:
        gen_cfg = _kw_value(node, "generation_config")
        if gen_cfg is None:
            return True
        if isinstance(gen_cfg, ast.Dict):
            if not _dict_has_key_str(gen_cfg, "max_output_tokens"):
                return True
            return False
        elif isinstance(gen_cfg, ast.Name):
            root = _get_root(node)
            call_line = getattr(node, "lineno", 10**9)
            d = _find_last_dict_assignment(root, gen_cfg.id, call_line)
            if isinstance(d, ast.Dict):
                if not _dict_has_key_str(d, "max_output_tokens"):
                    return True
                return False
            else:
                return True  # indéterminable ⇒ conservateur : unbounded

    # **kwargs (dicts passés en bloc)
    for kw in node.keywords:
        if kw.arg is None:
            v = kw.value
            if isinstance(v, ast.Dict):
                if any(_dict_has_key_str(v, k) for k in [ "max_output_tokens", "timeout"]):
                    return False
            elif isinstance(v, ast.Name):
                root = _get_root(node)
                call_line = getattr(node, "lineno", 10**9)
                d = _find_last_dict_assignment(root, v.id, call_line)
                if isinstance(d, ast.Dict):
                    if any(_dict_has_key_str(d, k) for k in ["max_output_tokens", "timeout"]):
                        return False

    # Timeout dans un 'with client.with_options(timeout=...)' englobant
    parent = getattr(node, "parent", None)
    while parent:
        if isinstance(parent, ast.With):
            for item in parent.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute) and ctx.func.attr == "with_options":
                    if _kw_value(ctx, "timeout") is not None:
                        return False
        parent = getattr(parent, "parent", None)

    # aucune borne trouvée → unbounded
    return True

def isNotSDKClient(node: ast.AST) -> bool:
    """Retourne False si le call correspond au SDK OpenAI (client, with_options, etc.),
    True sinon (cas LangChain LLM, ChatOpenAI, responses.create, etc.)"""
    if not isinstance(node, ast.Call):
        return True

    # Reconstitue le chemin
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # Exclusions explicites : OpenAI SDK
    if call_tuple in {('OpenAI',), ('client', 'with_options')}:
        return False

    return True

def isPipelineCall(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    # Reconstituer le chemin
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # Cas typiques de pipelines/orchestrateurs
    PIPELINE_CALLS = {
        ('pipeline',),
        ('StructuredOutputParser', 'parse'),
        ('PydanticOutputParser', 'parse'),
        ('OpenAI', 'bind'),
        ('LLMChain',),
        ('load_summarize_chain',),
    }
    if call_tuple in PIPELINE_CALLS:
        return True

    return False

def hasNoStructuredOutput(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    # Vérifie si un argument impose un format structuré
    for kw in node.keywords:
        if kw.arg in {"response_format", "format", "parser", "output_parser"}:
            if isinstance(kw.value, ast.Constant) and str(kw.value.value).lower() in {"json", "json_object"}:
                return False  # structuré via JSON
            if isinstance(kw.value, ast.Name) or isinstance(kw.value, ast.Call):
                return False  # parser fourni

    # Si aucun format ou parser explicite → smell
    return True

def isUnstructuredLLMCallInPipeline(node: ast.AST) -> bool:
    """
    True si l'appel est une génération LLM *et* aucun format/parseur structuré n'est fourni.
    Évite les faux positifs:
      - Ne considère pas asyncio.run / objets locaux arbitraires .run()
      - Ne retourne pas True par défaut quand l'origine est inconnue
    """
    if not isinstance(node, ast.Call):
        return False

    def _has_structured_kwargs(call: ast.Call) -> bool:
        for kw in call.keywords:
            if kw.arg in {"response_format", "format"} and isinstance(kw.value, ast.Constant):
                val = str(getattr(kw.value, "value", "")).lower()
                if val in {"json", "json_object"}:
                    return True
            if kw.arg in {"parser", "output_parser"}:
                return True
        return False

    # Reconstituer le chemin
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # 1) Appels create/messages/responses/generate_content (OpenAI/Anthropic/Gemini...)
    gen_suffixes = {
        ("ChatCompletion", "create"),
        ("Completion", "create"),
        ("responses", "create"),
        ("messages", "create"),
        ("chat", "completions", "create"),
        ("generative_models", "generate_content"),
    }
    if any(len(call_tuple) >= len(suf) and call_tuple[-len(suf):] == suf for suf in gen_suffixes):
        return not _has_structured_kwargs(node)

    # 2) LangChain: chain.run()/invoke()/predict()/call()
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"run", "invoke", "predict", "call"}:
        base = node.func.value
        # exclusions explicites (non LLM)
        if isinstance(base, ast.Name) and base.id in {"asyncio"}:
            return False
        if isinstance(base, ast.Name):
            varname = base.id
            root = _get_root(node)
            call_line = getattr(node, "lineno", 10**9)
            last_assign = _find_last_assignment(root, varname, call_line)
            if isinstance(last_assign, ast.Assign):
                v = last_assign.value
                if isinstance(v, ast.Call):
                    is_llmchain_ctor = (
                        (isinstance(v.func, ast.Name) and v.func.id == "LLMChain") or
                        (isinstance(v.func, ast.Attribute) and v.func.attr == "LLMChain")
                    )
                    if is_llmchain_ctor:
                        has_parser = any(akw.arg in {"parser", "output_parser"} for akw in v.keywords)
                        return not has_parser
            # On n'a pas prouvé que c'est une chaîne LLM → NE PAS reporter
            return False

    # 3) transformers.pipeline('text-generation') puis var(...)
    if isinstance(node.func, ast.Name):
        varname = node.func.id
        root = _get_root(node)
        call_line = getattr(node, "lineno", 10**9)
        last_assign = _find_last_assignment(root, varname, call_line)
        if isinstance(last_assign, ast.Assign):
            v = last_assign.value
            if isinstance(v, ast.Call):
                vf = v.func
                is_pipeline = (isinstance(vf, ast.Name) and vf.id == "pipeline") or \
                              (isinstance(vf, ast.Attribute) and vf.attr == "pipeline")
                if is_pipeline and v.args and isinstance(v.args[0], ast.Constant):
                    if str(v.args[0].value) == "text-generation":
                        # Pas de notion de response_format ici → considéré "unstructured"
                        return True

    return False


def _find_last_assignment(root: ast.AST, varname: str, before_line: int) -> ast.Assign:
    """Retourne la dernière ast.Assign qui assigne `varname` avant `before_line`, sinon None."""
    last = None
    last_ln = -1
    for n in ast.walk(root):
        if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < before_line:
            # var à gauche ?
            if any(isinstance(t, ast.Name) and t.id == varname for t in n.targets):
                ln = getattr(n, "lineno", 0)
                if ln > last_ln:
                    last, last_ln = n, ln
    return last


def isTextGeneratingCall(node: ast.AST) -> bool:
    """
    True si l'appel correspond à une génération de texte.
    - Suffixes connus (.create/.messages.create/.responses.create/…)
    - Méthodes génératives (.run/.invoke/.predict/.generate/.generate_content/.start_chat)
    - Appel d'une variable issue de transformers.pipeline('text-generation'):  gen(...)
    """
    if not isinstance(node, ast.Call):
        return False

    # Reconstitue le chemin de l'appel
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # Exclusions: constructeurs/helpers non génératifs
    constructor_like = {
        ("OpenAI",), ("LLMChain",), ("ChatOpenAI",), ("ChatAnthropic",),
        ("ChatCohere",), ("ChatVertexAI",), ("HuggingFacePipeline",),
    }
    helper_factories = {
        ("StructuredOutputParser", "from_response_schemas"),
        ("PydanticOutputParser",),
    }
    if call_tuple in constructor_like or call_tuple in helper_factories:
        return False

    # Suffixes génératifs connus
    gen_suffixes = {
        ("ChatCompletion", "create"),
        ("Completion", "create"),
        ("responses", "create"),
        ("messages", "create"),
        ("chat", "completions", "create"),
        ("generative_models", "generate_content"),
    }
    for suf in gen_suffixes:
        if len(call_tuple) >= len(suf) and call_tuple[-len(suf):] == suf:
            return True

    # Méthodes génératives courantes
    if isinstance(node.func, ast.Attribute) and node.func.attr in {
        "run", "invoke", "predict", "generate", "generate_content", "start_chat"
    }:
        return True

    # Appel d'une variable : gen(...) – vérifier si gen provient de pipeline('text-generation')
    if isinstance(node.func, ast.Name):
        varname = node.func.id
        root = _get_root(node)
        call_line = getattr(node, "lineno", 10**9)
        last_assign = _find_last_assignment(root, varname, call_line)
        if isinstance(last_assign, ast.Assign):
            v = last_assign.value
            if isinstance(v, ast.Call):
                vf = v.func
                is_pipeline = (isinstance(vf, ast.Name) and vf.id == "pipeline") or \
                              (isinstance(vf, ast.Attribute) and vf.attr == "pipeline")
                if is_pipeline and v.args and isinstance(v.args[0], ast.Constant):
                    return str(v.args[0].value) == "text-generation"
    return False

def isReasoningModelCall(node: ast.AST) -> bool:
    """
    Détecte si l'appel utilise un modèle de raisonnement (reasoning-capable model).
    Couvre:
      - OpenAI: o1, o1-mini, o1-preview, gpt-5.x
      - Anthropic: claude-3-7-sonnet (thinking mode)
      - Google: gemini-2.0-flash-thinking-exp
    """
    if not isinstance(node, ast.Call):
        return False

    # Reconstitue le chemin de l'appel
    path = []
    f = node.func
    while isinstance(f, ast.Attribute):
        path.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        path.insert(0, f.id)
    call_tuple = tuple(path)

    # Modèles de raisonnement connus
    reasoning_model_patterns = [
        r"^o1(-preview|-mini)?$",           # OpenAI o1, o1-preview, o1-mini
        r"^gpt-5",                          # OpenAI GPT-5.x
        r"^claude-3-7-sonnet",              # Claude 3.7 Sonnet (thinking)
        r"^gemini-2\.0-flash-thinking",     # Gemini 2.0 thinking
    ]

    # Vérifie le paramètre 'model' pour OpenAI/Anthropic
    if call_tuple and call_tuple[-1] in {"create", "generate_content"}:
        model_kw = _kw_value(node, "model")
        if model_kw and isinstance(model_kw, ast.Constant) and isinstance(model_kw.value, str):
            model_name = model_kw.value
            for pattern in reasoning_model_patterns:
                if re.search(pattern, model_name):
                    return True

        # Vérifie dans **kwargs
        for kw in node.keywords:
            if kw.arg is None:
                v = kw.value
                if isinstance(v, ast.Dict):
                    for k, val in zip(v.keys, v.values):
                        if isinstance(k, ast.Constant) and k.value == "model":
                            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                for pattern in reasoning_model_patterns:
                                    if re.search(pattern, val.value):
                                        return True
                elif isinstance(v, ast.Name):
                    root = _get_root(node)
                    call_line = getattr(node, "lineno", 10**9)
                    d = _find_last_dict_assignment(root, v.id, call_line)
                    if isinstance(d, ast.Dict):
                        for k, val in zip(d.keys, d.values):
                            if isinstance(k, ast.Constant) and k.value == "model":
                                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                    for pattern in reasoning_model_patterns:
                                        if re.search(pattern, val.value):
                                            return True

    # Pour Gemini: GenerativeModel("gemini-2.0-flash-thinking-...")
    if call_tuple and call_tuple[-1] == "GenerativeModel":
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            model_name = node.args[0].value
            for pattern in reasoning_model_patterns:
                if re.search(pattern, model_name):
                    return True

    return False

def hasNoReasoningEffort(node: ast.AST) -> bool:
    """
    Retourne True si l'appel à un modèle de raisonnement n'a pas de paramètre
    explicite pour contrôler l'effort/profondeur de raisonnement.

    Paramètres recherchés:
      - reasoning, reasoning_effort, reasoning_depth (OpenAI)
      - thinking, thinking_config (Anthropic/Google)
    """
    if not isinstance(node, ast.Call):
        return False

    # Paramètres de contrôle du raisonnement
    reasoning_params = {
        "reasoning", "reasoning_effort", "reasoning_depth",
        "thinking", "thinking_config"
    }

    # Vérifie les paramètres directs
    for kw in node.keywords:
        if kw.arg in reasoning_params:
            return False  # Paramètre présent -> pas de smell

    # Vérifie dans **kwargs (dict littéral)
    for kw in node.keywords:
        if kw.arg is None:
            v = kw.value
            if isinstance(v, ast.Dict):
                for k in v.keys:
                    if isinstance(k, ast.Constant) and k.value in reasoning_params:
                        return False
            elif isinstance(v, ast.Name):
                # **params (variable dict)
                root = _get_root(node)
                call_line = getattr(node, "lineno", 10**9)
                d = _find_last_dict_assignment(root, v.id, call_line)
                if isinstance(d, ast.Dict):
                    for k in d.keys:
                        if isinstance(k, ast.Constant) and k.value in reasoning_params:
                            return False

    # Pour Gemini: vérifie generation_config
    gen_cfg = _kw_value(node, "generation_config")
    if gen_cfg:
        if isinstance(gen_cfg, ast.Dict):
            if _dict_has_key_str(gen_cfg, "thinking_config"):
                return False
        elif isinstance(gen_cfg, ast.Name):
            root = _get_root(node)
            call_line = getattr(node, "lineno", 10**9)
            d = _find_last_dict_assignment(root, gen_cfg.id, call_line)
            if isinstance(d, ast.Dict) and _dict_has_key_str(d, "thinking_config"):
                return False

    # Aucun paramètre de raisonnement trouvé -> smell détecté
    return True

def isVisionModelCall(node: ast.AST) -> bool:
    """
    Détecte si l'appel utilise un modèle/API avec capacités vision.
    Version améliorée avec plus de patterns et providers.
    """
    if not isinstance(node, ast.Call):
        return False

    parts = []
    f = node.func
    while isinstance(f, ast.Attribute):
        parts.insert(0, f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.insert(0, f.id)
    tup = tuple(parts)

    # Patterns étendus couvrant plus de providers et méthodes
    vision_patterns = [
        ("OpenAI", "responses", "create"),
        ("client", "responses", "create"),
        ("openai", "ChatCompletion", "create"),
        ("openai", "chat", "completions", "create"),
        ("chat", "completions", "create"),
        ("completions", "create"),
        ("anthropic", "messages", "create"),
        ("Anthropic", "messages", "create"),
        ("client", "messages", "create"),
        ("messages", "create"),
        ("model", "generate_content"),
        ("GenerativeModel", "generate_content"),
        ("genai", "GenerativeModel", "generate_content"),
        ("generate_content",),
        ("AzureOpenAI", "chat", "completions", "create"),
        ("azure_client", "chat", "completions", "create"),
        ("cohere", "chat"),
        ("co", "chat"),
        ("bedrock", "invoke_model"),
        ("bedrock_runtime", "invoke_model"),
        ("invoke_model",),
        ("pipeline",),
        ("transformers", "pipeline"),
        ("mistral", "chat"),
        ("MistralClient", "chat"),
        ("ollama", "chat"),
        ("ollama", "generate"),
        ("ChatOpenAI",),
        ("ChatAnthropic",),
        ("ChatGoogleGenerativeAI",),
        ("llm", "complete"),
        ("llm", "chat"),
    ]

    for p in vision_patterns:
        if len(tup) >= len(p) and tuple(tup[-len(p):]) == p:
            return True

    if parts:
        last = parts[-1]
        if last in {"create", "generate_content", "messages", "chat", "complete",
                    "invoke", "invoke_model", "predict", "generate", "run"}:
            context_indicators = {"client", "model", "llm", "chat", "ai", "gpt",
                                  "claude", "gemini", "anthropic", "openai",
                                  "bedrock", "cohere", "mistral", "ollama"}
            if any(ind in p.lower() for p in parts for ind in context_indicators):
                return True

    try:
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"generate_content", "generate"}:
            obj = node.func.value
            if isinstance(obj, ast.Name):
                root = _get_root(node)
                call_line = getattr(node, "lineno", 10**9)
                last_assign = _find_last_assignment(root, obj.id, call_line)
                if isinstance(last_assign, ast.Assign) and isinstance(last_assign.value, ast.Call):
                    cf = last_assign.value.func
                    model_constructors = ["generativ", "chatmodel", "llm", "visionmodel",
                                         "multimodal", "gpt", "claude", "generativemodel"]
                    if isinstance(cf, ast.Attribute):
                        if any(mc in cf.attr.lower() for mc in model_constructors):
                            return True
                    if isinstance(cf, ast.Name):
                        if any(mc in cf.id.lower() for mc in model_constructors):
                            return True
    except Exception:
        pass

    # Ajout explicite pour ollama.* calls (ex: ollama.chat(...))
    try:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base = node.func.value.id.lower()
            meth = node.func.attr.lower()
            if base == "ollama" and meth in {"chat", "generate", "run"}:
                return True
    except Exception:
        pass

    return False

def hasImageContent(node: ast.AST) -> bool:
    """
    Détecte si l'appel contient du contenu image.
    Version améliorée avec plus de formats et patterns.
    """
    #if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
       #if node.func.value.id.lower() == "ollama":
          #return False

    if not isinstance(node, ast.Call):
        return False

    def _is_image_bytes_or_url(obj: ast.AST) -> bool:
        if isinstance(obj, ast.Constant):
            v = obj.value
            if isinstance(v, (bytes, bytearray, memoryview)):
                return True
            if isinstance(v, str):
                vl = v.lower()
                if vl.startswith("data:image/") or vl.startswith("http://") or vl.startswith("https://") or vl.startswith("file://") or vl.startswith("s3://") or vl.startswith("gs://"):
                    return True
                image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg", ".ico", ".heic", ".heif"}
                if any(vl.endswith(ext) for ext in image_exts):
                    return True
        return False

    def _contains_image_deep(obj: ast.AST, depth: int = 12) -> bool:
        if depth <= 0 or obj is None:
            return False
        if _is_image_bytes_or_url(obj):
            return True
        if isinstance(obj, ast.Dict):
            if _is_image_dict(obj):
                return True
            for v in obj.values:
                if _contains_image_deep(v, depth - 1):
                    return True
            return False
        if isinstance(obj, (ast.List, ast.Tuple)):
            for elt in obj.elts:
                if _contains_image_deep(elt, depth - 1):
                    return True
            return False
        if isinstance(obj, ast.Name):
            name = obj.id.lower()
            image_indicators = ["image", "img", "screenshot", "photo", "picture", "pic", "bytes", "data", "raw", "binary", "media", "file", "vision", "visual", "frame", "capture", "scan"]
            if any(tok in name for tok in image_indicators):
                root = _get_root(node)
                call_line = getattr(node, "lineno", 10**9)
                last_assign = _find_last_assignment(root, name, call_line)
                if isinstance(last_assign, ast.Assign):
                    return _contains_image_deep(last_assign.value, depth - 1)
                return True
            return False
        if isinstance(obj, ast.Call):
            fn = obj.func
            fname = fn.id.lower() if isinstance(fn, ast.Name) else fn.attr.lower() if isinstance(fn, ast.Attribute) else None
            image_func_indicators = ["read", "open", "load", "imread", "image", "pil", "cv2", "bytes", "getvalue", "encode", "decode", "fetch", "download", "from_file", "from_url", "get_image", "capture", "screenshot", "b64decode", "base64"]
            if fname and any(ind in fname for ind in image_func_indicators):
                return True
            if isinstance(fn, ast.Attribute):
                parts = []
                temp = fn
                while isinstance(temp, ast.Attribute):
                    parts.insert(0, temp.attr)
                    temp = temp.value
                if isinstance(temp, ast.Name):
                    parts.insert(0, temp.id)
                path = ".".join(parts).lower()
                image_module_patterns = ["pil.image", "image.open", "cv2.imread", "imageio.imread", "skimage.io", "matplotlib.image", "torchvision"]
                if any(pattern in path for pattern in image_module_patterns):
                    return True
            for a in obj.args:
                if _contains_image_deep(a, depth - 1):
                    return True
            for kw in obj.keywords:
                if _contains_image_deep(kw.value, depth - 1):
                    return True
        return False

    def _is_image_dict(obj: ast.AST) -> bool:
        if not isinstance(obj, ast.Dict):
            return False
        for k, v in zip(obj.keys, obj.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                key = k.value.lower()
                image_keys = {"type", "image_url", "source", "url", "image", "images", "image_data", "mime_type", "media_type", "content_type", "format", "inline_data", "data", "image_content", "binary_data", "file_path", "path", "file_url", "s3_uri", "gcs_uri"}
                if key in image_keys:
                    if _is_image_bytes_or_url(v):
                        return True
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        if any(tok in v.value.lower() for tok in ["image", "photo", "picture", "screenshot", "png", "jpg", "jpeg"]):
                            return True
                        if v.value.lower().startswith(("http", "data:", "file:", "s3:", "gs:")):
                            return True
                    if key == "images" and isinstance(v, (ast.List, ast.Tuple)) and _contains_image_deep(v):
                        return True
                if key == "mime_type" and isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.lower().startswith("image/"):
                    return True
                if key == "type" and isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.lower() == "image":
                    return True
                if key in {"content", "parts", "data", "source", "images", "media", "attachments", "payload"} and isinstance(v, (ast.Dict, ast.List, ast.Tuple)) and _contains_image_deep(v):
                    return True
        return False

    for arg in node.args:
        if _contains_image_deep(arg):
            return True

    image_param_names = {"input", "messages", "content", "image", "data", "input_image", "images", "media", "files", "attachments", "parts", "payload", "multimodal_content", "vision_input", "image_url", "image_data"}
    for kw in node.keywords:
        if kw.arg in image_param_names:
            if _contains_image_deep(kw.value):
                return True
        if kw.arg is None and _contains_image_deep(kw.value):
            return True

    return False


def hasImagePreprocessing(node: ast.AST, preprocessed_vars: Optional[Set[str]] = None) -> bool:
    """
    Détecte si l'image a été prétraitée.
    Version étendue avec plus de bibliothèques et patterns.
    """
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
       if node.func.value.id.lower() == "ollama":
          return False


    def _check_preprocessing_in_value(val: ast.AST, depth: int = 6) -> bool:
        if depth <= 0 or val is None:
            return False
        if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
            root = _get_root(node)
            funcname = val.func.id
            for f in ast.walk(root):
                if isinstance(f, ast.FunctionDef) and f.name == funcname:
                    for stmt in ast.walk(f):
                       if isinstance(stmt, ast.Call) and _call_indicates_preprocessing(stmt):
                          return True

        if _value_is_preprocessed(val, node):
            return True
        if isinstance(val, ast.Call) and _call_indicates_preprocessing(val):
            return True
        if isinstance(val, ast.Name):
            if preprocessed_vars and val.id in preprocessed_vars:
                return True
            if any(ind in val.id.lower() for ind in PREPROCESSING_NAME_HINTS):
                return True
            root = _get_root(node)
            call_line = getattr(node, "lineno", 10**9)
            last_assign = _find_last_assignment(root, val.id, call_line)
            if isinstance(last_assign, ast.Assign):
                return _check_preprocessing_in_value(last_assign.value, depth - 1)
        if isinstance(val, ast.Dict):
            for v in val.values:
                if _check_preprocessing_in_value(v, depth - 1):
                    return True
        if isinstance(val, (ast.List, ast.Tuple)):
            for elt in val.elts:
                if _check_preprocessing_in_value(elt, depth - 1):
                    return True
        return False

    detail_kw = _kw_value(node, "detail")
    if isinstance(detail_kw, ast.Constant) and detail_kw.value in {"low", "high"}:
        return True

    quality_params = {"quality", "compression", "image_quality", "resolution", "max_size", "max_dimension"}
    for kw in node.keywords:
        if kw.arg in quality_params:
            return True

    for arg in node.args:
        if _check_preprocessing_in_value(arg):
            return True

    relevant_kwargs = {"input", "messages", "content", "image", "data", "images", "media", "files", "image_url", "image_data"}
    for kw in node.keywords:
        if kw.arg in relevant_kwargs and _check_preprocessing_in_value(kw.value):
            return True
        if kw.arg is None and isinstance(kw.value, ast.Dict):
            for v in kw.value.values:
                if _check_preprocessing_in_value(v):
                    return True

    return False


def hasExplicitDetailLevel(node: ast.AST) -> bool:
    """
    Vérifie si un appel configure explicitement le niveau de détail de l'image.
    """
    if not isinstance(node, ast.Call):
        return False

    detail_keys = {"detail", "quality", "image_detail", "image_quality", "resolution", "fidelity", "precision", "detail_level", "max_image_tokens", "tokens_per_image"}
    explicit_values = {"low", "high", "medium", "detailed", "basic", "full"}
    non_explicit_values = {"auto", "default"}
    name_indicators = {"detail", "quality", "lowres", "highres"}

    root = _get_root(node)
    call_line = getattr(node, "lineno", 10**9)

    def _resolve_name(name: ast.Name) -> Optional[ast.AST]:
        assign = _find_last_assignment(root, name.id, call_line)
        if isinstance(assign, ast.Assign):
            return assign.value
        return None

    def _detail_value_status(val: ast.AST, depth: int) -> Optional[bool]:
        if depth <= 0 or val is None:
            return None
        if isinstance(val, ast.Constant):
            const_val = val.value
            if isinstance(const_val, str):
                lowered = const_val.lower()
                if lowered in non_explicit_values:
                    return False
                if lowered in explicit_values or lowered:
                    return True
            if isinstance(const_val, (int, float)):
                return True
            return None
        if isinstance(val, ast.Name):
            resolved = _resolve_name(val)
            if resolved is None:
                if any(tok in val.id.lower() for tok in name_indicators):
                    return True
                return None
            return _detail_value_status(resolved, depth - 1)
        if isinstance(val, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            return _scan_for_detail(val, depth - 1)
        if isinstance(val, ast.Call):
            for kw in val.keywords:
                if kw.arg and kw.arg.lower() in detail_keys:
                    status = _detail_value_status(kw.value, depth - 1)
                    if status is not None:
                        return status
            for arg in val.args:
                status = _detail_value_status(arg, depth - 1)
                if status:
                    return True
            return True
        return True

    def _scan_for_detail(expr: ast.AST, depth: int) -> Optional[bool]:
        if depth <= 0 or expr is None:
            return None
        if isinstance(expr, ast.Dict):
            for key, value in zip(expr.keys, expr.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    lowered = key.value.lower()
                    if lowered in detail_keys:
                        status = _detail_value_status(value, depth - 1)
                        if status is not None:
                            return status
                status = _scan_for_detail(value, depth - 1)
                if status is not None:
                    return status
            return None
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            for elt in expr.elts:
                status = _scan_for_detail(elt, depth - 1)
                if status is not None:
                    return status
            return None
        if isinstance(expr, ast.Name):
            resolved = _resolve_name(expr)
            if resolved is None:
                if any(tok in expr.id.lower() for tok in name_indicators):
                    return True
                return None
            return _scan_for_detail(resolved, depth - 1)
        if isinstance(expr, ast.Call):
            for kw in expr.keywords:
                if kw.arg and kw.arg.lower() in detail_keys:
                    status = _detail_value_status(kw.value, depth - 1)
                    if status is not None:
                        return status
                status = _scan_for_detail(kw.value, depth - 1)
                if status is not None:
                    return status
            for arg in expr.args:
                status = _scan_for_detail(arg, depth - 1)
                if status is not None:
                    return status
            return None
        return None

    for kw in node.keywords:
        if kw.arg in detail_keys:
            status = _detail_value_status(kw.value, 6)
            if status:
                return True

    structured_kwargs = {"messages", "input", "content", "image_url", "image", "images", "media", "parts", "multimodal_content", "attachments"}
    for kw in node.keywords:
        if kw.arg in structured_kwargs:
            status = _scan_for_detail(kw.value, 6)
            if status:
                return True
        if kw.arg is None:
            status = _scan_for_detail(kw.value, 6)
            if status:
                return True

    for arg in node.args:
        status = _scan_for_detail(arg, 6)
        if status:
            return True

    return False


def _value_is_preprocessed(val: ast.AST, call: ast.Call) -> bool:
    """
    Détecte si une valeur provient clairement d'un pipeline de preprocessing.
    Suit buffer.tobytes() / buffer.getvalue() après imencode/save.
    """
    root = _get_root(call)
    call_line = getattr(call, "lineno", 10**9)

    if isinstance(val, ast.Call):
        func = val.func
        fname = func.id.lower() if isinstance(func, ast.Name) else func.attr.lower() if isinstance(func, ast.Attribute) else ""
        if fname and any(p in fname for p in PREPROCESSING_FUNC_NAMES):
            return True
        if isinstance(func, ast.Attribute) and func.attr.lower() in {"tobytes", "getvalue", "to_bytes"}:
            buf = func.value
            if isinstance(buf, ast.Name):
                bufname = buf.id
                for n in ast.walk(root):
                    if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < call_line:
                        for target in n.targets:
                            matched_var = False
                            if isinstance(target, ast.Name) and target.id == bufname:
                                matched_var = True
                            elif isinstance(target, (ast.Tuple, ast.List)):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name) and elt.id == bufname:
                                        matched_var = True
                                        break
                            if matched_var:
                                rhs = n.value
                                if isinstance(rhs, ast.Call) and _call_indicates_preprocessing(rhs):
                                    return True
                for n in ast.walk(root):
                    if isinstance(n, ast.Call) and getattr(n, "lineno", 0) < call_line:
                        if isinstance(n.func, ast.Attribute) and n.func.attr.lower() == "save":
                            recv = n.func.value
                            for arg in n.args:
                                if isinstance(arg, ast.Name) and arg.id == bufname:
                                    if _value_is_preprocessed(recv, call):
                                        return True
        for a in val.args:
            if _value_is_preprocessed(a, call):
                return True
        for kw in val.keywords:
            if _value_is_preprocessed(kw.value, call):
                return True
        return False

    if isinstance(val, ast.Name):
        name = val.id
        if any(ind in name.lower() for ind in PREPROCESSING_NAME_HINTS):
            return True
        last = _find_last_assignment(root, name, call_line)
        if isinstance(last, ast.Assign) and _value_is_preprocessed(last.value, call):
            return True
        for n in ast.walk(root):
            if isinstance(n, ast.Call) and getattr(n, "lineno", 0) < call_line:
                if isinstance(n.func, ast.Attribute):
                    recv = n.func.value
                    if isinstance(recv, ast.Name) and recv.id == name:
                        if any(p in n.func.attr.lower() for p in PREPROCESSING_FUNC_NAMES):
                            return True
        return False

    if isinstance(val, ast.Dict):
        for v in val.values:
            if _value_is_preprocessed(v, call):
                return True
        return False

    if isinstance(val, (ast.List, ast.Tuple)):
        for elt in val.elts:
            if _value_is_preprocessed(elt, call):
                return True
        return False

    return False

PREPROCESSING_FUNC_NAMES = {
    'resize', 'resize_and_crop', 'thumbnail', 'imencode', 'imwrite', 'save',
    'crop', 'downscale', 'downscale_image', 'compress', 'optimize', 'encode', 'getvalue', 'tobytes',
}
PREPROCESSING_NAME_HINTS = {
    'resized', 'img_resized', 'small_image', 'small', 'cropped', 'processed',
    'optimized', 'compressed', 'buffer', 'buf', 'thumbnail', 'thumb', 'img_bytes', 'image_bytes',
}

def _call_indicates_preprocessing(call: ast.AST) -> bool:
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    fname = ''
    if isinstance(func, ast.Name):
        fname = func.id.lower()
    elif isinstance(func, ast.Attribute):
        fname = getattr(func, 'attr', '').lower()
    # détection simple sur le nom de la fonction/méthode
    if fname and any(pref in fname for pref in PREPROCESSING_FUNC_NAMES):
        return True
    # détecter imencode(...) qui retourne buffer (cv2.imencode)
    if fname and 'imencode' in fname:
        return True
    return False

def hasParameterSet(node: ast.AST, name: str) -> bool:
    """
    Retourne True si le paramètre `name` est explicitement fourni à l'appel `node`
    (mot clé direct ou via **kwargs dict littéral ou variable).
    Sinon False.
    """
    if not isinstance(node, ast.Call):
        return False
    # 1) Mot clé direct: foo(..., name=...)
    for kw in node.keywords:
        if kw.arg == name:
            return True
    # 2) Via **kwargs
    for kw in node.keywords:
        if kw.arg is None:
            val = kw.value
            # **{ "temperature": ..., "top_p": ... }
            if isinstance(val, ast.Dict):
                if _dict_has_key_str(val, name):
                    return True
            # **params où params est un dict défini plus haut
            if isinstance(val, ast.Name):
                root = _get_root(node)
                call_line = getattr(node, "lineno", float("inf"))
                d = _find_last_dict_assignment(root, val.id, call_line)
                if isinstance(d, ast.Dict) and _dict_has_key_str(d, name):
                    return True
    # On ne prouve pas la présence de `name`
    return False

def hasOverspecifiedSampling(node: ast.AST) -> bool:
    """
    Smell: sur spécification des paramètres de sampling.
    True si un appel LLM combine une température explicite
    avec top_p ou top_k.
    """
    if not isinstance(node, ast.Call):
        return False
    # 1) Cas direct sur l'appel LLM
    if isLLMCall(node) and hasParameterSet(node, "temperature"):
        if hasParameterSet(node, "top_p") or hasParameterSet(node, "top_k"):
            return True
    # 2) Cas où la config vient d'un client.with_options(...) englobant
    parent = getattr(node, "parent", None)
    while isinstance(parent, ast.AST):
        if isinstance(parent, ast.With):
            for item in parent.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute) and ctx.func.attr == "with_options":
                    if hasParameterSet(ctx, "temperature") and (
                        hasParameterSet(ctx, "top_p") or hasParameterSet(ctx, "top_k")
                    ):
                        return True
        parent = getattr(parent, "parent", None)
    return False

def _get_enclosing_scope(node: ast.AST) -> Optional[ast.AST]:
    """
    Retourne le bloc englobant (FunctionDef, AsyncFunctionDef ou Module)
    dans lequel se trouve le noeud.
    """
    parent = getattr(node, "parent", None)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        parent = getattr(parent, "parent", None)
    return parent


def hasMultiUserContext(node: ast.AST) -> bool:
    """
    Détecte un contexte multi utilisateur autour de l appel.
    Approximations statiques
    - présence de request.user.id
    - présence de current_user.id
    - accès à session["user_id"] ou session["uid"]
    """
    if not isinstance(node, ast.Call):
        return False

    scope = _get_enclosing_scope(node)
    if scope is None:
        return False

    for n in ast.walk(scope):
        # request.user.id ou variantes proches
        if isinstance(n, ast.Attribute) and n.attr == "id":
            base = n.value
            # request.user.id
            if isinstance(base, ast.Attribute) and base.attr == "user":
                recv = base.value
                if isinstance(recv, ast.Name) and recv.id in {"request"}:
                    return True
            # current_user.id ou user.id
            if isinstance(base, ast.Name) and base.id in {"current_user", "user"}:
                return True

        # session["user_id"] ou session["uid"]
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "session":
            sl = n.slice
            key = None
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                key = sl.value
            elif isinstance(sl, ast.Index) and isinstance(sl.value, ast.Constant) and isinstance(sl.value.value, str):
                key = sl.value.value
            if key in {"user_id", "uid"}:
                return True

    return False


def _dict_has_user_identifier(d: ast.Dict, root: ast.AST, call_line: int, depth: int = 4) -> bool:
    """
    Cherche des indices de propagation utilisateur dans un dict
    - clés user, user_id, end_user_id
    - clé metadata contenant récursivement une clé user_id
    """
    if depth <= 0:
        return False

    user_keys = {"user", "user_id", "end_user_id", "endUserId"}

    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            key = k.value
            if key in user_keys:
                return True
            if key == "metadata":
                if isinstance(v, ast.Dict):
                    if _dict_has_user_identifier(v, root, call_line, depth - 1):
                        return True
                elif isinstance(v, ast.Name):
                    meta_dict = _find_last_dict_assignment(root, v.id, call_line)
                    if isinstance(meta_dict, ast.Dict) and _dict_has_user_identifier(meta_dict, root, call_line, depth - 1):
                        return True

    return False


def hasUserAttribution(node: ast.AST) -> bool:
    """
    Détecte si l appel LLM attribue explicitement un identifiant utilisateur
    - paramètre user
    - paramètre metadata avec user_id ou équivalent
    - présence de ces clés dans un dict passé via **kwargs
    """
    if not isinstance(node, ast.Call):
        return False

    root = _get_root(node)
    call_line = getattr(node, "lineno", 10**9)

    # Cas simple paramètre user=...
    if hasParameterSet(node, "user"):
        return True

    # Paramètre metadata=...
    metadata_val = _kw_value(node, "metadata")
    if isinstance(metadata_val, ast.Dict):
        if _dict_has_user_identifier(metadata_val, root, call_line):
            return True
    elif isinstance(metadata_val, ast.Name):
        meta_dict = _find_last_dict_assignment(root, metadata_val.id, call_line)
        if isinstance(meta_dict, ast.Dict) and _dict_has_user_identifier(meta_dict, root, call_line):
            return True

    # Cas via **kwargs où le dict contient user ou metadata avec user_id
    for kw in node.keywords:
        if kw.arg is None:
            val = kw.value
            # **{ "user": ..., "user_id": ... }
            if isinstance(val, ast.Dict):
                if _dict_has_user_identifier(val, root, call_line):
                    return True
            # **params où params est un dict défini plus haut
            elif isinstance(val, ast.Name):
                d = _find_last_dict_assignment(root, val.id, call_line)
                if isinstance(d, ast.Dict) and _dict_has_user_identifier(d, root, call_line):
                    return True

    return False

def isHuggingFacePipelineConstructor(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    is_pipeline = (isinstance(fn, ast.Name) and fn.id == "pipeline") or \
                  (isinstance(fn, ast.Attribute) and fn.attr == "pipeline")
    if not is_pipeline:
        return False
    if node.args and isinstance(node.args[0], ast.Constant) and str(node.args[0].value) == "text-generation":
        return True
    return False

def isLLMCallRequiringTemperature(node: ast.AST) -> bool:
    """
    Même logique que isLLMCall, mais on ne considère pas le constructeur HF pipeline
    comme un call où temperature est attendue.
    """
    if not isLLMCall(node):
        return False
    if isHuggingFacePipelineConstructor(node):
        return False
    return True

def rule_R26(ast_node):
    import ast
    add_parent_info(ast_node)
    #set_deterministic_flag(ast_node)
    # "LLM Version Pinning Not Explicitly Set"
    variable_ops = gather_scale_sensitive_ops(ast_node)
    scaled_vars = gather_scaled_vars(ast_node)
    problems = {}
    for sub in ast.walk(ast_node):
        if ((isModelVersionedLLMCall(sub) and hasNoModelVersionPinning(sub))):
            line = getattr(sub, 'lineno', '?')
            if line != '?':
                problems[line] = sub
    for line, node in problems.items():
        report_with_line("LLM call without model version pinning at line {lineno}", node)
