import os
import json
import time
import random
import numpy as np
from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras import backend as K

random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)

population_size = 30
num_generations = 40
mutation_rate = 0.10
num_epochs = 100
n_channels = 2
input_shape = (128, 128, 128, n_channels)
num_classes = 4

# fitness weights 
WD, WI = 0.5, 0.5                
LMB_PARAMS, LMB_FLOPS, LMB_LAT = 0.20, 0.20, 0.00   # penalties on normalized params/flops/latency


# search space
search_space = {
    'num_layers': [2, 3, 4],                       
    'filters': (16, 127),                          
    'kernel_size': [3, 4, 5, 6],                   
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],  
    'pooling_type': ['avg', 'max'],                 
    'combine_strategy': ['add', 'concat'],          
    'dilation_rate': [[1,2,3], [1,3,5], [1,5,7]],   
    'use_1x1_fusion': [True, False],               
    'dropout_rate': (0.10, 0.30),                 
    'uncertainty_weight': (1.00, 1.50),                    
}

def _sample_float(lo, hi, round2=False):
    x = random.uniform(float(lo), float(hi))
    if round2:
        x = round(x, 2)
        if x < lo: x = lo
        if x > hi: x = hi
    return x

def sample_individual(space: Dict[str, Any]) -> Dict[str, Any]:
    loF, hiF = space['filters']
    loR, hiR = space['dropout_rate']
    loA, hiA = space['uncertainty_weight']
    indiv = {
        'num_layers': random.choice(space['num_layers']),
        'filters': random.randint(int(loF), int(hiF)),
        'kernel_size': random.choice(space['kernel_size']),
        'pooling_type': random.choice(space['pooling_type']),
        'combine_strategy': random.choice(space['combine_strategy']),
        'dilation_rate': random.choice(space['dilation_rate']),
        'use_1x1_fusion': random.choice(space['use_1x1_fusion']),
        'activation': random.choice(space['activation']),
        'dropout_rate': _sample_float(loR, hiR, round2=True),
        'uncertainty_weight': _sample_float(loA, hiA, round2=True),
    }
    return indiv

def mutate(indiv: Dict[str, Any], space: Dict[str, Any], p: float = 0.1) -> Dict[str, Any]:
    child = dict(indiv)
    if random.random() < p:
        child['num_layers'] = random.choice(space['num_layers'])
    if random.random() < p:
        loF, hiF = space['filters']
        child['filters'] = random.randint(int(loF), int(hiF))
    if random.random() < p:
        child['kernel_size'] = random.choice(space['kernel_size'])
    if random.random() < p:
        child['pooling_type'] = random.choice(space['pooling_type'])
    if random.random() < p:
        child['combine_strategy'] = random.choice(space['combine_strategy'])
    if random.random() < p:
        child['dilation_rate'] = random.choice(space['dilation_rate'])
    if random.random() < p:
        child['use_1x1_fusion'] = random.choice(space['use_1x1_fusion'])
    if random.random() < p:
        child['activation'] = random.choice(space['activation'])
    if random.random() < p:
        loR, hiR = space['dropout_rate']
        child['dropout_rate'] = _sample_float(loR, hiR, round2=True)
    if random.random() < p:
        loA, hiA = space['uncertainty_weight']
        child['uncertainty_weight'] = _sample_float(loA, hiA, round2=True)
    return child

def crossover(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(p1.keys())
    return {k: (p1[k] if random.random() < 0.5 else p2[k]) for k in keys}

def crossover_and_mutate(p1: Dict[str, Any], p2: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
    return mutate(crossover(p1, p2), search_space, p=mutation_rate)


import numpy as np
from scipy.spatial.distance import cdist

# base binary-mask metrics
def dice_np(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> float:
    yt = (y_true > 0.5).astype(np.float32)
    yp = (y_pred > thr).astype(np.float32)
    inter = (yt * yp).sum()
    return float((2.0 * inter + eps) / (yt.sum() + yp.sum() + eps))

def iou_np(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> float:
    yt = (y_true > 0.5).astype(np.uint8)
    yp = (y_pred > thr).astype(np.uint8)
    inter = np.logical_and(yt, yp).sum()
    union = np.logical_or(yt, yp).sum()
    return float((inter + eps) / (union + eps))

def hd95_np(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.5) -> float:
    yt = (y_true > 0.5).astype(np.uint8)
    yp = (y_pred > thr).astype(np.uint8)
    pts_t = np.array(np.where(yt == 1)).T
    pts_p = np.array(np.where(yp == 1)).T
    if pts_t.size == 0 or pts_p.size == 0:
        return 0.0
    D = cdist(pts_t, pts_p, metric='euclidean')
    hd_tp = np.percentile(np.min(D, axis=1), 95)
    hd_pt = np.percentile(np.min(D, axis=0), 95)
    return float(max(hd_tp, hd_pt))

# generic class-wise
def classwise_metrics_generic(
    y_true_onehot: np.ndarray,
    y_pred_onehot: np.ndarray,
    *,
    class_names: list = None,       # e.g., ['bg','c1','c2','c3']
    bg_index: int | None = 0,       # set None to include all in macro
    composites: dict | None = None, # e.g., {'wt':[1,2,3], 'tc':[1,3]}
    macro_keys: list | None = None, # names to average for macro; if None, use native excluding bg_index
    thr: float = 0.5
) -> dict:
    """
    y_* shapes is [b,h,w,c] or [b,h,w,d,c] with one-hot channels.
    composites is a dict from new-name -> list of native channel indices to union.
    macro_keys if provided is the list of names to average for macro (e.g., ['wt','tc','et']).
    returns a dict with macro and per-class metrics (dice, iou, hd95).
    """
    if y_true_onehot.ndim not in (4, 5) or y_pred_onehot.ndim != y_true_onehot.ndim:
        raise ValueError("expected [b,h,w,c] or [b,h,w,d,c] one-hot tensors")

    C = y_true_onehot.shape[-1]
    if class_names is None:
        class_names = [f"class{c}" for c in range(C)]

    # native per-channel
    dsc_pc = {}
    iou_pc = {}
    hd_pc  = {}
    for c in range(C):
        name = class_names[c]
        dsc_pc[name] = dice_np(y_true_onehot[..., c], y_pred_onehot[..., c], thr=thr)
        iou_pc[name] = iou_np(y_true_onehot[..., c], y_pred_onehot[..., c], thr=thr)
        hd_pc[name]  = hd95_np(y_true_onehot[..., c], y_pred_onehot[..., c], thr=thr)

    # composites (unions of native channels)
    if composites:
        for new_name, idx_list in composites.items():
            yt = (np.sum([y_true_onehot[..., i] for i in idx_list], axis=0) > 0.5).astype(np.float32)
            yp = (np.sum([y_pred_onehot[..., i] for i in idx_list], axis=0)).astype(np.float32)
            dsc_pc[new_name] = dice_np(yt, yp, thr=thr)
            iou_pc[new_name] = iou_np(yt, yp, thr=thr)
            hd_pc[new_name]  = hd95_np(yt, yp, thr=thr)

    # macro selection
    if macro_keys is not None:
        keys_for_macro = macro_keys
    else:
        # default is native classes excluding background
        keys_for_macro = [class_names[i] for i in range(C) if (bg_index is None or i != bg_index)]

    dice_macro = float(np.mean([dsc_pc[k] for k in keys_for_macro])) if keys_for_macro else 0.0
    iou_macro  = float(np.mean([iou_pc[k] for k in keys_for_macro])) if keys_for_macro else 0.0
    hd95_macro = float(np.mean([hd_pc[k] for k in keys_for_macro])) if keys_for_macro else 0.0

    return {
        'dice': dice_macro,
        'iou': iou_macro,
        'hd95': hd95_macro,
        'dice_per_class': dsc_pc,
        'iou_per_class': iou_pc,
        'hd95_per_class': hd_pc
    }



# model & complexity utilities
def best_model(input_shape,
               num_layers,
               dilation_rate,
               filters,
               kernel_size,
               activation,
               dropout_rate,
               pooling_type,
               combine_strategy,
               use_1x1_fusion,
               uncertainty_weight,
               num_classes):
    """
    placeholder that should build your evosegnet variant.
    replace with your actual builder. below we put a tiny 3d model
    so the script structure is complete; swap it out in your codebase.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    f = int(filters)
    act = tf.keras.layers.Activation(activation)

    # simple encoder blocks
    for i in range(int(num_layers)):
        x = tf.keras.layers.Conv3D(f, kernel_size, padding='same')(x)
        x = act(x)
        if pooling_type == 'avg':
            x = tf.keras.layers.AveragePooling3D()(x)
        else:
            x = tf.keras.layers.MaxPooling3D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        f = min(f * 2, 512)

    # bottleneck
    x = tf.keras.layers.Conv3D(f, kernel_size, padding='same')(x)
    x = act(x)

    # simple decoder blocks
    for i in range(int(num_layers)):
        f = max(f // 2, int(filters))
        x = tf.keras.layers.Conv3DTranspose(f, kernel_size, strides=2, padding='same')(x)
        x = act(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Conv3D(num_classes, 1, padding='same', activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=[])
    return model

def compute_params_m(model: tf.keras.Model) -> float:
    return float(model.count_params() / 1e6)

def compute_flops_g(model: tf.keras.Model, sample_shape: Tuple[int, ...]) -> float:
    """
    lightweight proxy; accurate flops need tf profiler. we return 0 and
    rely on normalization to null it out unless you replace this with your estimator.
    """
    return 0.0

def estimate_latency_ms(model: tf.keras.Model, sample: np.ndarray = None, iters: int = 5) -> float:
    """
    naive latency estimate: forward passes on a small batch.
    swap with your real measurement if needed.
    """
    if sample is None:
        sample = np.zeros((1,) + model.input_shape[1:], dtype=np.float32)
    # warmup
    _ = model.predict(sample, verbose=0)
    t0 = time.time()
    for _ in range(iters):
        _ = model.predict(sample, verbose=0)
    dt = (time.time() - t0) / iters
    return float(dt * 1000.0)


# evaluation / fitness
def minmax_norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (x - lo) / (hi - lo)

def evaluate_fitness(indiv: Dict[str, Any],
                     train_gen,
                     val_gen,
                     epochs: int,
                     n_channels: int,
                     gen: int,
                     idx: int) -> Tuple[float, Dict[str, Any]]:
    # 1) build model for this individual
    model = best_model(
        input_shape=input_shape,
        num_layers=indiv['num_layers'],
        dilation_rate=indiv['dilation_rate'],
        filters=indiv['filters'],
        kernel_size=indiv['kernel_size'],
        activation=indiv['activation'],
        dropout_rate=indiv['dropout_rate'],
        pooling_type=indiv['pooling_type'],
        combine_strategy=indiv['combine_strategy'],
        use_1x1_fusion=indiv['use_1x1_fusion'],
        uncertainty_weight=indiv['uncertainty_weight'],
        num_classes=num_classes
    )

    # 2) train
    history = model.fit(train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        verbose=0)

    # 3) eval on validation (per-class then mean, like paper)
    dices, ious, hd95s = [], [], []
    for x_val, y_val in val_gen:
        y_hat = model.predict(x_val, verbose=0)
        # assume one-hot [B, H, W, D, C] or [B, H, W, C]; adapt as needed
        if y_hat.ndim == 5:
            # 3d: [b,h,w,d,c]
            Cc = y_hat.shape[-1]
            for c in range(Cc):
                y_true_c = y_val[..., c]
                y_pred_c = y_hat[..., c]
                dices.append(dice_np(y_true_c, y_pred_c))
                ious.append(iou_np(y_true_c, y_pred_c))
                # compute hd95 per volume (merge batch dims)
                # we collapse to a single volume for hd95 (common in practice)
                yt_vol = (y_true_c > 0.5).astype(np.uint8)
                yp_vol = (y_pred_c > 0.5).astype(np.uint8)
                hd95s.append(hd95_np(yt_vol, yp_vol))
        else:
            # 2d: [b,h,w,c]
            Cc = y_hat.shape[-1]
            for c in range(Cc):
                y_true_c = y_val[..., c]
                y_pred_c = y_hat[..., c]
                dices.append(dice_np(y_true_c, y_pred_c))
                ious.append(iou_np(y_true_c, y_pred_c))
                yt_img = (y_true_c > 0.5).astype(np.uint8)
                yp_img = (y_pred_c > 0.5).astype(np.uint8)
                hd95s.append(hd95_np(yt_img, yp_img))

    dsc_val = float(np.mean(dices)) if len(dices) else 0.0
    iou_val = float(np.mean(ious)) if len(ious) else 0.0
    hd95_val = float(np.mean(hd95s)) if len(hd95s) else 0.0

    # 4) complexity
    params_m = compute_params_m(model)
    flops_g = compute_flops_g(model, sample_shape=input_shape)
    latency_ms = estimate_latency_ms(model)

    # 5) normalize complexity within reasonable fixed bounds or track per-gen min/max
    # here we use fixed ranges aligned with your tables; you can replace with per-gen min/max cache.
    params_n = minmax_norm(params_m, 1.0, 64.0)
    flops_n  = minmax_norm(flops_g, 10.0, 60.0)   # if flops_g==0, this becomes negative; clamp to 0
    if flops_g <= 0: flops_n = 0.0
    lat_n    = minmax_norm(latency_ms, 15.0, 40.0)

    # 6) scalarized multi-objective: (dsc + iou) minus penalties
    acc_combo = WD * dsc_val + WI * iou_val
    fitness = acc_combo - (LMB_PARAMS * params_n + LMB_FLOPS * flops_n + LMB_LAT * lat_n)

    metrics = {
        'dice': dsc_val,
        'iou': iou_val,
        'hd95': hd95_val,
        'params': params_m,
        'flops': flops_g,
        'latency': latency_ms
    }
    return fitness, metrics

# selection helper
def select_best_individuals(population: List[Dict[str, Any]],
                            fitness_scores: List[float],
                            num_selections: int) -> List[Dict[str, Any]]:
    sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_pop[:num_selections]

 
# evolutionary search loop
population = [sample_individual(search_space) for _ in range(population_size)]
global_best_fitness = -np.inf
global_best_individual = None
global_best_metrics = None
log = []

for generation in range(num_generations):
    print(f"\nGeneration {generation + 1}/{num_generations}")

    fitness_scores = []
    for individual_index, individual in enumerate(population):
        print(f"  Individual {individual_index + 1}: {individual}")

        fitness, metrics = evaluate_fitness(
            individual,
            training_generator,
            valid_generator,
            num_epochs,
            n_channels,
            generation,
            individual_index
        )
        fitness_scores.append(fitness)

        rec = {
            'generation': generation,
            'individual': individual_index,
            'architecture': individual,
            **metrics,
            'fitness': fitness
        }
        log.append(rec)

        if fitness > global_best_fitness:
            global_best_fitness = fitness
            global_best_individual = individual
            global_best_metrics = metrics
            print(f"    new global best: fitness={fitness:.4f} dsc={metrics['dice']:.4f} iou={metrics['iou']:.4f}")

    # selection
    num_selections = max(2, population_size // 2)
    selected = select_best_individuals(population, fitness_scores, num_selections)

    # reproduction
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected, 2)
        child = crossover_and_mutate(parent1, parent2, mutation_rate)
        new_population.append(child)
    population = new_population

# save logs and best model
with open("nas_log.json", "w") as f:
    json.dump(log, f, indent=2)

print("\nbest individual (global):")
print(global_best_individual)
print("metrics:",
      f"Dice={global_best_metrics['dice']:.4f}, "
      f"IoU={global_best_metrics['iou']:.4f}, "
      f"HD95={global_best_metrics['hd95']:.2f}, "
      f"Params={global_best_metrics['params']:.2f}M, "
      f"FLOPs={global_best_metrics['flops']:.2f}G, "
      f"Latency={global_best_metrics['latency']:.2f}ms")

# Build and save best model
best_net = best_model(
    input_shape=input_shape,
    num_layers=global_best_individual['num_layers'],
    dilation_rate=global_best_individual['dilation_rate'],
    filters=global_best_individual['filters'],
    kernel_size=global_best_individual['kernel_size'],
    activation=global_best_individual['activation'],
    dropout_rate=global_best_individual['dropout_rate'],
    pooling_type=global_best_individual['pooling_type'],
    combine_strategy=global_best_individual['combine_strategy'],
    use_1x1_fusion=global_best_individual['use_1x1_fusion'],
    uncertainty_weight=global_best_individual['uncertainty_weight'],
    num_classes=num_classes
)
best_net.save("best_model_3D_UNet.h5")
print("saved: best_model_3D_UNet.h5")
