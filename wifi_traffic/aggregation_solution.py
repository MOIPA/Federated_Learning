import torch

torch.manual_seed(0)

def weighted_average_aggregation(models_to_aggregate, reputations, alpha=3):
    print("Non-linear reputation used")
    total_reputation = sum([r**alpha for r in reputations.values()])
    weighted_models = []

    for model_dict in models_to_aggregate:
        for drone_id, local_model in model_dict.items():
            weight = (reputations[drone_id]**alpha) / total_reputation

            weighted_model = {
                k: v * weight for k, v in local_model.state_dict().items()
            }

            weighted_models.append(weighted_model)

    # 通过加权平均聚合本地模型
    aggregated_model = {}
    for model in weighted_models:
        for k, v in model.items():
            if k not in aggregated_model:
                aggregated_model[k] = v
            else:
                aggregated_model[k] += v

    return aggregated_model



def average_aggregation(models_to_aggregate):
    torch.manual_seed(0)
    print(" average_aggregation used")
    num_models = len(models_to_aggregate)

    aggregated_model = {}

    for model_dict in models_to_aggregate:
        for _, local_model in model_dict.items():
            if not aggregated_model:
                aggregated_model = {
                    k: v / num_models for k, v in local_model.state_dict().items()
                }
            else:
                for k, v in local_model.state_dict().items():
                    aggregated_model[k] += v / num_models

    return aggregated_model
