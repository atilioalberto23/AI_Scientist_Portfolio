import torch


def gradient_norm(model):
    """
    Global L2 norm of all available gradients.

    Must be called after loss.backward().
    """

    total_squared = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        total_squared += (
            grad.pow(2)
            .sum()
            .item()
        )

    return total_squared ** 0.5

def gradient_stats(model, eps=1e-12):
    """
    Gradient and parameter statistics for each
    trainable tensor in the model.
    """

    stats = []

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()
        weights = parameter.detach()

        grad_norm_value = grad.norm().item()
        weight_norm_value = weights.norm().item()

        ratio = (
            grad_norm_value
            / (weight_norm_value + eps)
        )

        stats.append({
            "parameter": name,
            "grad_norm": grad_norm_value,
            "weight_norm": weight_norm_value,
            "grad_weight_ratio": ratio,
            "gradient_finite":
                torch.isfinite(grad)
                .all()
                .item()
        })

    return stats

def has_nonfinite_gradients(model):
    """
    Return True if any gradient contains NaN or Inf.
    """

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():

            return True

    return False

def has_nonfinite_parameters(model):
    """
    Return True if any model parameter contains NaN or Inf.
    """

    for parameter in model.parameters():

        if not torch.isfinite(
            parameter
        ).all():

            return True

    return False