def compute_mse(true_values, predicted_values):
    """Compute Mean Squared Error between true and predicted values."""
    return ((true_values - predicted_values) ** 2).mean().item()