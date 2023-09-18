# @title Metrics Calculation Method
def calculate_metrics(df, prefix, actual_col, expected_col, unknown_phrase):
    """
    Calculates precision, recall, and F1 for a completed eval table

    Args:
    - df(DataFrame): The completed eval table.
    - prefix (str): What to prepend to data columns
    - actual_col(str): The name of the predicted label column
    - expected_col(str): The name of the ground-truth label column
    - unkown_phrase(str): What the model outputs when it doesn't know

    Returns
    - dict: The performance metrics as a dictionary

    """
    metrics = {}
    df[f"{prefix}_TP"] = df.apply(
        lambda row: 1
        if (row[actual_col] != unknown_phrase)
        and (row[actual_col] in row[expected_col])
        else 0,
        axis=1,
    )
    df[f"{prefix}_FP"] = df.apply(
        lambda row: 1
        if (row[actual_col] != unknown_phrase)
        and (row[actual_col] not in row[expected_col])
        else 0,
        axis=1,
    )
    df[f"{prefix}_TN"] = df.apply(
        lambda row: 1
        if (row[actual_col] == unknown_phrase)
        and (row[actual_col] in row[expected_col])
        else 0,
        axis=1,
    )
    df[f"{prefix}_FN"] = df.apply(
        lambda row: 1
        if (row[actual_col] == unknown_phrase)
        and (row[actual_col] not in row[expected_col])
        else 0,
        axis=1,
    )
    metrics["Trial Name"] = prefix
    metrics["Overall Precision"] = df[f"{prefix}_TP"].sum() / (
        df[f"{prefix}_TP"].sum() + df[f"{prefix}_FP"].sum()
    )
    metrics["Overall Recall"] = df[f"{prefix}_TP"].sum() / (
        df[f"{prefix}_TP"].sum() + df[f"{prefix}_FN"].sum()
    )
    metrics["Overall F1 Score"] = (
        2
        * (metrics["Overall Precision"] * metrics["Overall Recall"])
        / (metrics["Overall Precision"] + metrics["Overall Recall"])
    )

    return metrics
