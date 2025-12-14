from flower.registry import register_model, list_models, load_latest_model

@app.command()
def register(
    model: Path = typer.Option("artifacts/checkpoints/best_model.pt"),
    model_cfg: Path = typer.Option("configs/model_cnn.yaml"),
    train_cfg: Path = typer.Option("configs/train.yaml"),
    experiment: str = typer.Option(None)
):
    """Register best model to registry"""
    # Load metrics from evaluation
    import json
    with open('artifacts/evaluation/metrics.json') as f:
        metrics = json.load(f)
    
    register_model(model, model_cfg, train_cfg, metrics, experiment)

@app.command()
def list_registry():
    """List all registered models"""
    list_models()