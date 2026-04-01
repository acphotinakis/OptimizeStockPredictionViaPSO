import typer
from data.data_ingestion import AlpacaIngestor
from src.optimization.pso import ImprovedPSO

app = typer.Typer()


@app.command()
def ingest(tickers: str = "SPY,AAPL"):
    """Fetch raw OHLCV data from Alpaca."""
    ingestor = AlpacaIngestor()
    ingestor.run(tickers.split(","))


@app.command()
def optimize(iterations: int = 50, particles: int = 20):
    """Run PSO hyperparameter optimization."""
    pso = ImprovedPSO(iterations=iterations, n_particles=particles)
    best_config = pso.search()
    print(f"Global Best Found: {best_config}")


if __name__ == "__main__":
    app()
