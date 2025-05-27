from datetime import datetime
from src.data_loader import generate_context_summary
from src.predictor import predict_home_run_probability

summary = generate_context_summary(
    batter_id=592450,  # Aaron Judge
    pitcher_id=607536,  # Kyle Freeland
    batter_team="NYY",
    pitcher_team="COL",
    game_date=datetime(2025, 5, 24),
    batter_name="Aaron Judge",
    pitcher_name="Kyle Freeland",
    ballpark_name="Coors Field"
)

prediction = predict_home_run_probability(summary)
print(prediction)
