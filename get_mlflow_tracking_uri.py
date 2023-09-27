from zenml.client import Client

client = Client()
last_run = client.get_pipeline("Training_pipeline").last_run
trainer_step = last_run.get_step("decision_tree_trainer")
tracking_url = trainer_step.metadata.get("experiment_tracker_url")
print(tracking_url.value)