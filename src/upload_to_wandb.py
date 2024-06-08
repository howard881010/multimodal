import wandb

wandb.init(project="finetune")

art = wandb.Artifact("qlora", type="model")

art.add_dir("finetune/outputs/qlora-out")

wandb.log_artifact(art)

wandb.finish()