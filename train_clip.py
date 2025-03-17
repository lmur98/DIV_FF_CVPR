import train
from opt import get_opts

def main(hparams):
    # Load NeuralDiff pre-trained model
    system = train.init_model(hparams.ckpt_path_pretrained, dataset=None, hparams=hparams)

    """# Freeze every layer except features prediction layers
    if (hparams.freeze_neuraldiff):
        print("Freezing NeuralDiff layers...")
        system.models["fine"].requires_grad_(False)
        system.models["coarse"].requires_grad_(False)
        for key, value in system.embeddings.items():
            value.requires_grad_(False)"""

    # Train the model
    trainer = train.init_trainer(hparams)
    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_opts()
    print(hparams)
    main(hparams)