from dataset.ajoumc_anemia import load_ajoumc_seghgb_dataset


def load_dataset(args, cv_fold=5, seed=42, w_filename=False):
    if args.dataset.lower().startswith("ajoumc"):
        return load_ajoumc_seghgb_dataset(args, cv_fold, seed, w_filename)
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported!")
