import argparse

import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        choices=[
                            "aacn", "ppo_aacn", "baseline", "ppo_discrete",
                            "ppo_continuous", "test", "trial_training"
                        ],
                        default="test")
    args = parser.parse_args()

    if args.type == "aacn":
        trainer.train_aacn()
    elif args.type == "baseline":
        trainer.train_baseline()
    elif args.type == "ppo_aacn":
        trainer.train_ppo_aacn()
    elif args.type == "ppo_discrete":
        trainer.train_ppo_discrete()
    elif args.type == "ppo_continuous":
        trainer.train_ppo_continuous()
    elif args.type == "test":
        trainer.train_aacn()
    elif args.type == "trial_training":
        for i in range(10):
            print(f"trial: {i}")
            # print(f"train aacn")
            # trainer.train_ppo_aacn(trial=i, seed=i, save_npy=True)
            print(f"train baseline")
            trainer.train_ppo_baseline(trial=i, seed=i, save_npy=True)
            # print(f"train continuous")
            # trainer.train_ppo_continuous(trial=i, seed=i, save_npy=True)
            print(f"")


if __name__ == "__main__":
    main()