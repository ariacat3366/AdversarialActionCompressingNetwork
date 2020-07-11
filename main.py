import argparse

import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["aacn", "ppo_aacn", "ppo_discrete", "ppo_continuous", "test"],
        default="test")
    args = parser.parse_args()

    if args.type == "aacn":
        trainer.train_aacn()
    elif args.type == "ppo_aacn":
        trainer.train_ppo_aacn()
    elif args.type == "ppo_discrete":
        trainer.train_ppo_discrete()
    elif args.type == "ppo_continuous":
        trainer.train_ppo_continuous()
    elif args.type == "test":
        trainer.train_aacn()


if __name__ == "__main__":
    main()