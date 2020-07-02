import argparse

import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        choices=[
                            "all_discrete", "all_continuous", "aacn_discrete",
                            "aacn_continuous", "ppo_discrete",
                            "ppo_continuous", "test"
                        ],
                        default="test")
    args = parser.parse_args()

    if args.type == "all_discrete":
        pass
    elif args.type == "all_continuous":
        pass
    elif args.type == "aacn_discrete":
        trainer.train_aacn_discrete()
    elif args.type == "aacn_continuous":
        pass
    elif args.type == "ppo_discrete":
        trainer.train_ppo_discrete()
    elif args.type == "ppo_continuous":
        trainer.train_ppo_continuous()
    elif args.type == "test":
        trainer.train_aacn_discrete()


if __name__ == "__main__":
    main()