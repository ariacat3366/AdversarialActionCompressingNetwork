import argparse
import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        choices=[
                            "all_discrete", "all_continuous", "aacn_discrete",
                            "aacn_continuous", "ppo_discrete", "ppo_continuous"
                        ],
                        default="aacn_discrete")
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
        pass
    elif args.type == "ppo_continuous":
        pass


if __name__ == "__main__":
    main()