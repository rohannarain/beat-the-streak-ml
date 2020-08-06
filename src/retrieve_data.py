from api_utils import *

arg_parser = argparse.ArgumentParser(description="Run to generate training data from yesterday's games and test data from today's games")
arg_parser.add_argument("--train", help = "Use if you want to generate training data only", action="store_true")
arg_parser.add_argument("--test", help = "Use if you want to generate test data only", action="store_true")
arg_parser.add_argument("--date", help = "Use if you want to generate training data from a date that isn't yesterday or today, entered in MM/DD/YYYY format")
arg_parser.add_argument("--debug", help = "Enable debug mode (print player names as they are processed, and iteration)", action="store_true")
args = arg_parser.parse_args()

if args.train:
    generate_hits_data()
elif args.test:
    generate_hits_data(generate_train_data=False, generate_from_date=today)
elif args.date and args.debug:
    print("DEBUG MODE\n")
    generate_hits_data(generate_train_data=True, generate_from_date=args.date, debug=True)
elif args.date:
    generate_hits_data(generate_train_data=True, generate_from_date=args.date)
else:
    generate_hits_data()
    generate_hits_data(generate_train_data=False, generate_from_date=today)
    # generate_yesterdays_results()
    