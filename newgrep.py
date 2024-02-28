import argparse

def grep(pattern, filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                if pattern in line:
                    print(line, end='')
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


parser = argparse.ArgumentParser(description="Search for a pattern in a file.")
parser.add_argument("pattern", help="Pattern to search for")
parser.add_argument("filename", help="File to search in")
args = parser.parse_args()
grep(args.pattern, args.filename)