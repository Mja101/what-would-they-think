import argparse
from app.data_utils import parse_whatsapp_chat, save_parsed_chat

def main():
    """
    Parses a WhatsApp chat export .txt file and saves the parsed data as a .csv file.

    Uses command-line arguments to specify the input and output file paths.
    Calls the parsing and saving utilities from app.data_utils.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Parse WhatsApp chat export .txt to .csv")
    parser.add_argument('--input', type=str, required=True, help='Path to input .txt file')
    parser.add_argument('--output', type=str, required=True, help='Path to output .csv file')
    args = parser.parse_args()

    print(f"Parsing {args.input}...")
    df = parse_whatsapp_chat(args.input)
    print(f"Found {len(df)} messages. Saving to {args.output}...")
    save_parsed_chat(df, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
